"""
Kaskádová extrakční pipeline — jádro řešení vyhodnoceného v kapitole 5 práce.

    PRIMÁRNÍ CESTA   GPT-4.1 · OCR JSON-light · cs_en few-shot RAG (n=2)
    AKCEPTAČNÍ TEST  formát, aritmetika, povinná pole, jistota modelu
    ZÁLOŽNÍ CESTA    gpt-4.1-mini Vision · zero-shot · RAG seznam dodavatelů

Modul je společný pro dávkové vyhodnocení (``scripts/run_ds1_final.py``)
i pro REST API (``src/core/main_pipeline.py``). Obě cesty tak zpracovávají
dokument doslova stejným kódem — kdyby se rozešly, přestala by čísla v práci
odpovídat tomu, co systém v provozu skutečně dělá.

Vstupním bodem pro běžné použití je :func:`extract_invoice`; dávkové
vyhodnocení volá nižší vrstvu (:func:`cascade_extract`) přímo, protože si
řídí vlastní měření a ukládání.
"""
from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
import unicodedata
import warnings
from dataclasses import dataclass
from datetime import date as date_cls
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
import pandas as pd
import pytesseract
from dotenv import load_dotenv
from PIL import Image
from pydantic import Field, create_model
from pytesseract import Output

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from src.core.ocr_engine.tesseract_setup import nastav_tesseract

# Cesta k Tesseractu se zjišťuje za běhu — na Linuxu, macOS i v kontejneru bývá
# na PATH, na Windows v Program Files. Lze ji vynutit proměnnou TESSERACT_CMD.
TESSERACT_PATH = nastav_tesseract()

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURACE
# ─────────────────────────────────────────────────────────────────────────────


OCR_DPI            = 200
VISION_DPI         = 150        # nižší DPI = menší tokeny pro Vision
FIXED_OCR_CONFIG   = "--oem 3 --psm 4 -l ces"
FIXED_OCR_CONF_MIN = 30

# PRIMARY: GPT-4.1 plný, text, cs_en few-shot RAG n=2 (dle sekce 4.3 práce)
PRIMARY_MODEL = {
    "model_id":  "gpt-4.1-2025-04-14",
    "label":     "GPT-4.1 Apr-25",
    "price_in":  2.000,   # USD / 1M input tokens
    "price_out": 8.000,
}
# FALLBACK: gpt-4.1-mini, Vision, standard (dle sekce 4.3 práce)
VISION_MODEL = {
    "model_id":  "gpt-4.1-mini",
    "label":     "GPT-4.1-mini Vision",
    "price_in":  0.400,
    "price_out": 1.600,
}

# ── Kritéria eskalace ────────────────────────────────────────────────────────
#
# Logika: eskaluj pokud JAKÁKOLI z podmínek platí:
#   1. API error — primární model selhal úplně
#   2. Chybí klíčové povinné pole (viz _CASCADE_REQUIRED)
#   3. null_rate přesáhne práh specifický pro komoditu
#
# Pozor: každá komodita má jiný počet volitelných polí (opt.).
# S jednoduchým prahem 0.30 by VN eskalovala vždy — 20 z 26 polí je opt.
# → normální null_rate pro VN je 0.77 (vše opt. null). To není OCR selhání!
#
# Prahy jsou odvozeny jako: (opt_count / total_count) + 0.12 (bezpečnostní okraj).
# Tj. eskaluj pokud null_rate ukazuje, že null jsou nejen opt. pole, ale i povinná.
#
#  komodita       total  opt   opt/total  práh (opt+0.12)
#  elektrina_nn     14    4     0.286      0.40
#  elektrina_vn     26   20     0.769      0.85
#  plyn_mo          25   12     0.480      0.60
#  plyn_vo          25   11     0.440      0.55
#  teplo            17    5     0.294      0.41  ← cold_water_consumption změněno na opt.
#  voda             15    5     0.333      0.45

_CASCADE_NULL_THRESHOLDS: dict[str, float] = {
    "elektrina_nn": 0.40,
    "elektrina_vn": 0.85,
    "plyn_mo":      0.60,
    "plyn_vo":      0.55,
    "teplo":        0.41,
    "voda":         0.45,
}
_CASCADE_NULL_THRESHOLD_DEFAULT = 0.40  # fallback pro neznámou komoditu

# Cross-field constraints: alespoň jedno pole ze skupiny musí být non-null.
# Každá komodita může mít více skupin; eskaluj pokud CELÁ skupina je null.
#
#   elektrina_nn: VT nebo NT — alespoň jedna spotřeba musí být přítomna.
#     - jednosázkový odběr (D01): VT non-null, NT null → OK
#     - dvousázkový odběr (D02): oba non-null → OK
#     - oba null → OCR/extrakce selhala → eskalace
_CROSS_FIELD_AT_LEAST_ONE: dict[str, list[list[str]]] = {
    "elektrina_nn": [
        ["consumption_high_tariff", "consumption_low_tariff"],
    ],
}

# Klíčová povinná pole jejichž absence okamžitě spouští eskalaci
_CASCADE_REQUIRED: dict[str, list[str]] = {
    # elektrina_nn: consumption_high_tariff odstraněno — pokryto cross-field pravidlem níže
    "elektrina_nn": ["invoice_number", "amount_inc_vat", "period_from", "period_to"],
    "elektrina_vn": ["invoice_number", "supply_consumption", "supply_charge",
                     "period_from", "period_to"],
    "plyn_mo":      ["invoice_number", "amount_inc_vat",
                     "consumption_mwh", "consumption_m3", "period_from"],
    "plyn_vo":      ["invoice_number", "amount_inc_vat",
                     "consumption_mwh", "consumption_m3", "period_from"],
    "teplo":        ["invoice_number", "amount_inc_vat",
                     "consumption_gj", "period_from"],
    "voda":         ["invoice_number", "amount_inc_vat",
                     "consumption_vodne_m3", "period_from"],
}

MAX_TOKENS_TEXT   = 2048
MAX_TOKENS_VISION = 3500   # prostor pro cot reasoning + JSON
TEMPERATURE       = 0.0
USD_TO_CZK        = 23.5
VISION_MAX_PAGES  = 999    # všechny strany dokumentu

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EXCLUDE_COMMODITIES: list[str] = []

try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except ImportError:
    openai_client = None

# ── RAG few-shot retriever (optional) ────────────────────────────────────────
# Aktivuje se automaticky pokud existuje data/retrieval/faiss.index.
# Pokud index chybí, vision fallback funguje beze změny.
try:
    sys.path.insert(0, str(ROOT))
    from src.core.retrieval import FewShotRetriever as _FewShotRetriever
    _retriever = _FewShotRetriever()
    print(f"[RAG] Retriever loaded: {_retriever._index.ntotal} vectors", flush=True)
except Exception as _rag_err:
    _retriever = None
    print(f"[RAG] Retriever not available ({_rag_err.__class__.__name__}: {_rag_err})", flush=True)

# ── Fix 3: deterministic amount_inc_vat correction ───────────────────────────
# Recompute amount_inc_vat = amount_ex_vat × (1 + VAT_rate/100) from the
# commodity/date VAT rate (parsed from data/dph.csv) and replace the extracted
# value when it disagrees by more than the tolerance. Toggle for before/after.
sys.path.insert(0, str(ROOT))

ENABLE_VAT_INC_CORRECTION = True    # Fix 3 on/off flag
VAT_INC_TOLERANCE_CZK     = 1.0
OUTPUT_TAG                = "v3"     # results/summary/audit file suffix for the rerun
_VAT_AUDIT: list[dict]    = []       # collected replacement records for the audit CSV

# ─────────────────────────────────────────────────────────────────────────────
# FIELD REGISTRY — přesné shody s GT CSV sloupcemi
# ─────────────────────────────────────────────────────────────────────────────
#
# Každý field dict:
#   field_name  — přesně odpovídá sloupci v GT CSV
#   field_type  — "string" | "date" | "float"
#   is_required — 1 pokud je pole označeno * v sekci 2.2 práce
#   label_cs    — český název pro prompt
#   label_en    — anglický název pro Vision prompt
#   anchor_cs   — hledané vzory v OCR textu (česky)
#   anchor_en   — hledané vzory ve Vision (anglicky)
#   unit        — fyzikální jednotka (pro prompt)
#   note        — upřesnění / anti-confusion rule
# ─────────────────────────────────────────────────────────────────────────────

# --- Společná pole (11 polí, přítomná v 5 z 6 komodit; VN nemá amounts/IČO) ---

_F_INVOICE_NUMBER = {
    "field_name": "invoice_number", "field_type": "string", "is_required": 1,
    "label_cs": "Číslo faktury", "label_en": "Invoice number",
    "anchor_cs": (
        "Číslo faktury, Č. faktury, Faktura č., Doklad č., Variabilní symbol, "
        "Číslo dokladu, Číslo vyúčtování, Faktura číslo, VS:, Var. symbol, "
        "Číslo zálohy, Záloha č., Číslo zakázky"
    ),
    "anchor_en": "Invoice number, Document number, Variable symbol, Billing number",
    "unit": "",
    "note": (
        "Alfanumerická sekvence identifikující fakturu — hledej v záhlaví dokumentu. "
        "Preferuj pole označené 'Číslo faktury', 'Č. faktury', 'Číslo dokladu', 'Doklad č.' — "
        "to je primární identifikátor, nikoli Variabilní symbol (ten je platební reference). "
        "Vrať celé číslo včetně číslic a písmen (příklad: '1125007758P')."
    ),
}
_F_PERIOD_FROM = {
    "field_name": "period_from", "field_type": "date", "is_required": 1,
    "label_cs": "Fakturační/odběrné období od", "label_en": "Billing period from",
    "anchor_cs": (
        "Odběrné období od, Fakturační období od, Zúčtovací období od, Období od, "
        "Vyúčtování za období DD.MM.YYYY -, Dodávka za období, "
        "Mimořádné vyúčtování za období DD.MM.YYYY"
    ),
    "anchor_en": "Billing period from, Period from, Service period start",
    "unit": "YYYY-MM-DD",
    "note": (
        "Začátek fakturačního/odběrného období. "
        "Hledej jako rozsah 'DD.MM.YYYY - DD.MM.YYYY' v záhlaví faktury nebo v řádku "
        "'Vyúčtování za období' / 'Zúčtovací období'. "
        "POZOR: period_from NENÍ Datum vystavení, DUZP, ani Datum splatnosti. "
        "period_from je vždy před period_to a před issue_date."
    ),
}
_F_PERIOD_TO = {
    "field_name": "period_to", "field_type": "date", "is_required": 1,
    "label_cs": "Fakturační/odběrné období do", "label_en": "Billing period to",
    "anchor_cs": (
        "Odběrné období do, Fakturační období do, Zúčtovací období do, Období do, "
        "- DD.MM.YYYY (konec intervalu), Konec zúčtovacího období"
    ),
    "anchor_en": "Billing period to, Period to, Service period end",
    "unit": "YYYY-MM-DD",
    "note": (
        "Konec fakturačního/odběrného období. "
        "Typicky odpovídá DUZP (datu uskutečnění zdanitelného plnění). "
        "NENÍ: Datum splatnosti ani issue_date. "
        "period_to je vždy po period_from a před nebo roven issue_date."
    ),
}
_F_ISSUE_DATE = {
    "field_name": "issue_date", "field_type": "date", "is_required": 0,
    "label_cs": "Datum vystavení", "label_en": "Issue date",
    "anchor_cs": "Datum vystavení, Datum dokladu, Vystaveno dne",
    "anchor_en": "Issue date, Date of issue",
    "unit": "YYYY-MM-DD", "note": "Invoice issue date. NOT the due date.",
}
_F_DUE_DATE = {
    "field_name": "due_date", "field_type": "date", "is_required": 0,
    "label_cs": "Datum splatnosti", "label_en": "Due date",
    "anchor_cs": (
        "Datum splatnosti, Splatnost, Uhradit do, Splatit do, "
        "Doporučené datum úhrady, Zaplaťte do, Do data splatnosti, "
        "Uhraďte do, Termín splatnosti"
    ),
    "anchor_en": "Due date, Payment due, Pay by, Recommended payment date",
    "unit": "YYYY-MM-DD",
    "note": (
        "Datum splatnosti faktury. "
        "Vždy POZDĚJŠÍ než Datum vystavení (issue_date). "
        "Typicky 14–30 dní po vystavení. "
        "NENÍ issue_date, NENÍ DUZP, NENÍ period_to."
    ),
}
_F_TAX_POINT_DATE = {
    "field_name": "tax_point_date", "field_type": "date", "is_required": 0,
    "label_cs": "Datum UZP (uskutečnění zdanitelného plnění)", "label_en": "Tax point date",
    "anchor_cs": "Datum uskutečnění zdanitelného plnění, DUZP, Datum UZP",
    "anchor_en": "Tax point date, Date of taxable supply, DUZP",
    "unit": "YYYY-MM-DD", "note": "Date of taxable supply (DUZP). Often same as period end.",
}
_F_AMOUNT_EX_VAT = {
    "field_name": "amount_ex_vat", "field_type": "float", "is_required": 1,
    "label_cs": "Celková částka bez DPH (Kč)", "label_en": "Total amount excl. VAT (CZK)",
    "anchor_cs": (
        "Základ daně, Cena celkem bez DPH, Celkem bez DPH, Základ DPH, "
        "Daňový základ, Souhrn částek celkem bez DPH, Vyúčtování celkem (bez DPH sloupec), "
        "Celkem (nedaňový základ)"
    ),
    "anchor_en": "Tax base, Total excl. VAT, Net amount, Subtotal excl. VAT",
    "unit": "CZK",
    "note": (
        "Hrubá fakturovaná částka BEZ DPH — součet PŘED odpočtem záloh. "
        "Hledej v rekapitulačním sloupci 'Základ daně' nebo 'bez DPH'. "
        "ZAKÁZANÉ HODNOTY: "
        "'Nedoplatek' (rozdíl po zálohu), "
        "'Doplatek' (přeplatek refundace), "
        "'K úhradě' (platba po záloze), "
        "'Záloha' / 'Zálohová platba' (předem zaplaceno), "
        "'Přeplatek' (záporný zůstatek). "
        "amount_ex_vat je VŽDY menší než amount_inc_vat."
    ),
}
_F_AMOUNT_INC_VAT = {
    "field_name": "amount_inc_vat", "field_type": "float", "is_required": 1,
    "label_cs": "Celková fakturovaná částka s DPH (Kč)", "label_en": "Total amount incl. VAT (CZK)",
    "anchor_cs": (
        "Celkem s DPH, Celkem vč. DPH, Celková fakturovaná částka, Fakturovaná částka, "
        "Celkové náklady, Souhrn částek celkem (s DPH), Vyúčtování celkem (s DPH sloupec), "
        "Celkem včetně DPH, Daňový přehled Celkem s DPH, Rozdíl ke zdanění (celková řádka)"
    ),
    "anchor_en": "Total incl. VAT, Gross amount, Total invoice amount, Grand total",
    "unit": "CZK",
    "note": (
        "HRUBÁ fakturovaná částka vč. DPH PŘED odpočtem zálohovýc plateb. "
        "Toto je součet všech fakturovaných položek vč. DPH — ne výsledná platba po záloze. "
        "ZAKÁZANÉ HODNOTY (model je NESMÍ použít): "
        "'Nedoplatek' = K úhradě po záloze (menší než amount_inc_vat pokud jsou zálohy); "
        "'Doplatek' = synonymum Nedoplatek; "
        "'K úhradě' / 'K úhradě celkem' = zůstatek k platbě (ne celková faktura); "
        "'Přeplatek' = záporný zůstatek; "
        "'Záloha' = předem zaplacená částka; "
        "'Haléřové vyrovnání' / 'Zaokrouhlení' = drobná korekce. "
        "ALGORITMUS: Najdi tabulku rekapitulace → sloupec 'Celkem s DPH' nebo 'vč. DPH' → "
        "vezmi SOUČTOVÝ řádek, ne řádek záloh. "
        "amount_inc_vat ≥ amount_ex_vat (DPH je kladná daň u standardní faktury)."
    ),
}
_F_SUPPLIER_TAX_ID = {
    "field_name": "supplier_tax_id", "field_type": "string", "is_required": 1,
    "label_cs": "IČO dodavatele", "label_en": "Supplier tax ID (IČO)",
    "anchor_cs": (
        "IČO dodavatele, IČO: [jméno dodavatele], Dodavatel IČO, "
        "IČO pod sekcí 'Dodavatel' / 'Prodejce' / 'Vystavovatel faktury'"
    ),
    "anchor_en": "Supplier IČO, Supplier tax ID, Vendor registration number",
    "unit": "8 digits",
    "note": (
        "Přesně 8 číslic. "
        "IČO dodavatele se nachází v sekci 'Dodavatel', 'Prodejce', nebo v hlavičce faktury. "
        "Na faktuře se VŽDY zobrazují obě IČO (dodavatelé i odběratelé) — vyber IČO DODAVATELE."
    ),
}
_F_CUSTOMER_TAX_ID = {
    "field_name": "customer_tax_id", "field_type": "string", "is_required": 1,
    "label_cs": "IČO odběratele", "label_en": "Customer tax ID (IČO)",
    "anchor_cs": (
        "IČO odběratele, IČO zákazníka, IČO: [jméno odběratele], "
        "IČO pod sekcí 'Odběratel' / 'Zákazník' / 'Odběrné místo'"
    ),
    "anchor_en": "Customer IČO, Customer tax ID, Buyer registration number",
    "unit": "8 digits",
    "note": (
        "Přesně 8 číslic. "
        "IČO odběratele se nachází v sekci 'Odběratel' nebo 'Zákazník' — NIKDY ne v sekci dodavatele. "
        "Pokud jsou na faktuře dvě různá 8-místná čísla, vyber to z části odběratele. "
        "NIKDY nepoužívej IČO dodavatele jako IČO odběratele."
    ),
}
_F_CONSUMPTION_POINT_CODE = {
    "field_name": "consumption_point_code", "field_type": "string", "is_required": 1,
    "label_cs": "Kód odběrného místa (EAN/EIC/lokální kód)", "label_en": "Supply point code (EAN/EIC/local)",
    "anchor_cs": (
        "EAN, EIC, Číslo odběrného místa, Kód odběrného místa, IČO-MO, "
        "EAN kódem, EAN OPM, s EAN kódem, kódem EAN, Odběrné místo č., "
        "číslem EAN, Číslo OPM, OPM, Odběrné místo (číslo), "
        "Identifikátor odběrného místa, Evidenční číslo odběrného místa"
    ),
    "anchor_en": "EAN code, EIC code, EAN OPM, Supply point code, Metering point ID, OPM number",
    "unit": "EAN=18 digits (859182...), EIC=16 chars (27ZG...), or local supplier code",
    "note": (
        "EAN = přesně 18 číslic, v ČR začíná '859182'. "
        "EIC = 16 znaků začínající '27ZG'. "
        "Pro teplo a vodu může být kratší lokální kód (6-12 číslic) přiřazený dodavatelem — vrať ho celý. "
        "Pokud OCR rozbije číslo přes více řádků, rekonstruuj celý řetězec. "
        "NIKDY nepoužívej 8-místné IČO jako kód odběrného místa. "
        "Kód odběrného místa se liší od čísla faktury a variabilního symbolu."
    ),
}

# Společná pole bez amounts/IČO (pro elektrina_vn kde GT nemá tyto sloupce)
_COMMON_BASE = [
    _F_INVOICE_NUMBER, _F_PERIOD_FROM, _F_PERIOD_TO,
    _F_ISSUE_DATE, _F_DUE_DATE, _F_TAX_POINT_DATE,
    _F_CONSUMPTION_POINT_CODE,
]
# Plná sada společných polí (pro ostatní komodity)
_COMMON_FULL = _COMMON_BASE + [
    _F_AMOUNT_EX_VAT, _F_AMOUNT_INC_VAT,
    _F_SUPPLIER_TAX_ID, _F_CUSTOMER_TAX_ID,
]

# --- Elektřina NN ---
_ELEKTRINA_NN_FIELDS: list[dict] = _COMMON_FULL + [
    {"field_name": "consumption_high_tariff", "field_type": "float", "is_required": 0,
     "label_cs": "Spotřeba VT — vysoký tarif (kWh)", "label_en": "High-tariff consumption (kWh)",
     "anchor_cs": (
         "Spotřeba VT, VT spotřeba, Vysoký tarif, Vysoký tarif (VT/T1), "
         "VT celkem, Dodané množství elektřiny (VT), T1, TDO1, Tarif VT, Pásmo VT"
     ),
     "anchor_en": "High tariff, VT consumption, VT/T1, peak tariff, T1",
     "unit": "kWh",
     "note": (
         "Spotřeba v vysokém tarifu (VT/T1) za AKTUÁLNÍ fakturační období v kWh, ne MWh. "
         "Hledej explicitní řádek 'VT', 'T1', 'Vysoký tarif' v tabulce spotřeby/přehledu plateb. "
         "Pokud faktura má pouze jeden tarif (bez NT sekce), jde vždy o VT. "
         "Může být null pouze pokud NT je non-null (extrémně vzácné) — alespoň jedno z VT/NT musí být vyplněno."
     )},
    {"field_name": "consumption_low_tariff", "field_type": "float", "is_required": 0,
     # is_required=0: single-tariff (D01/T1) invoices have no NT at all — do NOT escalate
     "label_cs": "Spotřeba NT — nízký tarif (kWh)", "label_en": "Low-tariff consumption (kWh)",
     "anchor_cs": (
         "Spotřeba NT, NT spotřeba, Nízký tarif, Noční tarif, Noční proud, "
         "Nízký tarif (NT/T2), T2, TDO2, Tarif NT, Pásmo NT, "
         "Dodané množství elektřiny (NT)"
     ),
     "anchor_en": "Low tariff, NT consumption, NT/T2, off-peak tariff, T2, night tariff",
     "unit": "kWh",
     "note": (
         "Spotřeba v nízkém tarifu (NT/T2) za AKTUÁLNÍ fakturační období v kWh, ne MWh. "
         "POKUD faktura neobsahuje sekci NT / T2 (jednosázkový odběr D01, pouze VT řádek) — vrať null. "
         "NT je typicky výrazně menší hodnota než VT (noční spotřeba). "
         "NIKDY nekopíruj hodnotu VT do NT. "
         "Hledej explicitní řádek 'NT', 'T2', 'Nízký tarif' v tabulce spotřeby."
     )},
    {"field_name": "total_consumption", "field_type": "float", "is_required": 0,
     "label_cs": "Celková spotřeba elektřiny (kWh)", "label_en": "Total electricity consumption (kWh)",
     "anchor_cs": (
         "Celková spotřeba, Spotřeba celkem, Dodané množství elektřiny celkem, "
         "Celkem dodané energie, VT + NT celkem"
     ),
     "anchor_en": "Total consumption, Total energy delivered",
     "unit": "kWh",
     "note": (
         "Součet VT + NT v kWh za fakturační období. "
         "Pokud je celková spotřeba v MWh (VN faktury), nepoužívej toto pole. "
         "Vrať null pokud není explicitně uvedena."
     )},
]

# --- Elektřina VN ---
_ELEKTRINA_VN_FIELDS: list[dict] = _COMMON_BASE + [
    # GT nemá amount_ex_vat, amount_inc_vat, supplier_tax_id, customer_tax_id
    {"field_name": "supply_consumption", "field_type": "float", "is_required": 1,
     "label_cs": "Spotřeba silové elektřiny (MWh)", "label_en": "Active energy consumption (MWh)",
     "anchor_cs": "Spotřeba silové elektřiny, Elektrická energie odběr, Silová elektřina",
     "anchor_en": "Active energy, Supply consumption, Electricity consumption",
     "unit": "MWh", "note": "In MWh (NOT kWh). Typically hundreds to thousands of MWh."},
    {"field_name": "supply_charge", "field_type": "float", "is_required": 1,
     "label_cs": "Cena za silovou elektřinu (Kč)", "label_en": "Supply charge (CZK)",
     "anchor_cs": "Cena za silovou elektřinu, Cena komodity, Komoditní složka celkem",
     "anchor_en": "Supply charge, Commodity charge",
     "unit": "CZK", "note": "Total price paid for the active energy commodity."},
    {"field_name": "supply_tax_charge", "field_type": "float", "is_required": 0,
     "label_cs": "Daň z elektřiny (Kč)", "label_en": "Electricity tax (CZK)",
     "anchor_cs": "Daň z elektřiny, Ekologická daň, Spotřební daň",
     "anchor_en": "Electricity tax, Environmental tax, Excise duty",
     "unit": "CZK", "note": ""},
    {"field_name": "quarter_hour_max", "field_type": "float", "is_required": 0,
     "label_cs": "Čtvrthodinové maximum (kW)", "label_en": "Quarter-hour peak demand (kW)",
     "anchor_cs": "Čtvrthodinové maximum, 1/4h max, Čtvrthodinnové maximum",
     "anchor_en": "Quarter-hour maximum, Peak demand",
     "unit": "kW", "note": "Maximum 15-minute interval power demand in kW."},
    {"field_name": "annual_reserved_capacity", "field_type": "float", "is_required": 0,
     "label_cs": "Roční rezervovaná kapacita — množství (kW)", "label_en": "Annual reserved capacity (kW)",
     "anchor_cs": "Roční rezervovaná kapacita, Roční RK, Rezervovaná kapacita roční",
     "anchor_en": "Annual reserved capacity",
     "unit": "kW", "note": "Contracted annual capacity in kW. NOT the price."},
    {"field_name": "annual_reserved_capacity_charge", "field_type": "float", "is_required": 0,
     "label_cs": "Roční rezervovaná kapacita — částka (Kč)", "label_en": "Annual reserved capacity charge (CZK)",
     "anchor_cs": "Cena za roční RK, Roční rezervovaná kapacita cena",
     "anchor_en": "Annual reserved capacity charge",
     "unit": "CZK", "note": "Price paid for the annual reserved capacity."},
    {"field_name": "monthly_reserved_capacity", "field_type": "float", "is_required": 0,
     "label_cs": "Měsíční rezervovaná kapacita — množství (kW)", "label_en": "Monthly reserved capacity (kW)",
     "anchor_cs": "Měsíční rezervovaná kapacita, Měsíční RK",
     "anchor_en": "Monthly reserved capacity",
     "unit": "kW", "note": "Monthly contracted capacity in kW. NOT the price."},
    {"field_name": "monthly_reserved_capacity_charge", "field_type": "float", "is_required": 0,
     "label_cs": "Měsíční rezervovaná kapacita — částka (Kč)", "label_en": "Monthly reserved capacity charge (CZK)",
     "anchor_cs": "Cena za měsíční RK, Měsíční RK cena",
     "anchor_en": "Monthly reserved capacity charge",
     "unit": "CZK", "note": ""},
    {"field_name": "grid_usage_rate", "field_type": "float", "is_required": 0,
     "label_cs": "Sazba přenosové soustavy (Kč/MWh)", "label_en": "Transmission grid usage rate (CZK/MWh)",
     "anchor_cs": "Sazba za použití přenosové soustavy, Přenosová soustava sazba",
     "anchor_en": "Transmission rate, Grid usage rate",
     "unit": "CZK/MWh", "note": "Per-MWh rate for transmission grid usage. NOT the total."},
    {"field_name": "grid_usage_charge", "field_type": "float", "is_required": 0,
     "label_cs": "Cena přenosové soustavy (Kč)", "label_en": "Transmission grid charge (CZK)",
     "anchor_cs": "Cena za použití přenosové soustavy, Přenosová soustava celkem",
     "anchor_en": "Transmission charge, Grid usage charge",
     "unit": "CZK", "note": ""},
    {"field_name": "reserved_capacity_excess", "field_type": "float", "is_required": 0,
     "label_cs": "Překročení rezervované kapacity — množství (kW)", "label_en": "Reserved capacity excess (kW)",
     "anchor_cs": "Překročení rezervované kapacity, Překročení RK množství",
     "anchor_en": "Reserved capacity excess, Capacity overrun",
     "unit": "kW", "note": "Amount by which actual demand exceeded contracted capacity."},
    {"field_name": "reserved_capacity_excess_rate", "field_type": "float", "is_required": 0,
     "label_cs": "Sazba za překročení RK (Kč/kW)", "label_en": "Reserved capacity excess rate (CZK/kW)",
     "anchor_cs": "Sazba za překročení RK, Překročení RK sazba",
     "anchor_en": "Excess capacity rate",
     "unit": "CZK/kW", "note": ""},
    {"field_name": "power_factor", "field_type": "float", "is_required": 0,
     "label_cs": "tg φ — účiník odběru VN", "label_en": "Power factor (tg φ)",
     "anchor_cs": "tg φ, Tangens fí, Účiník, tgφ",
     "anchor_en": "Power factor, tg phi, tan phi",
     "unit": "dimensionless 0.00–1.00", "note": "Tangent of the phase angle. Dimensionless, typically 0.00–0.40."},
    {"field_name": "reactive_power_quantity", "field_type": "float", "is_required": 0,
     "label_cs": "Množství jalové energie (MVArh)", "label_en": "Reactive energy quantity (MVArh)",
     "anchor_cs": "Jalová energie množství, Reaktivní energie, MVArh",
     "anchor_en": "Reactive energy, MVArh",
     "unit": "MVArh", "note": ""},
    {"field_name": "reactive_power_rate", "field_type": "float", "is_required": 0,
     "label_cs": "Sazba za jalovou energii (Kč/MVArh)", "label_en": "Reactive energy rate (CZK/MVArh)",
     "anchor_cs": "Sazba za jalovou energii, Jalová energie sazba",
     "anchor_en": "Reactive energy rate",
     "unit": "CZK/MVArh", "note": ""},
    {"field_name": "reactive_power_charge", "field_type": "float", "is_required": 0,
     "label_cs": "Cena za jalovou energii (Kč)", "label_en": "Reactive energy charge (CZK)",
     "anchor_cs": "Cena za jalovou energii, Jalová energie celkem",
     "anchor_en": "Reactive energy charge",
     "unit": "CZK", "note": ""},
    {"field_name": "service_price", "field_type": "float", "is_required": 0,
     "label_cs": "Systémové služby (Kč)", "label_en": "System services (CZK)",
     "anchor_cs": "Systémové služby, Cena systémových služeb",
     "anchor_en": "System services",
     "unit": "CZK", "note": ""},
    {"field_name": "operating_price", "field_type": "float", "is_required": 0,
     "label_cs": "Nesíťová infrastruktura (Kč)", "label_en": "Non-network infrastructure (CZK)",
     "anchor_cs": "Nesíťová infrastruktura, Provoz nesíťové infrastruktury",
     "anchor_en": "Non-network infrastructure, Operating price",
     "unit": "CZK", "note": ""},
    {"field_name": "renewable_energy_fee", "field_type": "float", "is_required": 0,
     "label_cs": "POZE — příspěvek na obnovitelné zdroje (Kč)", "label_en": "Renewable energy fee (CZK)",
     "anchor_cs": "POZE, Podpora obnovitelných zdrojů energie, OZE příspěvek",
     "anchor_en": "POZE, Renewable energy contribution",
     "unit": "CZK", "note": ""},
]

# --- Plyn MO ---
_PLYN_MO_FIELDS: list[dict] = _COMMON_FULL + [
    {"field_name": "conversion_coefficient", "field_type": "float", "is_required": 1,
     "label_cs": "Přepočtový koeficient objemu plynu", "label_en": "Gas volume conversion coefficient",
     "anchor_cs": (
         "Přepočtový koeficient, Koeficient přepočtu, Přepočtový faktor, "
         "Přepoč. koef., Koeficient množství, Přepočtový součinitel, Korekční koeficient"
     ),
     "anchor_en": "Conversion coefficient, Přepoč. koef., Volume correction factor",
     "unit": "dimensionless 1.00–1.10",
     "note": (
         "Bezrozměrný koeficient přepočtu naměřeného m³ na fakturované m³. Typicky 1,000–1,100. "
         "Na Pražské plynárenské: sloupec 'Přepoč. koef.' v detailní tabulce na str. 3+ (<!-- page 3 -->). "
         "POKUD má faktura více měřidel: vezmi koeficient pro první nebo dominantní měřidlo. "
         "POZOR: hodnota ~1,0 (ne 10,x — to je spalné teplo). "
         "Pokud vidíš '1,0225' nebo '1,0009' — to je conversion_coefficient."
     )},
    {"field_name": "combustion_heat", "field_type": "float", "is_required": 1,
     "label_cs": "Spalné teplo (kWh/m³)", "label_en": "Combustion heat value (kWh/m³)",
     "anchor_cs": (
         "Spalné teplo, Výhřevnost, Tepelná hodnota plynu, "
         "Spalné teplo plynu, Průměrné spalné teplo"
     ),
     "anchor_en": "Combustion heat, Calorific value, Heating value",
     "unit": "kWh/m³ ~10.5",
     "note": (
         "Výhřevnost zemního plynu v kWh/m³. Typicky 10,0–11,5 kWh/m³. "
         "Hledej NA DVOU MÍSTECH: "
         "(1) Str. 1 body text: 'bylo použito spalné teplo 10,7641' nebo 'spalné teplo 10,xxxx'; "
         "(2) Str. 3+ detailní tabulka: sloupec 'Spalné teplo'. Preferuj detailní tabulku. "
         "Pokud vidíš '10,7641', '10,9255', '10,9119' — to je combustion_heat v kWh/m³. "
         "Pokud dodavatel uvádí MJ/m³ (~38–39), vyděl 3,6 PŘED vrácením. "
         "NIKDY nekombinuj se conversion_coefficient (~1,05)."
     )},
    {"field_name": "consumption_mwh", "field_type": "float", "is_required": 1,
     "label_cs": "Spotřeba přepočtená (MWh)", "label_en": "Converted gas consumption (MWh)",
     "anchor_cs": (
         "Spotřeba přepočtená, Spotřeba v MWh, Fakturované MWh, "
         "přepočtené spotřebě energie ve výši, Celkem MWh, "
         "Dodané množství plynu (vezmi MWh hodnotu, ne kWh)"
     ),
     "anchor_en": "Converted consumption, Consumption in MWh, Billed MWh, energy in MWh",
     "unit": "MWh",
     "note": (
         "Spotřeba přepočtená na MWh. Formula: m³ × koef × spalné_teplo_kWh/m³ / 1 000. "
         "DŮLEŽITÉ: faktura může zobrazit tatáž hodnota v kWh ('Dodané množství plynu XXXX kWh') — "
         "vždy vrať hodnotu v MWh (ne kWh). "
         "Hledej větu 'přepočtené spotřebě energie ve výši X,X MWh' nebo sloupec MWh v tabulce. "
         "Pokud vidíš jen kWh, vyděl 1 000. NENÍ m³ hodnota."
     )},
    {"field_name": "consumption_m3", "field_type": "float", "is_required": 1,
     "label_cs": "Spotřeba naměřená (m³)", "label_en": "Measured gas consumption (m³)",
     "anchor_cs": (
         "Nepřepočtená spotřeba, Spotřeba naměřená, Spotřeba v m³, Odečet plynu, "
         "Spotřeba ZP, Spotřeba zemního plynu, Množství plynu m3, Spotřeba m³"
     ),
     "anchor_en": "Measured consumption, Nepřepočtená spotřeba, Gas meter reading",
     "unit": "m³",
     "note": (
         "Naměřená spotřeba v m³ — surový odečet plynoměru. NENÍ MWh hodnota. "
         "Na Pražské plynárenské faktuře: 'Nepřepočtená spotřeba XX,00 m*'. "
         "V OCR: m³ → 'm*', 'm3', 'm?' — přečti číslo PŘED touto jednotkou."
     )},
    {"field_name": "gas_tax_total", "field_type": "float", "is_required": 1,
     "label_cs": "Daň ze zemního plynu celkem (Kč)", "label_en": "Natural gas tax total (CZK)",
     "anchor_cs": (
         "Daň ze zemního plynu, Spotřební daň ZP, Daň ze ZP celkem, "
         "Daň ze ZP, Ekologická daň ZP, daň ze ZP, "
         "DAŇ ZE ZEMNÍHO PLYNU: Celkem, DAŇ ZE ZEMNÍHO PLYNU PRO MĚŘIDLO"
     ),
     "anchor_en": "Gas tax, Natural gas excise duty, Gas tax total, DAŇ ZE ZEMNÍHO PLYNU",
     "unit": "CZK",
     "note": (
         "Celková spotřební daň ze zemního plynu v Kč. "
         "Na fakturách Pražské plynárenské: sekce 'DAŇ ZE ZEMNÍHO PLYNU:' s řádkem 'Celkem: XX,XX' na str. 3+. "
         "POKUD faktura má více měřidel (MĚŘIDLO ČÍSLO 1, 2...): každé má svou sekci DAŇ — "
         "seč VŠECHNY 'Celkem' hodnoty ze všech sekcí DAŇ ZE ZP a vrať součet. "
         "NENÍ součástí ceny komodity — jde jako samostatná položka."
     )},
    {"field_name": "unit_price_commodity", "field_type": "float", "is_required": 0,
     "label_cs": "Jednotková cena komodity (Kč/MWh)", "label_en": "Commodity unit price (CZK/MWh)",
     "anchor_cs": "Jednotková cena komodity, Cena plynu za MWh, Komoditní sazba",
     "anchor_en": "Commodity unit price, Gas price per MWh",
     "unit": "CZK/MWh", "note": ""},
    {"field_name": "commodity_price", "field_type": "float", "is_required": 0,
     "label_cs": "Cena komodity celkem (Kč)", "label_en": "Commodity total price (CZK)",
     "anchor_cs": "Cena komodity celkem, Komoditní složka celkem",
     "anchor_en": "Commodity total",
     "unit": "CZK", "note": ""},
    {"field_name": "unit_price_standing_charge", "field_type": "float", "is_required": 0,
     "label_cs": "Stálý měsíční plat — jednotková cena (Kč/měs.)", "label_en": "Standing charge unit price (CZK/month)",
     "anchor_cs": "Stálý měsíční plat sazba, Paušál jednotková cena",
     "anchor_en": "Standing charge unit price, Monthly fixed fee rate",
     "unit": "CZK/month", "note": ""},
    {"field_name": "standing_charge", "field_type": "float", "is_required": 0,
     "label_cs": "Stálý měsíční plat celkem (Kč)", "label_en": "Standing charge total (CZK)",
     "anchor_cs": "Stálý měsíční plat celkem, Paušál celkem",
     "anchor_en": "Standing charge total, Monthly fixed fee total",
     "unit": "CZK", "note": ""},
    {"field_name": "unit_price_reserved_capacity", "field_type": "float", "is_required": 0,
     "label_cs": "Přistavená kapacita — jednotková cena (Kč/MWh)", "label_en": "Reserved capacity unit price (CZK/MWh)",
     "anchor_cs": "Přistavená kapacita sazba, Kapacita jednotková cena",
     "anchor_en": "Reserved capacity unit price",
     "unit": "CZK/MWh", "note": ""},
    {"field_name": "reserved_capacity_price", "field_type": "float", "is_required": 0,
     "label_cs": "Přistavená kapacita celkem (Kč)", "label_en": "Reserved capacity total (CZK)",
     "anchor_cs": "Přistavená kapacita celkem, Kapacita cena celkem",
     "anchor_en": "Reserved capacity total",
     "unit": "CZK", "note": ""},
    {"field_name": "unit_price_distribution", "field_type": "float", "is_required": 0,
     "label_cs": "Distribuce plynu — jednotková cena (Kč/MWh)", "label_en": "Gas distribution unit price (CZK/MWh)",
     "anchor_cs": "Distribuce sazba, Distribuce plynu jednotková cena",
     "anchor_en": "Distribution unit price",
     "unit": "CZK/MWh", "note": ""},
    {"field_name": "distribution_price", "field_type": "float", "is_required": 0,
     "label_cs": "Distribuce plynu celkem (Kč)", "label_en": "Gas distribution total (CZK)",
     "anchor_cs": "Distribuce celkem, Distribuce plynu cena celkem",
     "anchor_en": "Distribution total",
     "unit": "CZK", "note": ""},
    {"field_name": "market_operator_fee", "field_type": "float", "is_required": 0,
     "label_cs": "Cena za činnost OTE — operátor trhu (Kč)", "label_en": "Market operator fee (CZK)",
     "anchor_cs": "OTE, Operátor trhu, Cena za činnost operátora trhu",
     "anchor_en": "OTE, Market operator fee",
     "unit": "CZK", "note": ""},
]

# --- Plyn VO ---
_PLYN_VO_FIELDS: list[dict] = _COMMON_FULL + [
    {"field_name": "conversion_coefficient", "field_type": "float", "is_required": 1,
     "label_cs": "Přepočtový koeficient objemu plynu", "label_en": "Gas volume conversion coefficient",
     "anchor_cs": (
         "Přepočtový koeficient, Koeficient přepočtu, Přepočtový faktor, "
         "Koeficient množství, Přepočtový součinitel"
     ),
     "anchor_en": "Conversion coefficient, Volume correction factor",
     "unit": "dimensionless 1.00–1.10",
     "note": "Bezrozměrný koeficient ~1,05. NENÍ spalné teplo (~10,5 kWh/m³). Hodnota vždy kolem 1,0."},
    {"field_name": "combustion_heat", "field_type": "float", "is_required": 1,
     "label_cs": "Spalné teplo (kWh/m³)", "label_en": "Combustion heat value (kWh/m³)",
     "anchor_cs": (
         "Spalné teplo, Výhřevnost, Tepelná hodnota plynu, "
         "Spalné teplo plynu, Průměrné spalné teplo"
     ),
     "anchor_en": "Combustion heat, Calorific value, Heating value",
     "unit": "kWh/m³ ~10.5",
     "note": (
         "Výhřevnost plynu v kWh/m³, typicky 10,0–11,5. "
         "V OCR: holá desetinná hodnota za labelem 'spalné teplo', např. 'spalné teplo 10,7641'. "
         "Někteří VO dodavatelé uvádí v MJ/m³ (~38–39 MJ/m³ → vyděl 3,6). "
         "NENÍ conversion_coefficient (~1,05)."
     )},
    {"field_name": "consumption_mwh", "field_type": "float", "is_required": 1,
     "label_cs": "Spotřeba přepočtená (MWh)", "label_en": "Converted gas consumption (MWh)",
     "anchor_cs": (
         "Spotřeba přepočtená, Spotřeba v MWh, Fakturované MWh, "
         "Celková spotřeba MWh"
     ),
     "anchor_en": "Converted consumption in MWh, Billed MWh",
     "unit": "MWh", "note": "Přepočtená spotřeba v MWh. NENÍ m³ hodnota."},
    {"field_name": "consumption_m3", "field_type": "float", "is_required": 1,
     "label_cs": "Spotřeba naměřená (m³)", "label_en": "Measured gas consumption (m³)",
     "anchor_cs": (
         "Spotřeba naměřená, Spotřeba v m³, Odečet plynu, "
         "Spotřeba ZP, Množství plynu, Spotřeba m³"
     ),
     "anchor_en": "Measured consumption in m³, Gas meter reading",
     "unit": "m³",
     "note": "Naměřená spotřeba v m³. V OCR: m³ může být 'm*', 'm3', 'm ³' — přečti číslo před tím."},
    {"field_name": "daily_reserved_capacity", "field_type": "float", "is_required": 1,
     "label_cs": "Denní rezervovaná kapacita (tis. m³/den)", "label_en": "Daily reserved capacity (thousand m³/day)",
     "anchor_cs": "Denní rezervovaná kapacita, DRK, Denní RK, Přistavená denní kapacita",
     "anchor_en": "Daily reserved capacity, DRC",
     "unit": "thousand m³/day", "note": "V tisících m³/den (tis. m³/den). Typicky 0,xxx (desetinná hodnota menší než 1)."},
    {"field_name": "gas_tax_total", "field_type": "float", "is_required": 1,
     "label_cs": "Daň ze zemního plynu celkem (Kč)", "label_en": "Natural gas tax total (CZK)",
     "anchor_cs": (
         "Daň ze zemního plynu, Daň ze ZP celkem, Spotřební daň ZP, "
         "Daň ze ZP, Ekologická daň ZP"
     ),
     "anchor_en": "Gas tax total, Natural gas excise duty",
     "unit": "CZK",
     "note": "Celková spotřební daň v Kč. Může být na 2. straně faktury v detailní části — hledej na všech stránkách."},
    {"field_name": "other_supply_services", "field_type": "float", "is_required": 0,
     "label_cs": "Ostatní služby dodávky (Kč)", "label_en": "Other supply services (CZK)",
     "anchor_cs": "Ostatní služby dodávky, Ostatní služby",
     "anchor_en": "Other supply services",
     "unit": "CZK", "note": ""},
    {"field_name": "unit_price_trade_reserved_capacity", "field_type": "float", "is_required": 0,
     "label_cs": "Obchod RK — jednotková cena (Kč)", "label_en": "Trade reserved capacity unit price (CZK)",
     "anchor_cs": "Obchod rezervovaná kapacita sazba, Obchod RK jednotková",
     "anchor_en": "Trade reserved capacity unit price",
     "unit": "CZK", "note": ""},
    {"field_name": "trade_reserved_capacity_price", "field_type": "float", "is_required": 0,
     "label_cs": "Obchod RK — cena celkem (Kč)", "label_en": "Trade reserved capacity total (CZK)",
     "anchor_cs": "Obchod rezervovaná kapacita celkem, Obchod RK cena",
     "anchor_en": "Trade reserved capacity total",
     "unit": "CZK", "note": ""},
    {"field_name": "distribution_services_price", "field_type": "float", "is_required": 0,
     "label_cs": "Distribuce — cena za službu celkem (Kč)", "label_en": "Distribution services total (CZK)",
     "anchor_cs": "Distribuce služba celkem, Distribuční služby cena",
     "anchor_en": "Distribution services total",
     "unit": "CZK", "note": ""},
    {"field_name": "unit_price_distr_services", "field_type": "float", "is_required": 0,
     "label_cs": "Distribuce — jednotková cena za službu (Kč/MWh)", "label_en": "Distribution service unit price (CZK/MWh)",
     "anchor_cs": "Distribuce sazba za službu, Distribuce jednotková cena",
     "anchor_en": "Distribution service unit price",
     "unit": "CZK/MWh", "note": ""},
    {"field_name": "unit_price_distr_reserved_capacity", "field_type": "float", "is_required": 0,
     "label_cs": "Distribuce RK — jednotková cena (Kč)", "label_en": "Distribution reserved capacity unit price (CZK)",
     "anchor_cs": "Distribuce rezervovaná kapacita sazba, Distribuce RK jednotková",
     "anchor_en": "Distribution reserved capacity unit price",
     "unit": "CZK", "note": ""},
    {"field_name": "distribution_reserved_capacity_price", "field_type": "float", "is_required": 0,
     "label_cs": "Distribuce RK — cena celkem (Kč)", "label_en": "Distribution reserved capacity total (CZK)",
     "anchor_cs": "Distribuce rezervovaná kapacita celkem, Distribuce RK cena",
     "anchor_en": "Distribution reserved capacity total",
     "unit": "CZK", "note": ""},
    {"field_name": "market_operator_fee", "field_type": "float", "is_required": 0,
     "label_cs": "Cena za činnost OTE (Kč)", "label_en": "Market operator fee (CZK)",
     "anchor_cs": "OTE, Operátor trhu",
     "anchor_en": "OTE, Market operator fee",
     "unit": "CZK", "note": ""},
]

# --- Teplo ---
_TEPLO_FIELDS: list[dict] = _COMMON_FULL + [
    {"field_name": "consumption_gj", "field_type": "float", "is_required": 1,
     "label_cs": "Spotřeba tepla celkem (GJ)", "label_en": "Total heat consumption (GJ)",
     "anchor_cs": (
         "Spotřeba tepla, Teplo celkem, Spotřeba v GJ, "
         "Teplo ostatní, Teplo celkem GJ, Dodávka tepla, Tepelná energie"
     ),
     "anchor_en": "Heat consumption, GJ total, Heat energy, Teplo ostatní",
     "unit": "GJ",
     "note": (
         "Celková spotřeba tepla v GJ (gigajoule), NIKOLI kWh ani MWh. 1 GJ ≈ 277,8 kWh. "
         "Na fakturách CZT se nejčastěji označuje 'Teplo ostatní' nebo 'Teplo celkem'. "
         "Hledej v tabulce komodit řádek s GJ jednotkou — NIKOLI řádek s kW (to je rezervovaná kapacita). "
         "Pokud jsou dvě teplo hodnoty (vytápění + TV), vezmi řádek 'Teplo ostatní' (ne TV)."
     )},
    {"field_name": "hot_water_consumption", "field_type": "float", "is_required": 1,
     "label_cs": "Spotřeba tepla pro ohřev TV (GJ)", "label_en": "Heat for hot water preparation (GJ)",
     "anchor_cs": (
         "Spotřeba tepla pro TV, TV spotřeba, Ohřev teplé vody GJ, "
         "TV ostatní, Teplá voda GJ, Ohřev TV, TV celkem GJ"
     ),
     "anchor_en": "Hot water heating, TV heat consumption, TV ostatní",
     "unit": "GJ",
     "note": (
         "Tepelná energie v GJ použitá pro ohřev teplé vody (TV = teplá voda, NE televize). "
         "Na fakturách označeno 'TV ostatní' nebo 'Ohřev TV'. "
         "NENÍ to celková spotřeba tepla — je to jen složka pro teplou vodu. "
         "Typicky výrazně menší hodnota než consumption_gj."
     )},
    {"field_name": "cold_water_consumption", "field_type": "float", "is_required": 0,
     # Changed: is_required=0 because some teplo invoices may not include cold water separately
     "label_cs": "Dodávka studené vody pro ohřev TV (m³)", "label_en": "Cold water for hot water heating (m³)",
     "anchor_cs": (
         "Dodávka studené vody, Studená voda pro TV, Studená voda m³, "
         "SV pro TV, Studená voda celkem, Studená voda"
     ),
     "anchor_en": "Cold water delivery, Cold water for TV, Cold water volume",
     "unit": "m³",
     "note": (
         "Objem studené vody v m³ použité pro přípravu teplé vody (TV). "
         "Nemusí být na všech tepelných fakturách — vrať null pokud chybí. "
         "V OCR: m³ může být 'm*', 'm3' — přečti číslo před touto jednotkou."
     )},
    {"field_name": "reserved_capacity", "field_type": "float", "is_required": 1,
     "label_cs": "Rezervovaná kapacita tepla (kW)", "label_en": "Reserved heat capacity (kW)",
     "anchor_cs": (
         "Rezervovaná kapacita, RK teplo, Přistavená tepelná kapacita, "
         "Rezervovaná kapacita tepla, Sjednaná kapacita"
     ),
     "anchor_en": "Reserved capacity, Contracted heat capacity",
     "unit": "kW",
     "note": (
         "Sjednaná tepelná kapacita v kW. "
         "OCR někdy zobrazuje 'kW' jako 'kWw' nebo 'kWv' — jde stále o kW. "
         "NIKOLI GJ (spotřeba) ani Kč (cena) — hledej číslo s jednotkou kW."
     )},
    {"field_name": "supplementary_water", "field_type": "float", "is_required": 1,
     "label_cs": "Doplňovací voda — náhrada ztrát (m³)", "label_en": "Supplementary water — loss replacement (m³)",
     "anchor_cs": (
         "Doplňovací voda, Náhrada ztrát, Doplňovací voda m³, "
         "Doplňovací voda (náhrada ztrát), Náhrada ztráty vody"
     ),
     "anchor_en": "Supplementary water, Make-up water, Loss replacement",
     "unit": "m³",
     "note": (
         "Objem vody v m³ doplněný do otopné soustavy jako náhrada za úniky. "
         "V OCR: m³ může být 'm*', 'm3' — přečti číslo před touto jednotkou."
     )},
    {"field_name": "total_heat_consumption", "field_type": "float", "is_required": 0,
     "label_cs": "Celková spotřeba tepla — součtové pole (GJ)", "label_en": "Total heat — sum control field (GJ)",
     "anchor_cs": "Celková spotřeba tepla, Spotřeba tepla celkem součet, Celkem teplo GJ",
     "anchor_en": "Total heat consumption sum",
     "unit": "GJ", "note": "Volitelné součtové pole — součet všech tepelných složek v GJ. Vrať null pokud chybí."},
]

# --- Voda ---
_VODA_FIELDS: list[dict] = _COMMON_FULL + [
    {"field_name": "consumption_vodne_m3", "field_type": "float", "is_required": 1,
     "label_cs": "Vodné — dodávka pitné vody (m³)", "label_en": "Water supply (m³)",
     "anchor_cs": (
         "Vodné, Pitná voda, Dodávka vody, Vodné m³, "
         "Fakt. vodné, vodné odb.místo, Vodné celkem, Voda dodaná"
     ),
     "anchor_en": "Water supply, Drinking water, Vodné, Fakt. vodné",
     "unit": "m³",
     "note": (
         "Objem dodané pitné vody v m³. "
         "Label na faktuře: 'Vodné', 'Fakt. vodné', nebo 'vodné odb.místo'. "
         "V OCR: m³ může být 'm*', 'm3' nebo 'm ³' — přečti číslo před touto jednotkou. "
         "Pokud je v závorce objem v litrech (např. '(1 304 000 l)'), ignoruj ho — použij m³ hodnotu."
     )},
    {"field_name": "consumption_stocne_m3", "field_type": "float", "is_required": 1,
     "label_cs": "Stočné celkem — odvod odpadní vody (m³)", "label_en": "Total sewage (m³)",
     "anchor_cs": (
         "Stočné celkem, Odvedení odpadní vody, Stočné m³, "
         "Fakt. stočné, stočné odb.místo, Stočné, Odvod odpadní vody"
     ),
     "anchor_en": "Sewage total, Wastewater, Stočné, Fakt. stočné",
     "unit": "m³",
     "note": (
         "Celkový objem odvodu odpadní vody v m³ (odpadní + srážkové dohromady). "
         "Label: 'Stočné', 'Fakt. stočné', nebo 'stočné odb.místo'. "
         "Pokud jsou odpadní a srážkové stočné odděleny, vezmi celkový součet pro toto pole. "
         "V OCR: m³ může být 'm*', 'm3' — přečti číslo před touto jednotkou."
     )},
    {"field_name": "consumption_odpadni_m3", "field_type": "float", "is_required": 0,
     "label_cs": "Odpadní stočné (m³)", "label_en": "Wastewater sewage (m³)",
     "anchor_cs": (
         "Odpadní stočné, Splašková voda, Odpadní voda m³, "
         "Splaškové stočné, Odpadní vody"
     ),
     "anchor_en": "Wastewater sewage, Splashwater, Odpadní stočné",
     "unit": "m³",
     "note": (
         "Objem odpadní (splaškové) vody odděleně od srážkové. "
         "Vrať null pokud faktura nerozděluje stočné na odpadní a srážkové. "
         "V OCR: m³ může být 'm*', 'm3'."
     )},
    {"field_name": "consumption_srazkove_m3", "field_type": "float", "is_required": 0,
     "label_cs": "Srážkové stočné (m³)", "label_en": "Rainwater sewage (m³)",
     "anchor_cs": (
         "Srážkové stočné, Dešťová voda, Srážkové vody m³, "
         "Srážková voda, Srážky odb.místo, Dešťové vody"
     ),
     "anchor_en": "Rainwater sewage, Storm water, Srážková voda",
     "unit": "m³",
     "note": (
         "Objem srážkové (dešťové) vody v m³ — pouze pokud je na faktuře zvlášť. "
         "Label: 'Srážkové stočné', 'Srážková voda', 'srážky odb.místo'. "
         "Vrať null pokud faktura nerozděluje srážkové stočné. "
         "V OCR: m³ může být 'm*', 'm3'."
     )},
]

FIELD_REGISTRY: dict[str, list[dict]] = {
    "elektrina_nn": _ELEKTRINA_NN_FIELDS,
    "elektrina_vn": _ELEKTRINA_VN_FIELDS,
    "plyn_mo":      _PLYN_MO_FIELDS,
    "plyn_vo":      _PLYN_VO_FIELDS,
    "teplo":        _TEPLO_FIELDS,
    "voda":         _VODA_FIELDS,
}

_COMMON_FIELDS_ALL = _COMMON_FULL  # fallback pro neznámou komoditu

def load_commodity_fields(commodity: str) -> list[dict]:
    return FIELD_REGISTRY.get(commodity, _COMMON_FIELDS_ALL)


# ─────────────────────────────────────────────────────────────────────────────
# GROUND TRUTH
# ─────────────────────────────────────────────────────────────────────────────

_META_COLS = {"scan_filename", "source_filename", "supplier", "commodity",
              "q_class", "link", "is_transitional"}

_GT_ALIASES = {
    "total_amount_ex_vat":  "amount_ex_vat",
    "total_amount_inc_vat": "amount_inc_vat",
}


def _normalize_stem(value: Any) -> str:
    text = Path(str(value).strip()).stem
    text = re.sub(r"__scan_q\d+_\d+$", "", text).strip()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def _fmt_amount(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)) or not str(v).strip():
        return ""
    try:
        return f"{float(str(v).replace(' ', '').replace(',', '.')):,.2f}".replace(",", "")
    except Exception:
        return str(v).strip()


def _fmt_date(v: Any) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)) or not str(v).strip():
        return ""
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(str(v).strip(), fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return str(v).strip()


# ─────────────────────────────────────────────────────────────────────────────
# OCR PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _gray_normalize(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


def _denoise(img: np.ndarray) -> np.ndarray:
    gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)


def _apply_preprocessing(img: np.ndarray) -> np.ndarray:
    return _denoise(_gray_normalize(img))


def _group_lines(df_ocr: pd.DataFrame, y_tol: int = 4) -> list[list[dict]]:
    if df_ocr.empty:
        return []
    words = df_ocr.sort_values(["top", "left"]).to_dict("records")
    lines: list[list[dict]] = []
    cur:   list[dict] = []
    for w in words:
        if not cur:
            cur.append(w); continue
        lt = min(x["top"] for x in cur)
        lb = max(x["top"] + max(x["height"], 1) for x in cur)
        if not (w["top"] + max(w["height"], 1) < lt - y_tol or w["top"] > lb + y_tol):
            cur.append(w)
        else:
            lines.append(sorted(cur, key=lambda x: x["left"]))
            cur = [w]
    if cur:
        lines.append(sorted(cur, key=lambda x: x["left"]))
    return lines


def _lines_to_json_light(lines: list[list[dict]], gap_threshold_px: int = 50) -> str:
    """Convert OCR lines to JSON-light format — NB03 winner over HTML and Markdown.

    Multi-column rows become lists, key:value pairs use the key as JSON key,
    single-column rows use r0/r1/... keys. Result is compact JSON (no spaces).
    """
    data: dict = {}
    row_idx = 0
    for line in lines:
        if not line:
            continue
        text = " ".join(w["text"] for w in sorted(line, key=lambda w: w["left"])).strip()
        if not text:
            continue
        # detect columns by gap
        segments: list[str] = []
        cur_seg = [line[0]["text"]]
        for i in range(1, len(line)):
            prev_right = line[i - 1]["left"] + line[i - 1].get("width", 0)
            curr_left  = line[i]["left"]
            if curr_left - prev_right > gap_threshold_px:
                segments.append(" ".join(cur_seg))
                cur_seg = [line[i]["text"]]
            else:
                cur_seg.append(line[i]["text"])
        segments.append(" ".join(cur_seg))

        if len(segments) >= 2:
            data[f"r{row_idx}"] = segments
        elif ":" in text:
            idx     = text.index(":")
            k_raw   = text[:idx].strip()
            v_raw   = text[idx + 1:].strip()
            k_clean = re.sub(r"[^\w\s]", "", k_raw, flags=re.UNICODE)
            k_clean = re.sub(r"\s+", "_", k_clean.strip())[:30].lower()
            if k_clean and v_raw:
                data[k_clean] = v_raw
            else:
                data[f"r{row_idx}"] = text
        else:
            data[f"r{row_idx}"] = text
        row_idx += 1
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _page_to_json_light(img: np.ndarray, y_offset: int = 0,
                        col_gap_px: int = 60) -> tuple[str, int]:
    proc = _apply_preprocessing(img)
    pil  = Image.fromarray(proc) if len(proc.shape) == 2 else \
           Image.fromarray(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
    df   = pytesseract.image_to_data(pil, config=FIXED_OCR_CONFIG, output_type=Output.DATAFRAME)
    df   = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df   = df[df["text"] != ""]
    df   = df[pd.to_numeric(df["conf"], errors="coerce").fillna(-1) >= FIXED_OCR_CONF_MIN]
    for col in ("left", "top", "width", "height"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["top"] = df["top"] + y_offset
    lines = _group_lines(df[["text", "left", "top", "width", "height", "conf"]])
    return _lines_to_json_light(lines), y_offset + img.shape[0]


def pdf_to_ocr_text(pdf_path: Path, dpi: int = OCR_DPI) -> str:
    """Render PDF pages, run Tesseract OCR, return JSON-light encoded text."""
    doc   = fitz.open(str(pdf_path))
    parts = []
    y_off = 0
    for pg in range(doc.page_count):
        page = doc[pg]
        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        text, y_off = _page_to_json_light(img, y_off)
        if text.strip():
            parts.append(f"<!-- page {pg + 1} -->\n{text}")
    doc.close()
    return "\n".join(parts) if parts else "[prazdny dokument]"


def pdf_to_images_b64(pdf_path: Path, dpi: int = VISION_DPI,
                      max_pages: int = VISION_MAX_PAGES) -> list[str]:
    """Render PDF pages to base64 PNG strings for Vision API.

    NB07: grayscale_denoise_en_standard F1=0.733 > raw_en_standard 0.727,
    latence 2801 ms vs 7872 ms — obrázky se předzpracovávají stejně jako OCR.
    """
    doc  = fitz.open(str(pdf_path))
    imgs = []
    for pg in range(min(doc.page_count, max_pages)):
        page = doc[pg]
        mat  = fitz.Matrix(dpi / 72, dpi / 72)
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        proc = _apply_preprocessing(img)
        ok, png = cv2.imencode(".png", proc)
        if not ok:
            png = np.frombuffer(pix.tobytes("png"), dtype=np.uint8)
        imgs.append(base64.b64encode(png.tobytes()).decode("ascii"))
    doc.close()
    return imgs


# ─────────────────────────────────────────────────────────────────────────────
# JSON SCHEMA + PYDANTIC
# ─────────────────────────────────────────────────────────────────────────────

def build_json_schema(fields: list[dict]) -> dict:
    props: dict[str, dict] = {}
    for f in fields:
        label = f.get("label_cs") or f["field_name"]
        unit  = f.get("unit", "")
        note  = f.get("note", "")
        desc  = label
        if unit:
            desc += f" [{unit}]"
        if f["field_type"] == "date":
            desc += " — return as YYYY-MM-DD string or null"
        elif f["field_type"] == "float":
            desc += " — return as numeric string with dot decimal, 2 decimal places, no units; null if not found"
        if note:
            desc += ". " + note
        props[f["field_name"]] = {"type": ["string", "null"], "description": desc}

    # Self-confidence object: all fields, so both required and optional fields are scored.
    # Required fields: escalate if confidence < FIELD_CONFIDENCE_THRESHOLD (0.65).
    # Optional fields: escalate only if value is non-null AND confidence < OPTIONAL_FIELD_CONFIDENCE_THRESHOLD (0.45).
    #   — an optional field with a non-null value and low confidence means "model guessed";
    #     that is worse than returning null (which is legitimate for optional fields).
    conf_props = {
        f["field_name"]: {
            "type": ["number", "null"],
            "description": (
                f"Confidence 0.0-1.0 for '{f['field_name']}' "
                f"({'required' if f.get('is_required') else 'optional'}). "
                "1.0=unambiguous. 0.0=null returned or value not found. Null if unknown."
            ),
        }
        for f in fields
    }
    all_conf_names = [f["field_name"] for f in fields]
    props["_confidence"] = {
        "type": "object",
        "description": (
            "Self-reported extraction confidence (0.0-1.0) for EVERY field. "
            "Required fields: be precise — used to decide Vision escalation. "
            "Optional fields: return 0.0 if null was returned, otherwise score your certainty. "
            "An optional field with a non-null value and low confidence is worse than null — be honest."
        ),
        "properties": conf_props,
        "required": all_conf_names,
        "additionalProperties": False,
    }

    all_required = [f["field_name"] for f in fields] + ["_confidence"]
    return {
        "name":   "invoice_extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": props,
            "required": all_required,
            "additionalProperties": False,
        },
    }


def build_pydantic_model(fields: list[dict]):
    fd = {
        f["field_name"]: (
            str | None,
            Field(None, description=f.get("label_en") or f["field_name"]),
        )
        for f in fields
    }
    return create_model("InvoiceExtraction", **fd)


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTY
# ─────────────────────────────────────────────────────────────────────────────

_COMMODITY_CTX_CS: dict[str, str] = {
    "elektrina_nn": (
        "Faktura je za dodávku elektrické energie nízkého napětí (NN). "
        "Obsahuje spotřebu VT (vysoký tarif) a NT (nízký tarif, pokud je dvousázkový odběr) v kWh. "
        "\n\n"
        "KRITICKÉ PRAVIDLO — amount_inc_vat vs Nedoplatek:\n"
        "Česká NN faktura zobrazuje DVA různé součty:\n"
        "  (A) Celková fakturovaná částka s DPH — hrubý součet všech položek PŘED odpočtem záloh.\n"
        "      Správné pole: 'Celkem s DPH', 'Celkové náklady', 'Souhrn částek celkem (s DPH)',\n"
        "      'Vyúčtování celkem (s DPH)', 'Rozdíl ke zdanění (celková řádka s DPH)'.\n"
        "      → Tato hodnota JDE do amount_inc_vat.\n"
        "  (B) Nedoplatek / K úhradě — výsledná platba PO odpočtu zálohovýcplateb.\n"
        "      'Nedoplatek', 'Doplatek', 'K úhradě', 'K úhradě celkem', 'Přeplatek'.\n"
        "      → Tato hodnota NIKDY nejde do amount_inc_vat!\n"
        "Pokud faktura nemá zálohy, (A) = (B). Ale pokud zálohy existují, jsou RŮZNÉ.\n"
        "Vždy použij (A) = hrubý součet z rekapitulační tabulky.\n"
        "\n"
        "KRITICKÉ PRAVIDLO — period_from/period_to vs ostatní data:\n"
        "Faktura obsahuje více dat: DUZP, Datum vystavení, Datum splatnosti, Odběrné období.\n"
        "  period_from / period_to = rozsah ODBĚRNÉHO OBDOBÍ, hledej jako 'DD.MM.YYYY - DD.MM.YYYY'\n"
        "  v záhlaví, v popisu faktury ('Vyúčtování za období ... - ...'), nebo v rekapitulaci.\n"
        "  issue_date = Datum vystavení (obvykle po period_to)\n"
        "  due_date = Datum splatnosti = 14-30 dní po issue_date\n"
        "\n"
        "KRITICKÉ PRAVIDLO — NT spotřeba:\n"
        "Jednosázkový odběr (D01/T1): pouze VT — vrať consumption_low_tariff = null.\n"
        "Dvousázkový odběr (D02/D25/T2): VT i NT — oba řádky v tabulce spotřeby."
    ),
    "elektrina_vn": (
        "Faktura je za dodávku elektrické energie vysokého napětí (VN). "
        "Spotřeba silové elektřiny je v MWh (nikoli kWh). "
        "Tarify jsou složité: komoditní složka, přenosová soustava, "
        "rezervované kapacity, jalová energie, POZE, systémové služby. "
        "Vyplň každé pole přesně dle popisu — rezervované kapacity jsou v kW, "
        "energie v MWh, sazby jsou Kč/MWh nebo Kč/kW."
    ),
    "plyn_mo": (
        "Faktura je za dodávku zemního plynu maloodběrateli (MO). "
        "\n\n"
        "STRUKTURA STRÁNEK faktury plyn MO (čti VŠECHNY stránky označené <!-- page N -->):\n"
        "  Str. 1: Souhrn — amount_inc_vat ('Cena celkem vč. DPH'), amount_ex_vat ('Cena celkem bez DPH'),\n"
        "           consumption_mwh ('přepočtené spotřebě energie ve výši X MWh'),\n"
        "           combustion_heat ('bylo použito spalné teplo XX,XXXX'),\n"
        "           period_from/to, issue_date, due_date, EAN, IČO.\n"
        "  Str. 2: Přehled odečtů — consumption_m3 ('Nepřepočtená spotřeba XX,XX m*').\n"
        "  Str. 3+: DETAILNÍ TABULKA — sloupce: m³ | Přepoč. koef. | Spalné teplo | MWh | Kč\n"
        "           → conversion_coefficient (sloupec 'Přepoč. koef.' ~1,000–1,100)\n"
        "           → combustion_heat (sloupec 'Spalné teplo' ~10,0–11,5 kWh/m³)\n"
        "           SEKCE 'DAŇ ZE ZEMNÍHO PLYNU:' s řádkem 'Celkem: XX,XX'\n"
        "           → gas_tax_total (součet VŠECH sekcí DAŇ pokud je více měřidel!)\n"
        "\n"
        "KRITICKÉ PRAVIDLO — formule a jednotky:\n"
        "  consumption_mwh = consumption_m3 × conversion_coefficient × combustion_heat / 1 000\n"
        "  (spalné teplo je v kWh/m³, proto dělíme 1 000, NE 3,6!)\n"
        "  conversion_coefficient ~1,000–1,100 — NIKDY 10,x (to je spalné teplo!)\n"
        "  consumption_mwh je v MWh (0,xxx nebo XXX,xx) — 'Dodané množství plynu XXXX kWh' je kWh → vyděl 1 000.\n"
        "  m³ v OCR → 'm*', 'm3', 'm?' — přečti číslo PŘED tímto artefaktem.\n"
        "  Daň ze ZP je ZVLÁŠTNÍ POLOŽKA — NENÍ součástí ceny komodity."
    ),
    "plyn_vo": (
        "Faktura je za dodávku zemního plynu velkoodběrateli (VO). "
        "Denní rezervovaná kapacita (daily_reserved_capacity) je v tisících m³/den (tis. m³/den), typicky 0,xxx. "
        "Spotřeba: consumption_m3 (naměřená) i consumption_mwh (přepočtená). "
        "SPRÁVNÁ FORMULA: consumption_mwh = consumption_m3 × conversion_coefficient × combustion_heat / 1 000 "
        "(spalné teplo v kWh/m³ ~10,0–11,5, přepočtový koeficient ~1,000–1,100 — nesměšuj tyto dvě hodnoty). "
        "gas_tax_total může být na 2.+ straně v detailní části — hledej na všech stránkách. "
        "Distribuce RK a distribuce služba jsou dvě různé položky — vyplň obě."
    ),
    "teplo": (
        "Faktura je za dálkové teplo (CZT). "
        "Spotřeba tepla (consumption_gj) je v GJ (gigajoule), nikoli kWh — 1 GJ ≈ 277,8 kWh. "
        "Na faktuře se označuje jako 'Teplo ostatní' nebo 'Teplo celkem' — hledej řádek s GJ jednotkou. "
        "TV = teplá voda (NE televize): hot_water_consumption je teplo pro ohřev TV, označeno 'TV ostatní'. "
        "Studená voda (cold_water_consumption, m³) = objem studené vody použité pro ohřev TV. "
        "Doplňovací voda (supplementary_water, m³) = náhrada ztrát v otopné soustavě. "
        "Rezervovaná kapacita (reserved_capacity) je v kW — OCR může zobrazit 'kWw' nebo 'kWv' místo 'kW'. "
        "Spotřeba m³ v OCR: m³ může být 'm*', 'm3' — čti číslo před ní."
    ),
    "voda": (
        "Faktura je za vodné a stočné. "
        "Vodné (consumption_vodne_m3, m³) = dodávka pitné vody — label: 'Vodné', 'Fakt. vodné', 'vodné odb.místo'. "
        "Stočné celkem (consumption_stocne_m3, m³) = odvod veškeré odpadní vody (odpadní + srážkové) — label: 'Stočné', 'Fakt. stočné'. "
        "Odpadní stočné (consumption_odpadni_m3) a srážkové stočné (consumption_srazkove_m3) jsou oddělené složky — vrať null pokud nejsou na faktuře zvlášť. "
        "OCR artefakt: m³ může být 'm*', 'm3', 'm ³' — čti číslo PŘED touto jednotkou. "
        "Pokud je vedle m³ hodnoty liter v závorce jako '(1 304 000 l)', ignoruj závorku — použij m³."
    ),
}

_COMMODITY_CTX_EN: dict[str, str] = {
    "elektrina_nn": (
        "This is a low-voltage electricity (NN) invoice. "
        "It lists high-tariff (VT/T1) and optionally low-tariff (NT/T2) consumption in kWh. "
        "\n\n"
        "CRITICAL — amount_inc_vat vs payment balance:\n"
        "Czech NN invoices show TWO different totals:\n"
        "  (A) Gross invoice total incl. VAT = 'Celkem s DPH', 'Celkové náklady', "
        "      'Souhrn částek celkem (s DPH)', 'Vyúčtování celkem (s DPH)'. "
        "      → This goes into amount_inc_vat.\n"
        "  (B) Payment balance after advance deduction = 'Nedoplatek', 'K úhradě', "
        "      'Doplatek', 'Přeplatek'. → NEVER use for amount_inc_vat.\n"
        "When zálohy (advance payments) exist, (A) > (B). Always use (A).\n"
        "\n"
        "CRITICAL — period dates:\n"
        "  period_from/period_to = billing period dates (DD.MM.YYYY - DD.MM.YYYY in header)\n"
        "  issue_date = Datum vystavení (after period_to)\n"
        "  due_date = Datum splatnosti / Doporučené datum úhrady (14-30 days after issue)\n"
        "\n"
        "CRITICAL — NT consumption:\n"
        "  Single-tariff (VT/T1 only): return consumption_low_tariff = null.\n"
        "  Dual-tariff (VT+NT): both rows in the consumption table."
    ),
    "elektrina_vn": (
        "This is a high-voltage electricity (VN) invoice. "
        "Active energy consumption is in MWh (not kWh). "
        "The invoice contains complex tariff components: commodity, transmission grid, "
        "reserved capacities (annual/monthly in kW), reactive power (MVArh), "
        "system services, POZE, non-network infrastructure. "
        "Extract each component into its specific field."
    ),
    "plyn_mo": (
        "This is a small-consumer natural gas (plyn MO) invoice. "
        "\n\n"
        "PAGE STRUCTURE (read ALL pages, each marked <!-- page N -->):\n"
        "  Page 1: Summary — amount_inc_vat ('Cena celkem vč. DPH'), amount_ex_vat ('Cena celkem bez DPH'),\n"
        "           consumption_mwh ('přepočtené spotřebě energie ve výši X MWh'),\n"
        "           combustion_heat ('bylo použito spalné teplo XX.XXXX'), dates, EAN, IČO.\n"
        "  Page 2: Meter readings — consumption_m3 ('Nepřepočtená spotřeba XX.XX m*').\n"
        "  Page 3+: DETAIL TABLE — columns: m³ | Přepoč. koef. | Spalné teplo | MWh | Kč\n"
        "           → conversion_coefficient (column 'Přepoč. koef.', value ~1.000–1.100)\n"
        "           → combustion_heat (column 'Spalné teplo', value ~10.0–11.5 kWh/m³)\n"
        "           Section 'DAŇ ZE ZEMNÍHO PLYNU:' → gas_tax_total\n"
        "           If multiple meters: SUM all gas tax sections.\n"
        "\n"
        "CRITICAL — formula and units:\n"
        "  consumption_mwh = consumption_m3 × conversion_coefficient × combustion_heat / 1000\n"
        "  (combustion_heat is kWh/m³ — divide by 1000 for MWh, NOT by 3.6!)\n"
        "  conversion_coefficient ~1.000–1.100 (NEVER ~10.x — that is combustion_heat!)\n"
        "  consumption_mwh in MWh — 'Dodané množství plynu XXXX kWh' is kWh → divide by 1000.\n"
        "  m³ in OCR → 'm*', 'm3', 'm?' — read the number BEFORE the artifact.\n"
        "  gas_tax is a SEPARATE line item — NOT part of commodity price."
    ),
    "plyn_vo": (
        "This is a large-consumer natural gas (plyn VO) invoice. "
        "Daily reserved capacity is in thousands of m³/day (tis. m³/den), typically 0.xxx. "
        "Both measured (m³) and converted (MWh) consumption are present. "
        "Distribution reserved capacity and distribution services are separate items."
    ),
    "teplo": (
        "This is a district heating (teplo/CZT) invoice. "
        "Consumption is in GJ (gigajoules), NOT kWh — 1 GJ ≈ 277.8 kWh. "
        "Hot water heat (TV) = heat energy for domestic hot water preparation in GJ. "
        "Cold water (m³) = cold water volume used for hot water heating. "
        "Supplementary water (m³) = make-up water for heating loop losses."
    ),
    "voda": (
        "This is a water and sewage invoice. "
        "Vodné (m³) = drinking water supply volume. "
        "Stočné celkem (m³) = total wastewater volume (may include both types). "
        "Odpadní stočné and srážkové stočné are separate sewage components."
    ),
}

_GLOBAL_RULES_CS = """\
Globální pravidla pro extrakci:
- Pokud hodnotu nelze spolehlivě určit, vrať null — NIKDY nehádej.
- Číselné hodnoty: tečka jako desetinný oddělovač, přesně 2 desetinná místa, bez jednotek (příklad: "1234.56").
- Česká desetinná čárka: číslo "10,7641" v dokumentu = hodnota 10.7641 (čárka je desetinný oddělovač).
- Datum: vždy YYYY-MM-DD (příklad: "2024-03-15"). Nikdy DD.MM.YYYY.
- amount_ex_vat: POUZE hrubý základ daně (Základ daně / Cena celkem bez DPH). NIKDY Nedoplatek/Doplatek/K úhradě/Záloha/Přeplatek.
- amount_inc_vat: POUZE hrubá celková fakturovaná částka vč. DPH PŘED odpočtem zálohovýcplateb. NIKDY Nedoplatek/Doplatek/K úhradě/Záloha/Přeplatek.
- Nedoplatek ≠ amount_inc_vat: Nedoplatek je výsledná platba PO záloze — jen pokud NEJSOU žádné zálohy, mohou být stejné.
- IČO: přesně 8 číslic. Rozliš IČO dodavatele od IČO odběratele — oba jsou na faktuře, vyber správný.
- due_date ≠ issue_date: splatnost je vždy POZDĚJŠÍ než datum vystavení (typicky 14–30 dní).
- period_from/period_to ≠ issue_date/due_date: odběrné období je PŘED datem vystavení.
- OCR text je ve formátu JSON-light: kompaktní JSON bez mezer. Vícekolumnové řádky jsou seznamy {"rN": ["sloupec1", "sloupec2"]}, klíč:hodnota páry jsou {"klíč": "hodnota"}, ostatní řádky {"rN": "text"}. Každá stránka je uvozena komentářem <!-- page N -->.
- Pokud je hodnota v tabulce (seznam), hledej správný prvek dle anchor vzoru.
- EAN kód odběrného místa: přesně 18 číslic, v ČR začíná 859182. Nekombinuj IČO s EAN.
- OCR artefakty jednotek: m³ může být zobrazeno jako "m*", "m3", nebo "m ³" — přečti číslo PŘED touto jednotkou.
- Kód odběrného místa pro teplo a vodu: může být kratší lokální kód (6–12 číslic), ne vždy 18-místné EAN.
- Pokud jsou hodnoty v závorce jako "(1 304 000 l)" = litrový ekvivalent, ignoruj — použij m³ hodnotu před závorkou.

Sebeohodnocení jistoty (_confidence):
- Pro každé povinné pole vrať skóre jistoty 0.0–1.0 v poli _confidence.
- 0.90–1.00: hodnota nalezena jednoznačně, přesný text/číslo je v dokumentu.
- 0.70–0.89: hodnota pravděpodobně správná, ale s drobnou nejistotou (OCR šum, zaokrouhlení).
- 0.50–0.69: hodnota nejistá — viděl jsem více kandidátů nebo byl popis nejednoznačný.
- 0.00–0.49: hodnota není spolehlivá nebo pole je null.
- Pokud vracíš null, vrať pro _confidence vždy 0.0.
- Buď upřímný: pokud jsi viděl více možností a musel odhadovat, uveď nižší skóre.
"""

_GLOBAL_RULES_EN = """\
Global extraction rules:
- Return null if a value cannot be determined with confidence — NEVER guess.
- Numbers: dot decimal separator, exactly 2 decimal places, no units (example: "1234.56").
- Czech decimal comma: the number "10,7641" in the document equals 10.7641 (comma is the decimal separator).
- Dates: always YYYY-MM-DD (example: "2024-03-15"). Never DD.MM.YYYY or MM/DD/YYYY.
- amount_ex_vat: ONLY the gross tax base BEFORE deducting advance payments. NEVER Nedoplatek/Doplatek/K úhradě/Přeplatek.
- amount_inc_vat: ONLY the gross invoice total incl. VAT BEFORE advance payment deduction. NEVER Nedoplatek/K úhradě/Přeplatek.
- 'Nedoplatek' ≠ amount_inc_vat: Nedoplatek is what remains to pay AFTER advances. Only equal if no advances exist.
- IČO: exactly 8 digits. Invoice shows BOTH supplier and customer IČO — pick the correct one. Supplier IČO is in the supplier/vendor section; customer IČO is in the Odběratel/Zákazník section.
- due_date is ALWAYS later than issue_date (typically 14-30 days after).
- period_from/period_to are ALWAYS before issue_date. Do not confuse with issue_date or DUZP.
- OCR text is in JSON-light format: compact JSON without spaces. Multi-column rows are lists {"rN": ["col1", "col2"]}, key:value pairs are {"key": "value"}, other rows are {"rN": "text"}. Each page is prefixed with <!-- page N -->.
- When a value is in a table (list element), find the correct element by its anchor pattern.
- EAN supply point code: exactly 18 digits, Czech EAN starts with 859182. For heat/water: may be a shorter local code (6-12 digits).
- OCR unit artifacts: m³ may appear as "m*", "m3", or "m ³" — read the number BEFORE this symbol.
- Liter conversions in parentheses like "(1 304 000 l)" should be ignored — use the m³ value before the parenthesis.
- Be aware of OCR errors: '0' vs 'O', '1' vs 'l', '5' vs 'S', '8' vs 'B'. Cross-check arithmetic when possible.

Self-confidence scoring (_confidence):
- For each REQUIRED field, return a confidence score 0.0–1.0 in the _confidence object.
- 0.90–1.00: value found unambiguously; exact text/number is present in the document.
- 0.70–0.89: value likely correct but with minor uncertainty (OCR noise, rounding).
- 0.50–0.69: uncertain — saw multiple candidates or ambiguous label.
- 0.00–0.49: value not reliable or field returned null.
- Always return 0.0 if the field value is null.
- Be honest: if you had to guess between candidates, use a lower score.
"""


def _field_list_text(fields: list[dict], use_cs: bool = True) -> str:
    lines = []
    for f in sorted(fields, key=lambda x: (0 if x.get("is_required") else 1, x["field_name"])):
        label  = f.get("label_cs" if use_cs else "label_en") or f["field_name"]
        req    = " [povinné]" if f.get("is_required") else " [volitelné]"
        anchor = str(f.get("anchor_cs" if use_cs else "anchor_en") or "").strip()
        unit   = f.get("unit", "")
        note   = f.get("note", "")
        line   = f"  • {label} ({f['field_name']}){req}"
        if unit:
            line += f" — {unit}"
        if anchor:
            line += f"\n    Hledej: {anchor}" if use_cs else f"\n    Search: {anchor}"
        if note:
            line += f"\n    Poznámka: {note}" if use_cs else f"\n    Note: {note}"
        lines.append(line)
    return "Extrahuj přesně tato pole:\n" + "\n".join(lines) if use_cs else \
           "Extract exactly these fields:\n" + "\n".join(lines)


def build_system_prompt_text(commodity: str, fields: list[dict],
                             few_shot: str = "") -> str:
    role = (
        "Jsi specialista na extrakci strukturovaných dat z českých faktur za energie a vodu. "
        "Pracuješ s OCR textem ve formátu JSON-light: kompaktní JSON, kde vícekolumnové řádky "
        "jsou uloženy jako seznam [sloupec1, sloupec2, ...], klíč:hodnota páry jako {klíč: hodnota} "
        "a ostatní řádky jako {rN: text}. Každá stránka je oddělena komentářem <!-- page N -->."
    )
    ctx              = _COMMODITY_CTX_CS.get(commodity, "")
    flist            = _field_list_text(fields, use_cs=True)
    few_shot_section = f"{few_shot}\n" if few_shot else ""
    return f"{role}\n\n{ctx}\n\n{few_shot_section}{_GLOBAL_RULES_CS}\n{flist}"


def build_user_prompt_text(html_text: str) -> str:
    return (
        "Analyzuj následující OCR text faktury (JSON-light formát) a vrať požadovaná pole jako JSON.\n\n"
        + html_text
    )


def build_system_prompt_vision(commodity: str, fields: list[dict], few_shot: str = "") -> str:
    role = (
        "You are a specialist in extracting structured data from scanned Czech utility invoices. "
        "You are reading raw invoice page images. "
        "The invoice may be in Czech or partially German/Slovak."
    )
    ctx              = _COMMODITY_CTX_EN.get(commodity, "")
    flist            = _field_list_text(fields, use_cs=False)
    few_shot_section = f"{few_shot}\n" if few_shot else ""
    return f"{role}\n\n{ctx}\n\n{few_shot_section}{_GLOBAL_RULES_EN}\n{flist}"


def build_user_prompt_vision(commodity: str) -> str:
    return (
        f"Extract the required fields from this Czech utility invoice "
        f"(commodity: {commodity}). Return JSON only."
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM CALLS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_llm_response(raw: str, fields: list[dict]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Returns (extracted_values, confidence_dict).
    confidence_dict has float scores (0-1) for required fields only.
    """
    empty      = {f["field_name"]: None for f in fields}
    all_names  = [f["field_name"] for f in fields]
    empty_conf = {fn: None for fn in all_names}

    def _extract(payload: dict) -> tuple[dict, dict]:
        confidence = payload.pop("_confidence", None) or {}
        # Normalise all field confidence values
        conf_out = {
            fn: (float(confidence[fn]) if fn in confidence and confidence[fn] is not None else None)
            for fn in all_names
        }
        model_cls = build_pydantic_model(fields)
        extracted = model_cls(**{k: v for k, v in payload.items()}).model_dump()
        return extracted, conf_out

    try:
        payload = json.loads(raw)
        return _extract(payload)
    except Exception:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                payload = json.loads(match.group())
                return _extract(payload)
            except Exception:
                pass
        return empty, empty_conf


def call_llm_text(html_text: str, model_cfg: dict,
                  commodity: str, fields: list[dict],
                  json_schema: dict) -> dict[str, Any]:
    empty     = {f["field_name"]: None for f in fields}
    few_shot  = _retriever.get_examples(html_text, n=2) if _retriever and html_text else ""
    sysp      = build_system_prompt_text(commodity, fields, few_shot=few_shot)
    usrp      = build_user_prompt_text(html_text)
    try:
        if openai_client is None:
            raise RuntimeError("OpenAI client not initialised")
        t0   = time.time()
        resp = openai_client.chat.completions.create(
            model=model_cfg["model_id"],
            messages=[
                {"role": "system", "content": sysp},
                {"role": "user",   "content": usrp},
            ],
            response_format={"type": "json_schema", "json_schema": json_schema},
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_TEXT,
            logprobs=True,   # vrací log-pravděpodobnosti výstupních tokenů
        )
        lat = (time.time() - t0) * 1000
        raw = resp.choices[0].message.content or "{}"
        pt, ct = resp.usage.prompt_tokens, resp.usage.completion_tokens
        extracted, confidence = _parse_llm_response(raw, fields)

        # Global logprob confidence — secondary signal (per-field confidence is primary).
        # Filters structural JSON tokens so only value-token log-probs contribute.
        mean_lp: float | None = None
        choice_lps = resp.choices[0].logprobs
        if choice_lps and choice_lps.content:
            _structural = set('{}[]:",' + " \n")
            value_lps = [
                t.logprob for t in choice_lps.content
                if t.token.strip() and t.token.strip() not in _structural
                and not t.token.strip().startswith('"')
            ]
            if value_lps:
                mean_lp = sum(value_lps) / len(value_lps)

        return {
            "extracted": extracted, "confidence": confidence,
            "prompt_tokens": pt, "completion_tokens": ct,
            "latency_ms": lat, "api_error": None, "mode": "text",
            "mean_logprob": mean_lp,
        }
    except Exception as e:
        all_names = [f["field_name"] for f in fields]
        return {"extracted": empty, "confidence": {fn: None for fn in all_names},
                "prompt_tokens": 0, "completion_tokens": 0,
                "latency_ms": 0.0, "api_error": str(e), "mode": "text",
                "mean_logprob": None}


def call_llm_vision(images_b64: list[str], model_cfg: dict,
                    commodity: str, fields: list[dict],
                    json_schema: dict,
                    ocr_text: str = "") -> dict[str, Any]:
    empty = {f["field_name"]: None for f in fields}
    if not images_b64:
        return {"extracted": empty, "prompt_tokens": 0, "completion_tokens": 0,
                "latency_ms": 0.0, "api_error": "no images", "mode": "vision"}
    few_shot = _retriever.get_examples(ocr_text, n=2) if _retriever and ocr_text else ""
    sysp = build_system_prompt_vision(commodity, fields, few_shot=few_shot)
    # Build user message with images
    content: list[dict] = [
        {"type": "text", "text": build_user_prompt_vision(commodity)}
    ]
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url":    f"data:image/png;base64,{b64}",
                "detail": "high",
            },
        })
    try:
        if openai_client is None:
            raise RuntimeError("OpenAI client not initialised")
        t0   = time.time()
        resp = openai_client.chat.completions.create(
            model=model_cfg["model_id"],
            messages=[
                {"role": "system", "content": sysp},
                {"role": "user",   "content": content},
            ],
            response_format={"type": "json_schema", "json_schema": json_schema},
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS_VISION,
        )
        lat = (time.time() - t0) * 1000
        raw = resp.choices[0].message.content or "{}"
        pt, ct = resp.usage.prompt_tokens, resp.usage.completion_tokens
        extracted, confidence = _parse_llm_response(raw, fields)
        return {"extracted": extracted, "confidence": confidence,
                "prompt_tokens": pt, "completion_tokens": ct,
                "latency_ms": lat, "api_error": None, "mode": "vision",
                "mean_logprob": None}
    except Exception as e:
        all_names = [f["field_name"] for f in fields]
        return {"extracted": empty, "confidence": {fn: None for fn in all_names},
                "prompt_tokens": 0, "completion_tokens": 0,
                "latency_ms": 0.0, "api_error": str(e), "mode": "vision",
                "mean_logprob": None}


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACE
# ─────────────────────────────────────────────────────────────────────────────

def normalize_value(field: str, value: Any, field_type: str = "string") -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    is_float = field_type == "float"
    is_date  = field_type == "date"
    if is_float:
        text = (text.replace("Kč", "").replace("Kc", "").replace("CZK", "")
                    .replace("\xa0", "").replace(" ", "").replace(" ", ""))
        if re.match(r"^\d{1,3}(?:\.\d{3})+,\d{2}$", text):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", ".")
        text = re.sub(r"[^0-9.\-]", "", text)
        try:
            return f"{float(text):.2f}"
        except Exception:
            return text
    if is_date:
        for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return text
    return re.sub(r"\s+", "", text).lower()


def compute_cost(model_cfg: dict, pt: int, ct: int) -> float:
    return (pt * model_cfg["price_in"] + ct * model_cfg["price_out"]) / 1_000_000


def _to_float_or_none(value: Any) -> float | None:
    """Parse a Czech/plain numeric string (or number) to float, else None."""
    norm = normalize_value("_", value, "float")
    try:
        return float(norm)
    except (ValueError, TypeError):
        return None


def _ref_date_for_vat(extracted: dict) -> date_cls | None:
    """VAT reference date: issue_date, falling back to period_from (Fix 3)."""
    for fld in ("issue_date", "period_from"):
        norm = normalize_value(fld, extracted.get(fld), "date")
        try:
            return datetime.strptime(norm, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# VALIDACE HODNOT — detekce nesprávných (ne jen chybějících) hodnot
# ─────────────────────────────────────────────────────────────────────────────
#
# Problém: model může vrátit špatnou hodnotu s plnou jistotou.
# null_rate to nedetekuje. Potřebujeme tři doplňkové vrstvy:
#
#   A) Logprob confidence  — průměrná log-pravděpodobnost výstupních tokenů.
#      Nižší logprob = model méně jistý. Pokud mean_logprob < prahu, eskaluj.
#      Výpočet: OpenAI API vrací logprobs per token; bereme průměr VALUE tokenů.
#      Threshold: -0.40 → exp(-0.40) ≈ 0.67 průměrná pravděpodobnost tokenu.
#
#   B) Formátová validace — kritická pole musejí splňovat pevná pravidla.
#      IČO: přesně 8 číslic. Datum: validní YYYY-MM-DD.
#      Pořadí: period_from < period_to, issue_date ≤ due_date.
#      Částky: amount_inc_vat ≥ amount_ex_vat.
#
#   C) Aritmetická konzistence — doménová pravidla pro kvantitativní pole.
#      Plyn: MWh ≈ m³ × koef × spalné_teplo_kWh/m³ / 1000  (tolerance 15 %)
#      Spalné teplo: 7.5–13.5 kWh/m³ (fyzikální meze pro zemní plyn v ČR)
#      Přepočtový koeficient: 0.88–1.15 (normované meze)
#      tg φ (VN): 0.0–3.0 (fyzikální meze)

LOGPROB_CONFIDENCE_THRESHOLD          = -0.40  # global objective signal — escalate if mean_logprob below this
FIELD_CONFIDENCE_THRESHOLD_REQUIRED   =  0.65  # required fields — escalate if self-confidence below this
FIELD_CONFIDENCE_THRESHOLD_OPTIONAL   =  0.45  # optional fields — escalate only if non-null AND below this
                                                # (model returned something it strongly doubts = worse than null)


def _safe_float(val: Any) -> float | None:
    if not val:
        return None
    try:
        return float(str(val).replace(",", ".").strip())
    except (ValueError, TypeError):
        return None


def _safe_date(val: Any) -> datetime | None:
    if not val:
        return None
    try:
        return datetime.strptime(str(val).strip(), "%Y-%m-%d")
    except ValueError:
        return None


def check_formats(extracted: dict) -> list[str]:
    """Formátová validace. Vrátí seznam popsaných porušení (prázdný = OK)."""
    errors: list[str] = []

    # IČO: přesně 8 číslic
    for fld in ("supplier_tax_id", "customer_tax_id"):
        val = extracted.get(fld)
        if val and not re.fullmatch(r"\d{8}", str(val).strip()):
            errors.append(f"{fld}='{val}' není 8 číslic")

    # Datum: validní YYYY-MM-DD
    parsed_dates: dict[str, datetime] = {}
    for fld in ("issue_date", "due_date", "period_from", "period_to", "tax_point_date"):
        val = extracted.get(fld)
        if val:
            d = _safe_date(val)
            if d is None:
                errors.append(f"{fld}='{val}' není validní YYYY-MM-DD")
            else:
                parsed_dates[fld] = d

    # Pořadí dat
    if "period_from" in parsed_dates and "period_to" in parsed_dates:
        if parsed_dates["period_from"] >= parsed_dates["period_to"]:
            errors.append(
                f"period_from({extracted['period_from']}) >= period_to({extracted['period_to']})"
            )
    if "issue_date" in parsed_dates and "due_date" in parsed_dates:
        if parsed_dates["issue_date"] > parsed_dates["due_date"]:
            errors.append(
                f"issue_date({extracted['issue_date']}) > due_date({extracted['due_date']})"
            )

    # Částky: incl. VAT ≥ excl. VAT (DPH je nenegativní)
    ex  = _safe_float(extracted.get("amount_ex_vat"))
    inc = _safe_float(extracted.get("amount_inc_vat"))
    if ex is not None and inc is not None and ex > 0:
        if inc < ex * 0.97:  # 3% tolerance pro zaokrouhlení
            errors.append(
                f"amount_inc_vat({inc}) < amount_ex_vat({ex}) — DPH nemůže být záporná"
            )

    return errors


def check_arithmetic(extracted: dict, commodity: str) -> list[str]:
    """Doménová aritmetická konzistence. Vrátí seznam porušení (prázdný = OK)."""
    errors: list[str] = []

    if commodity in ("plyn_mo", "plyn_vo"):
        m3   = _safe_float(extracted.get("consumption_m3"))
        mwh  = _safe_float(extracted.get("consumption_mwh"))
        coef = _safe_float(extracted.get("conversion_coefficient"))
        heat = _safe_float(extracted.get("combustion_heat"))

        # Přepočtový koeficient: fyzikální meze (normované pro ČR, dle vyhlášky č. 251/2001 Sb.)
        if coef is not None and not (0.88 <= coef <= 1.15):
            errors.append(f"conversion_coefficient={coef} mimo fyzikální rozsah [0.88, 1.15]")

        # Spalné teplo: průmyslové zemní plyn v ČR je typicky 10.0–11.5 kWh/m³
        if heat is not None and not (7.5 <= heat <= 13.5):
            errors.append(f"combustion_heat={heat} mimo rozsah [7.5, 13.5] kWh/m³")

        # Konzistence: MWh ≈ m³ × koeficient × spalné_teplo_kWh/m³ / 1000
        # POZOR: spalné_teplo je v kWh/m³ (~10.5), NIKOLI v MJ/m³ (~38.8)
        # Proto dělíme 1000 (kWh→MWh), NE 3.6 (to by bylo pro MJ→kWh přepočet)
        if m3 and mwh and coef and heat and m3 > 0:
            # Odmítnout fyzikálně nesmyslné hodnoty spalného tepla (MJ/m³ vs kWh/m³ záměna)
            if heat > 20:
                # Model zřejmě vrátil MJ/m³ (~38.8) místo kWh/m³ — konvertuj
                heat_kwh = heat / 3.6
            else:
                heat_kwh = heat
            expected = m3 * coef * heat_kwh / 1000
            rel_err  = abs(mwh - expected) / expected if expected > 0 else 1.0
            if rel_err > 0.15:  # 15% tolerance (faktura může mít více odběrných míst, zaokrouhlení)
                errors.append(
                    f"gas_mwh_mismatch: {m3}m³×{coef}×{heat_kwh:.4f}kWh/m³/1000"
                    f"={expected:.4f}MWh, got {mwh} (err={rel_err*100:.1f}%)"
                )

    if commodity == "elektrina_vn":
        pf = _safe_float(extracted.get("power_factor"))
        if pf is not None and not (0.0 <= pf <= 3.0):
            errors.append(f"power_factor(tg φ)={pf} mimo rozsah [0, 3]")

    if commodity == "teplo":
        gj = _safe_float(extracted.get("consumption_gj"))
        if gj is not None and (gj <= 0 or gj > 100_000):
            errors.append(f"consumption_gj={gj} mimo rozumný rozsah (0, 100000]")

    return errors


# ─────────────────────────────────────────────────────────────────────────────
# KASKÁDA — PRIMARY text → FALLBACK Vision
# ─────────────────────────────────────────────────────────────────────────────

def _check_cross_field_required(extracted: dict, commodity: str) -> list[str]:
    """Alespoň jedno pole ze skupiny musí být non-null.
    Vrátí seznam popisů porušení (prázdný = OK).
    """
    groups = _CROSS_FIELD_AT_LEAST_ONE.get(commodity, [])
    violations = []
    for group in groups:
        if all(not extracted.get(f) for f in group):
            violations.append("at_least_one_null:" + "+".join(group))
    return violations


def _check_required_confidence(confidence: dict, commodity: str) -> list[str]:
    """Required fields below FIELD_CONFIDENCE_THRESHOLD_REQUIRED."""
    if not confidence:
        return []
    low = []
    for fname in _CASCADE_REQUIRED.get(commodity, ["invoice_number"]):
        val = confidence.get(fname)
        if val is not None:
            try:
                if float(val) < FIELD_CONFIDENCE_THRESHOLD_REQUIRED:
                    low.append(f"{fname}={float(val):.2f}")
            except (TypeError, ValueError):
                pass
    return low


def _check_optional_confidence(confidence: dict, extracted: dict,
                                fields: list[dict]) -> list[str]:
    """Optional fields that have a non-null value but very low self-confidence.

    A non-null value with confidence < FIELD_CONFIDENCE_THRESHOLD_OPTIONAL means the model
    returned something it strongly doubts — worse than returning null, which is legitimate.
    """
    if not confidence:
        return []
    req_set = {f["field_name"] for f in fields if f.get("is_required")}
    low = []
    for f in fields:
        fname = f["field_name"]
        if fname in req_set:
            continue                          # required fields checked separately
        if not extracted.get(fname):
            continue                          # null/empty — that's fine for optional
        val = confidence.get(fname)
        if val is not None:
            try:
                if float(val) < FIELD_CONFIDENCE_THRESHOLD_OPTIONAL:
                    low.append(f"{fname}={float(val):.2f}")
            except (TypeError, ValueError):
                pass
    return low


def should_escalate(extracted: dict, commodity: str,
                    api_error: str | None = None,
                    mean_logprob: float | None = None,
                    field_confidence: dict | None = None,
                    fields: list[dict] | None = None) -> tuple[bool, str]:
    """
    Vrátí (escalate: bool, reason: str).

    Kritéria v pořadí priority:
      1. api_error                 — primární model API selhal
      2. required_missing          — povinné pole je null
      3. high_null_rate            — komoditně specifický práh překročen
      4. low_logprob               — mean_logprob < -0.40
                                     (objektivní globální signál; přijde dřív než
                                      self-confidence protože není self-reported)
      5. low_required_confidence   — self-reported confidence < 0.65 na povinném poli
      6. low_optional_confidence   — optional pole má non-null hodnotu a confidence < 0.45
                                     (model vrátil hodnotu, které sám nevěří → horší než null)
      7. format_error              — IČO, datum, VAT inkonzistence
      8. arith_error               — doménová aritmetická inkonzistence
    """
    fc = field_confidence or {}

    # 1. API selhání
    if api_error:
        return True, "api_error"

    # 2. Chybí klíčové povinné pole
    req     = _CASCADE_REQUIRED.get(commodity, ["invoice_number"])
    missing = [f for f in req if not extracted.get(f)]
    if missing:
        return True, f"required_missing:{','.join(missing)}"

    # 2.5. Cross-field: alespoň jedno pole ze skupiny musí být non-null
    cross = _check_cross_field_required(extracted, commodity)
    if cross:
        return True, f"cross_field:{cross[0]}"

    # 3. Komoditně specifická null_rate
    n_null    = sum(1 for v in extracted.values() if not v)
    null_rate = n_null / len(extracted) if extracted else 1.0
    threshold = _CASCADE_NULL_THRESHOLDS.get(commodity, _CASCADE_NULL_THRESHOLD_DEFAULT)
    if null_rate > threshold:
        return True, f"high_null:{null_rate:.2f}>{threshold}"

    # 4. Globální logprob — objektivní signál (API, není self-reported)
    if mean_logprob is not None and mean_logprob < LOGPROB_CONFIDENCE_THRESHOLD:
        return True, f"low_logprob:{mean_logprob:.3f}<{LOGPROB_CONFIDENCE_THRESHOLD}"

    # 5. Per-field self-confidence — povinná pole
    low_req = _check_required_confidence(fc, commodity)
    if low_req:
        return True, f"low_required_conf:{','.join(low_req)}"

    # 6. Per-field self-confidence — volitelná pole s non-null hodnotou
    low_opt = _check_optional_confidence(fc, extracted, fields or [])
    if low_opt:
        return True, f"low_optional_conf:{','.join(low_opt)}"

    # 7. Formátová validace
    fmt_errs = check_formats(extracted)
    if fmt_errs:
        return True, f"format_error:{fmt_errs[0][:80]}"

    # 8. Aritmetická konzistence
    arith_errs = check_arithmetic(extracted, commodity)
    if arith_errs:
        return True, f"arith_error:{arith_errs[0][:80]}"

    return False, "ok"


def cascade_extract(html_text: str, pdf_path: Path,
                    commodity: str, fields: list[dict],
                    json_schema: dict) -> tuple[dict, bool, float, str, str]:
    """Returns: (result_dict, escalated, total_cost_usd, mode_used, escalation_reason)"""
    # PRIMARY: GPT-4.1 Apr-25, text, cs_en few-shot RAG n=2
    res_primary  = call_llm_text(html_text, PRIMARY_MODEL, commodity, fields, json_schema)
    cost_primary = compute_cost(PRIMARY_MODEL, res_primary["prompt_tokens"],
                                res_primary["completion_tokens"])

    escalate, reason = should_escalate(
        res_primary["extracted"], commodity,
        api_error=res_primary["api_error"],
        mean_logprob=res_primary.get("mean_logprob"),
        field_confidence=res_primary.get("confidence"),
        fields=fields,
    )

    # Track LLM latencies separately so the report can split OCR vs LLM time
    # and primary vs fallback path.
    res_primary["text_llm_ms"]   = res_primary["latency_ms"]
    res_primary["vision_llm_ms"] = 0.0

    if not escalate:
        return res_primary, False, cost_primary, "text", "ok"

    # FALLBACK: gpt-4.1-mini, Vision, standard
    try:
        images_b64 = pdf_to_images_b64(pdf_path)
    except Exception as e:
        images_b64 = []
        print(f"    [WARN] Vision render failed: {e}", flush=True)

    res_vision  = call_llm_vision(images_b64, VISION_MODEL, commodity, fields, json_schema,
                                  ocr_text=html_text)
    cost_vision = compute_cost(VISION_MODEL, res_vision["prompt_tokens"],
                               res_vision["completion_tokens"])
    total_cost  = cost_primary + cost_vision
    # Full pipeline spans both the primary text call and the vision fallback.
    res_vision["text_llm_ms"]   = res_primary["latency_ms"]
    res_vision["vision_llm_ms"] = res_vision["latency_ms"]
    return res_vision, True, total_cost, "vision", reason


# ─────────────────────────────────────────────────────────────────────────────
# VSTUPNÍ BOD PRO JEDEN DOKUMENT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CascadeResult:
    """Výsledek zpracování jedné faktury kaskádovou pipeline."""

    commodity: str
    fields: dict[str, Any]           # extrahovaná pole po normalizaci
    confidence: dict[str, float]     # jistota modelu po jednotlivých polích
    escalated: bool                  # použila se záložní (Vision) cesta?
    mode: str                        # "text" | "vision"
    escalation_reason: str
    cost_usd: float
    ocr_ms: float
    llm_ms: float
    format_errors: list[str]
    arith_errors: list[str]
    api_error: str | None

    @property
    def mean_confidence(self) -> float:
        hodnoty = [v for v in self.confidence.values() if v is not None]
        return round(sum(hodnoty) / len(hodnoty), 4) if hodnoty else 0.0


def detect_commodity(ocr_text: str) -> str | None:
    """Odhad komodity z textu faktury.

    V provozu komoditu zadává uživatel při nahrání dokumentu — tohle je jen
    záloha pro případ, že ji neuvede. Rozlišit velkoodběr od maloodběru
    (a vysoké napětí od nízkého) podle textu spolehlivě nelze, proto se u plynu
    a elektřiny vrací varianta s širší sadou polí; ta ostatní pole nechá prázdná
    místo aby o ně přišla.
    """
    from src.core.extraction.regex_strategy import RegexExtractionStrategy

    zjisteno = RegexExtractionStrategy()._detect_commodity(ocr_text)
    return zjisteno.value if zjisteno else None


def extract_invoice(pdf_path: str | Path, commodity: str | None = None) -> CascadeResult:
    """Zpracuje jednu fakturu toutéž kaskádou, jaká je vyhodnocena v práci.

    Args:
        pdf_path: cesta k PDF.
        commodity: klíč komodity (``elektrina_nn``, ``plyn_MO``, …). Není-li
            uveden, odhadne se z textu — viz :func:`detect_commodity`.

    Raises:
        ValueError: nepodaří-li se komoditu určit ani odhadnout.
    """
    pdf_path = Path(pdf_path)

    t0 = time.time()
    ocr_text = pdf_to_ocr_text(pdf_path)
    ocr_ms = (time.time() - t0) * 1000

    if commodity is None:
        commodity = detect_commodity(ocr_text)
    if commodity is None:
        raise ValueError(
            "Komoditu se nepodařilo určit z textu faktury. Předejte ji "
            "parametrem commodity (elektrina_nn, elektrina_vn, plyn_MO, "
            "plyn_VO, teplo, voda)."
        )

    fields = load_commodity_fields(commodity)
    schema = build_json_schema(fields)

    vysledek, eskalovano, cena, rezim, duvod = cascade_extract(
        ocr_text, pdf_path, commodity, fields, schema,
    )

    extrahovano = {
        f["field_name"]: normalize_value(
            f["field_name"], vysledek["extracted"].get(f["field_name"]), f.get("field_type", "string"),
        )
        for f in fields
    }

    return CascadeResult(
        commodity=commodity,
        fields=extrahovano,
        confidence=vysledek.get("confidence") or {},
        escalated=eskalovano,
        mode=rezim,
        escalation_reason=duvod,
        cost_usd=cena,
        ocr_ms=round(ocr_ms, 1),
        llm_ms=round(vysledek.get("text_llm_ms", 0.0) + vysledek.get("vision_llm_ms", 0.0), 1),
        format_errors=check_formats(vysledek["extracted"]),
        arith_errors=check_arithmetic(vysledek["extracted"], commodity),
        api_error=vysledek.get("api_error"),
    )
