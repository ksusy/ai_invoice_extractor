"""
Generátor syntetických faktur pro testování nových layoutů.

Vytvoří 6 fiktivních dodavatelů (jeden za každou komoditu) × 3 úrovně kvality
× 9 faktur = 162 PDF.

Výstup:
    data/synthetic/{komodita}/{dodavatel}/{Q1,Q2,Q3}/faktura_NNN.pdf

Komodity a dodavatelé:
    elektrina_nn  — NovaTech Energie a.s.   (dvoupásmový layout s tabulkou)
    elektrina_vn  — VoltPro Energie a.s.    (vícetarifní VN faktura)
    plyn_mo       — BohemiaGas s.r.o.       (formulářový layout s EIC kódem)
    plyn_vo       — InduGas s.r.o.          (vícesložkové VO vyúčtování)
    teplo         — ThermoCity a.s.         (rámečkový formát s DPH rekapitulací)
    voda          — AquaRegion s.r.o.       (klasický dopisní styl)

Úrovně kvality (všechny tři jsou rastrované skeny — bez textové vrstvy):
    Q1 — čistý scan (200 DPI, bez artefaktů)
    Q2 — průměrný scan (150 DPI, mírné rozmazání, šum, drobné zkosení)
    Q3 — špatný scan (120 DPI, rozmazání, šum, razítko, zkosení, nízký kontrast)
         Text je čitelný, ale OCR bude obtížné.

Vedlejší soubor:
    data/synthetic/ground_truth.json — pravdivé hodnoty pro všechny faktury

Prerekvizity:
    pip install reportlab pillow pypdfium2
"""

from __future__ import annotations

import io
import math
import random
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

# ── Cesty ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR      = PROJECT_ROOT / "data" / "synthetic"

# ── Fonty (DejaVu = plná podpora češtiny) ─────────────────────────────────────
FONT_PATHS = [
    Path("C:/Windows/Fonts/DejaVuSans.ttf"),
    Path("C:/Windows/Fonts/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
]

def _register_fonts() -> tuple[str, str]:
    for p in FONT_PATHS:
        if "Bold" not in p.name and p.exists():
            bold = p.parent / p.name.replace("DejaVuSans", "DejaVuSans-Bold")
            if bold.exists():
                pdfmetrics.registerFont(TTFont("DV",      str(p)))
                pdfmetrics.registerFont(TTFont("DV-Bold", str(bold)))
                return "DV", "DV-Bold"
    return "Helvetica", "Helvetica-Bold"

FONT, FONT_BOLD = _register_fonts()


# ── Fiktivní zákazníci ────────────────────────────────────────────────────────
CUSTOMERS = [
    {
        "name":    "Základní škola Nový Svět, příspěvková organizace",
        "street":  "Školní 42/1",
        "city":    "400 01 Ústí nad Labem",
        "ico":     "00262812",
        "dic":     "CZ00262812",
        "account": "19-1234567890/0100",
    },
    {
        "name":    "Mateřská škola Sluníčko Teplice, p.o.",
        "street":  "Dětská 7",
        "city":    "415 01 Teplice",
        "ico":     "00271624",
        "dic":     "CZ00271624",
        "account": "27-9876543210/0800",
    },
    {
        "name":    "Domov seniorů Pohoda, příspěvková organizace",
        "street":  "Klidná 15/3",
        "city":    "430 01 Chomutov",
        "ico":     "00314021",
        "dic":     "CZ00314021",
        "account": "35-1122334455/0600",
    },
]

# Fakturační období: 3 zákazníci × 3 období = 9 kombinací
PERIODS = [
    (date(2024,  1, 1), date(2024,  6, 30)),
    (date(2024,  7, 1), date(2024, 12, 31)),
    (date(2025,  1, 1), date(2025,  6, 30)),
]


# ── Dodavatelé ────────────────────────────────────────────────────────────────
@dataclass
class Supplier:
    key:           str
    name:          str
    short:         str
    address:       str
    city:          str
    ico:           str
    dic:           str
    iban:          str
    account:       str
    email:         str
    phone:         str
    commodity:     str     # zobrazovaný název komodity
    commodity_key: str     # složka v data/synthetic/
    unit:          str     # kWh | m³ | GJ
    color:         tuple   # RGB 0–1

SUPPLIERS: list[Supplier] = [
    Supplier(
        key="novatech", name="NovaTech Energie a.s.", short="NTE",
        address="Nová 1234/5, 110 00 Praha 1 – Nové Město", city="110 00 Praha 1",
        ico="12345678", dic="CZ12345678",
        iban="CZ12 0100 0000 0012 3456 7890", account="12345678/0100",
        email="fakturace@novatech-energie.cz", phone="+420 224 000 100",
        commodity="elektřina NN", commodity_key="elektrina_nn",
        unit="kWh", color=(0.08, 0.35, 0.65),
    ),
    Supplier(
        key="voltpro", name="VoltPro Energie a.s.", short="VPE",
        address="Průmyslová 88/3, 102 00 Praha 10 – Hostivař", city="102 00 Praha 10",
        ico="23456789", dic="CZ23456789",
        iban="CZ23 0300 0000 0023 4567 8900", account="23456789/0300",
        email="vn@voltpro-energie.cz", phone="+420 234 000 200",
        commodity="elektřina VN", commodity_key="elektrina_vn",
        unit="kWh", color=(0.82, 0.38, 0.05),
    ),
    Supplier(
        key="bohemiagas", name="BohemiaGas s.r.o.", short="BGS",
        address="Plynová 22/8, 130 00 Praha 3 – Žižkov", city="130 00 Praha 3",
        ico="34567890", dic="CZ34567890",
        iban="CZ34 0600 0000 0034 5678 9000", account="34567890/0600",
        email="fakturace@bohemiagas.cz", phone="+420 222 000 300",
        commodity="plyn MO", commodity_key="plyn_mo",
        unit="kWh", color=(0.25, 0.18, 0.55),
    ),
    Supplier(
        key="indugas", name="InduGas s.r.o.", short="IGS",
        address="Závodní 412/7, 434 01 Most", city="434 01 Most",
        ico="45678901", dic="CZ45678901",
        iban="CZ45 0800 0000 0045 6789 0100", account="45678901/0800",
        email="vo@indugas.cz", phone="+420 476 000 400",
        commodity="plyn VO", commodity_key="plyn_vo",
        unit="kWh", color=(0.08, 0.42, 0.38),
    ),
    Supplier(
        key="thermocity", name="ThermoCity a.s.", short="TCY",
        address="Tepelná 789/2, 301 00 Plzeň", city="301 00 Plzeň",
        ico="56781234", dic="CZ56781234",
        iban="CZ56 0300 0000 0056 7812 3400", account="56781234/0300",
        email="vyuctovani@thermocity.cz", phone="+420 377 000 300",
        commodity="teplo", commodity_key="teplo",
        unit="GJ", color=(0.75, 0.20, 0.10),
    ),
    Supplier(
        key="aquaregion", name="AquaRegion s.r.o.", short="ARG",
        address="Vodní 56/3, 360 01 Karlovy Vary", city="360 01 Karlovy Vary",
        ico="87654321", dic="CZ87654321",
        iban="CZ87 0800 0000 0087 6543 2100", account="87654321/0800",
        email="info@aquaregion.cz", phone="+420 353 000 200",
        commodity="voda", commodity_key="voda",
        unit="m³", color=(0.10, 0.55, 0.45),
    ),
    # ── elektrina_nn — dodavatelé 2 a 3 ──────────────────────────────────────
    Supplier(
        key="energyplus", name="EnergyPlus s.r.o.", short="EPL",
        address="Obchodní 77/2, 602 00 Brno – Středí", city="602 00 Brno",
        ico="22334455", dic="CZ22334455",
        iban="CZ22 0100 0000 0022 3344 5500", account="22334455/0100",
        email="fakturace@energyplus.cz", phone="+420 543 000 200",
        commodity="elektřina NN", commodity_key="elektrina_nn",
        unit="kWh", color=(0.05, 0.50, 0.42),
    ),
    Supplier(
        key="sparkelit", name="SparkElit a.s.", short="SPE",
        address="Elektrárenská 3/14, 412 01 Litoměřice", city="412 01 Litoměřice",
        ico="33445566", dic="CZ33445566",
        iban="CZ33 0600 0000 0033 4455 6600", account="33445566/0600",
        email="info@sparkelit.cz", phone="+420 416 000 300",
        commodity="elektřina NN", commodity_key="elektrina_nn",
        unit="kWh", color=(0.38, 0.12, 0.58),
    ),
    # ── elektrina_vn — dodavatelé 2 a 3 ──────────────────────────────────────
    Supplier(
        key="highvolt", name="HighVolt Energy a.s.", short="HVE",
        address="Průmyslová 100/5, 301 00 Plzeň", city="301 00 Plzeň",
        ico="44556677", dic="CZ44556677",
        iban="CZ44 0300 0000 0044 5566 7700", account="44556677/0300",
        email="vn@highvolt.cz", phone="+420 377 100 400",
        commodity="elektřina VN", commodity_key="elektrina_vn",
        unit="kWh", color=(0.20, 0.35, 0.62),
    ),
    Supplier(
        key="elnord", name="ElNord a.s.", short="ELN",
        address="Severní 18/6, 460 01 Liberec", city="460 01 Liberec",
        ico="55667788", dic="CZ55667788",
        iban="CZ55 0800 0000 0055 6677 8800", account="55667788/0800",
        email="obchod@elnord.cz", phone="+420 485 000 500",
        commodity="elektřina VN", commodity_key="elektrina_vn",
        unit="kWh", color=(0.15, 0.22, 0.40),
    ),
    # ── plyn_mo — dodavatelé 2 a 3 ───────────────────────────────────────────
    Supplier(
        key="gaspraha", name="GasPraha s.r.o.", short="GPR",
        address="Václavská 12/3, 110 00 Praha 1", city="110 00 Praha 1",
        ico="66778899", dic="CZ66778899",
        iban="CZ66 0100 0000 0066 7788 9900", account="66778899/0100",
        email="fakturace@gaspraha.cz", phone="+420 224 100 600",
        commodity="plyn MO", commodity_key="plyn_mo",
        unit="kWh", color=(0.72, 0.34, 0.05),
    ),
    Supplier(
        key="termoplyn", name="TermoPlyn a.s.", short="TPL",
        address="Plynárenská 5/7, 326 00 Plzeň – Doubravka", city="326 00 Plzeň",
        ico="77889900", dic="CZ77889900",
        iban="CZ77 0300 0000 0077 8899 0000", account="77889900/0300",
        email="odber@termoplyn.cz", phone="+420 377 200 700",
        commodity="plyn MO", commodity_key="plyn_mo",
        unit="kWh", color=(0.10, 0.42, 0.22),
    ),
    # ── plyn_vo — dodavatelé 2 a 3 ───────────────────────────────────────────
    Supplier(
        key="progas_vo", name="ProGas VO a.s.", short="PGV",
        address="Závodní 200/1, 703 00 Ostrava – Vítkovice", city="703 00 Ostrava",
        ico="88990011", dic="CZ88990011",
        iban="CZ88 0800 0000 0088 9900 1100", account="88990011/0800",
        email="vo@progas.cz", phone="+420 596 000 800",
        commodity="plyn VO", commodity_key="plyn_vo",
        unit="kWh", color=(0.12, 0.22, 0.52),
    ),
    Supplier(
        key="gasind", name="GasInd s.r.o.", short="GIN",
        address="Hutní 33/2, 434 01 Most", city="434 01 Most",
        ico="99001122", dic="CZ99001122",
        iban="CZ99 0100 0000 0099 0011 2200", account="99001122/0100",
        email="info@gasind.cz", phone="+420 476 100 900",
        commodity="plyn VO", commodity_key="plyn_vo",
        unit="kWh", color=(0.35, 0.40, 0.10),
    ),
    # ── teplo — dodavatelé 2 a 3 ──────────────────────────────────────────────
    Supplier(
        key="termoplus", name="TermoPlus s.r.o.", short="TPS",
        address="Teplárenská 8/4, 434 01 Most", city="434 01 Most",
        ico="11223344", dic="CZ11223344",
        iban="CZ11 0300 0000 0011 2233 4400", account="11223344/0300",
        email="vyuctovani@termoplus.cz", phone="+420 476 200 100",
        commodity="teplo", commodity_key="teplo",
        unit="GJ", color=(0.55, 0.28, 0.10),
    ),
    Supplier(
        key="heatworks", name="HeatWorks a.s.", short="HWK",
        address="Průmyslová 44/1, 415 01 Teplice", city="415 01 Teplice",
        ico="22334411", dic="CZ22334411",
        iban="CZ22 0600 0000 0022 3344 1100", account="22334411/0600",
        email="billing@heatworks.cz", phone="+420 417 000 200",
        commodity="teplo", commodity_key="teplo",
        unit="GJ", color=(0.72, 0.38, 0.02),
    ),
    # ── voda — dodavatelé 2 a 3 ───────────────────────────────────────────────
    Supplier(
        key="aquatown", name="AquaTown a.s.", short="ATW",
        address="Nábřežní 21/3, 400 01 Ústí nad Labem", city="400 01 Ústí n. L.",
        ico="33441122", dic="CZ33441122",
        iban="CZ33 0100 0000 0033 4411 2200", account="33441122/0100",
        email="voda@aquatown.cz", phone="+420 475 000 300",
        commodity="voda", commodity_key="voda",
        unit="m³", color=(0.15, 0.40, 0.65),
    ),
    Supplier(
        key="clearwater", name="ClearWater s.r.o.", short="CLW",
        address="Čistá 7/2, 430 01 Chomutov", city="430 01 Chomutov",
        ico="44112233", dic="CZ44112233",
        iban="CZ44 0800 0000 0044 1122 3300", account="44112233/0800",
        email="info@clearwater.cz", phone="+420 474 000 400",
        commodity="voda", commodity_key="voda",
        unit="m³", color=(0.05, 0.55, 0.65),
    ),
]


# ── Generátor číselných dat faktury ──────────────────────────────────────────

def _invoice_amounts(supplier: Supplier, period_start: date, period_end: date,
                     rng: random.Random) -> dict[str, Any]:
    """Vygeneruje realistické finanční a měrné hodnoty faktury."""

    def fmt_d(d: date) -> str:
        if sys.platform == "win32":
            return d.strftime("%#d. %#m. %Y")
        return d.strftime("%-d. %-m. %Y")

    ck = supplier.commodity_key

    if ck == "elektrina_nn":
        consumption   = rng.randint(1500, 12000)
        unit_price    = rng.uniform(4.5, 6.5)
        vat_rate      = 0.21
        amount_ex_vat = round(consumption * unit_price, 2)
        extra: dict   = {}

    elif ck == "elektrina_vn":
        # Vícetarifní vyúčtování VN: rezervovaný příkon + VT + NT + sys. služby
        demand_kw     = rng.randint(50, 400)
        demand_price  = rng.uniform(15.0, 28.0)   # Kč/kW/měsíc
        demand_total  = round(demand_kw * demand_price * 6, 2)
        consumption   = rng.randint(30000, 180000)
        kWh_vt        = round(consumption * rng.uniform(0.55, 0.70))
        kWh_nt        = consumption - kWh_vt
        price_vt      = rng.uniform(3.5, 5.5)
        price_nt      = rng.uniform(1.8, 3.0)
        price_ss      = rng.uniform(0.15, 0.35)   # systémové služby Kč/kWh
        vt_total      = round(kWh_vt * price_vt, 2)
        nt_total      = round(kWh_nt * price_nt, 2)
        ss_total      = round(consumption * price_ss, 2)
        amount_ex_vat = round(demand_total + vt_total + nt_total + ss_total, 2)
        unit_price    = round(amount_ex_vat / consumption, 4)  # průměrná
        vat_rate      = 0.21
        extra = {
            "demand_kw":    demand_kw,   "demand_price": round(demand_price, 2),
            "demand_total": demand_total,
            "kWh_vt":       kWh_vt,      "price_vt":     round(price_vt, 4),
            "vt_total":     vt_total,
            "kWh_nt":       kWh_nt,      "price_nt":     round(price_nt, 4),
            "nt_total":     nt_total,
            "price_ss":     round(price_ss, 4), "ss_total": ss_total,
        }

    elif ck == "plyn_mo":
        consumption   = rng.randint(2000, 30000)
        unit_price    = rng.uniform(1.8, 3.2)
        vat_rate      = 0.21
        amount_ex_vat = round(consumption * unit_price, 2)
        extra         = {}

    elif ck == "plyn_vo":
        # Vícesložkové VO: komodita + distribuce pevná + distribuce variabilní
        consumption        = rng.randint(50000, 400000)
        unit_price         = rng.uniform(1.2, 2.0)         # Kč/kWh komodita
        distrib_fixed_mo   = rng.uniform(600.0, 2000.0)    # Kč/měsíc
        distrib_var_price  = rng.uniform(0.05, 0.12)       # Kč/kWh distribuce
        commodity_total    = round(consumption * unit_price, 2)
        distrib_fixed_tot  = round(distrib_fixed_mo * 6, 2)
        distrib_var_tot    = round(consumption * distrib_var_price, 2)
        amount_ex_vat      = round(commodity_total + distrib_fixed_tot + distrib_var_tot, 2)
        vat_rate           = 0.21
        extra = {
            "commodity_total":    commodity_total,
            "distrib_fixed_mo":   round(distrib_fixed_mo, 2),
            "distrib_fixed_total": distrib_fixed_tot,
            "distrib_var_price":  round(distrib_var_price, 4),
            "distrib_var_total":  distrib_var_tot,
        }

    elif ck == "voda":
        consumption   = rng.randint(80, 600)
        unit_price    = rng.uniform(68.0, 95.0)
        vat_rate      = 0.12
        amount_ex_vat = round(consumption * unit_price, 2)
        extra         = {}

    else:  # teplo
        consumption   = round(rng.uniform(10.0, 80.0), 2)
        unit_price    = rng.uniform(450.0, 750.0)
        vat_rate      = 0.21
        amount_ex_vat = round(consumption * unit_price, 2)
        extra         = {}

    vat_amount      = round(amount_ex_vat * vat_rate, 2)
    amount_with_vat = round(amount_ex_vat + vat_amount, 2)
    advances        = round(amount_with_vat * rng.uniform(0.7, 1.3) / 100) * 100
    balance         = round(amount_with_vat - advances, 2)
    invoice_date    = period_end + timedelta(days=rng.randint(5, 20))
    due_date        = invoice_date + timedelta(days=14)
    inv_number      = f"{supplier.short}{invoice_date.year}{rng.randint(100000, 999999)}"
    vs              = str(rng.randint(1000000000, 9999999999))

    result: dict[str, Any] = {
        "invoice_number":  inv_number,
        "invoice_date":    invoice_date.strftime("%d. %m. %Y"),
        "due_date":        due_date.strftime("%d. %m. %Y"),
        "period_start":    fmt_d(period_start),
        "period_end":      fmt_d(period_end),
        "consumption":     consumption,
        "unit":            supplier.unit,
        "unit_price":      unit_price,
        "vat_rate":        int(vat_rate * 100),
        "amount_ex_vat":   amount_ex_vat,
        "vat_amount":      vat_amount,
        "amount_with_vat": amount_with_vat,
        "advances":        advances,
        "balance":         balance,
        "variable_symbol": vs,
    }
    result.update(extra)
    return result


# ── Pomocné funkce pro renderování ────────────────────────────────────────────

def _fmt(n: float) -> str:
    """Formátuje číslo s mezerami jako tisícovými separátory (český styl)."""
    return f"{n:,.2f}".replace(",", "\xa0").replace(".", ",")


# ── Renderovací funkce ────────────────────────────────────────────────────────

def render_novatech(c_obj: canvas.Canvas, sup: Supplier,
                    cust: dict, inv: dict) -> None:
    """Layout A — NovaTech (elektřina NN): barevný záhlaví pás, dvousloupcový blok."""
    W, H = A4
    R, G, B = sup.color

    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, H - 28*mm, W, 28*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1)
    c_obj.setFont(FONT_BOLD, 16)
    c_obj.drawString(15*mm, H - 12*mm, sup.name)
    c_obj.setFont(FONT, 9)
    c_obj.drawString(15*mm, H - 20*mm, sup.address)
    c_obj.drawRightString(W - 15*mm, H - 12*mm, f"IČO: {sup.ico}  |  DIČ: {sup.dic}")
    c_obj.drawRightString(W - 15*mm, H - 20*mm, f"{sup.phone}  |  {sup.email}")

    c_obj.setFillColorRGB(R, G, B)
    c_obj.setFont(FONT_BOLD, 12)
    c_obj.drawString(15*mm, H - 38*mm,
                     f"FAKTURA – DAŇOVÝ DOKLAD č. {inv['invoice_number']}")

    y = H - 50*mm
    col1, col2 = 15*mm, W/2 + 5*mm
    for col, title, lines in [
        (col1, "DODAVATEL", [sup.name, sup.address,
                             f"IČO: {sup.ico}", f"DIČ: {sup.dic}",
                             f"IBAN: {sup.iban}"]),
        (col2, "ODBĚRATEL", [cust["name"], cust["street"], cust["city"],
                             f"IČO: {cust['ico']}", f"DIČ: {cust['dic']}"]),
    ]:
        c_obj.setFillColorRGB(R, G, B)
        c_obj.setFont(FONT_BOLD, 8)
        c_obj.drawString(col, y, title)
        c_obj.setFillColorRGB(0.2, 0.2, 0.2)
        c_obj.setFont(FONT, 8)
        for i, ln in enumerate(lines):
            c_obj.drawString(col, y - (i+1)*5*mm, ln)

    y2 = y - 38*mm
    c_obj.setFillColorRGB(0.2, 0.2, 0.2)
    for label, val in [
        ("Datum vystavení:", inv["invoice_date"]),
        ("Datum splatnosti:", inv["due_date"]),
        ("Fakturační období:", f"{inv['period_start']} – {inv['period_end']}"),
        ("Variabilní symbol:", inv["variable_symbol"]),
    ]:
        c_obj.setFont(FONT_BOLD, 8); c_obj.drawString(col1, y2, label)
        c_obj.setFont(FONT, 8);      c_obj.drawString(col1 + 45*mm, y2, val)
        y2 -= 5.5*mm

    y3 = y2 - 8*mm
    headers = ["Popis", "Množství", "Jedn.", "Cena/j. bez DPH", "DPH %", "Částka s DPH"]
    widths  = [65*mm, 22*mm, 12*mm, 35*mm, 14*mm, 32*mm]
    xs = [15*mm]
    for w in widths[:-1]:
        xs.append(xs[-1] + w)
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(15*mm, y3 - 6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
    for hdr, x in zip(headers, xs):
        c_obj.drawString(x + 1*mm, y3 - 4*mm, hdr)

    desc = f"Dodávka {sup.commodity} za období {inv['period_start']} – {inv['period_end']}"
    row  = [desc, str(inv["consumption"]), inv["unit"],
            _fmt(inv["unit_price"]) + " Kč", f"{inv['vat_rate']} %",
            _fmt(inv["amount_with_vat"]) + " Kč"]
    c_obj.setFillColorRGB(0.95, 0.95, 0.95)
    c_obj.rect(15*mm, y3 - 13*mm, W - 30*mm, 7*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
    for val, x in zip(row, xs):
        c_obj.drawString(x + 1*mm, y3 - 10*mm, val[:30])

    y4 = y3 - 25*mm
    bx, bw, bh = W - 80*mm, 65*mm, 38*mm
    c_obj.setFillColorRGB(0.97, 0.97, 0.97)
    c_obj.setStrokeColorRGB(R, G, B)
    c_obj.roundRect(bx, y4 - bh, bw, bh, 3*mm, fill=1, stroke=1)
    c_obj.setFillColorRGB(0.2, 0.2, 0.2); c_obj.setFont(FONT, 8)
    for i, (lbl, val) in enumerate([
        ("Základ daně:",     _fmt(inv["amount_ex_vat"])  + " Kč"),
        (f"DPH {inv['vat_rate']} %:", _fmt(inv["vat_amount"]) + " Kč"),
        ("Celkem s DPH:",    _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Přijaté zálohy:", f"–{_fmt(inv['advances'])} Kč"),
    ]):
        c_obj.drawString(bx + 3*mm,     y4 - 7*mm - i*6.5*mm, lbl)
        c_obj.drawRightString(bx + bw - 3*mm, y4 - 7*mm - i*6.5*mm, val)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 10)
    lbl = "NEDOPLATEK:" if inv["balance"] > 0 else "PŘEPLATEK:"
    c_obj.drawString(bx + 3*mm,     y4 - 34*mm, lbl)
    c_obj.drawRightString(bx + bw - 3*mm, y4 - 34*mm, _fmt(abs(inv["balance"])) + " Kč")

    c_obj.setFillColorRGB(0.6, 0.6, 0.6); c_obj.setFont(FONT, 7)
    c_obj.drawCentredString(W/2, 12*mm,
        f"{sup.name}  |  IBAN: {sup.iban}  |  Číslo účtu: {sup.account}")
    c_obj.drawCentredString(W/2, 8*mm,
        f"Strana 1 z 1  |  Variabilní symbol: {inv['variable_symbol']}")


def render_voltpro(c_obj: canvas.Canvas, sup: Supplier,
                   cust: dict, inv: dict) -> None:
    """
    Layout E — VoltPro Energie (elektřina VN):
    Oranžové záhlaví, informační pás s EAN + napěťová úroveň,
    vícetarifní tabulka (rez. příkon / VT / NT / systémové služby).
    """
    W, H = A4
    R, G, B = sup.color

    # ── záhlaví ───────────────────────────────────────────────────────────────
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, H - 26*mm, W, 26*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1)
    c_obj.setFont(FONT_BOLD, 15)
    c_obj.drawString(15*mm, H - 11*mm, sup.name)
    c_obj.setFont(FONT, 8)
    c_obj.drawString(15*mm, H - 19*mm, sup.address)
    c_obj.drawRightString(W - 15*mm, H - 11*mm,
                          f"IČO: {sup.ico}  |  DIČ: {sup.dic}")
    c_obj.drawRightString(W - 15*mm, H - 19*mm,
                          f"{sup.phone}  |  {sup.email}")

    # ── název dokumentu ───────────────────────────────────────────────────────
    c_obj.setFillColorRGB(R, G, B)
    c_obj.setFont(FONT_BOLD, 12)
    c_obj.drawString(15*mm, H - 36*mm,
                     f"VN FAKTURA – DAŇOVÝ DOKLAD č. {inv['invoice_number']}")
    c_obj.setFont(FONT, 8.5)
    c_obj.setFillColorRGB(0.3, 0.3, 0.3)
    c_obj.drawString(15*mm, H - 43*mm,
                     f"Datum vystavení: {inv['invoice_date']}   "
                     f"Datum splatnosti: {inv['due_date']}   "
                     f"VS: {inv['variable_symbol']}")

    # ── dvousloupcový blok dodavatel / odběratel ──────────────────────────────
    bw = (W - 35*mm) / 2
    bh = 36*mm
    by = H - 84*mm

    def box_section(x: float, title: str, lines: list[str]) -> None:
        c_obj.setStrokeColorRGB(0.70, 0.70, 0.70)
        c_obj.setLineWidth(0.4)
        c_obj.rect(x, by, bw, bh, fill=0, stroke=1)
        c_obj.setFillColorRGB(R, G, B)
        c_obj.rect(x, by + bh - 5*mm, bw, 5*mm, fill=1, stroke=0)
        c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
        c_obj.drawString(x + 2*mm, by + bh - 3.5*mm, title)
        c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
        for i, ln in enumerate(lines):
            c_obj.drawString(x + 2*mm, by + bh - 11*mm - i*4.5*mm, ln)

    box_section(15*mm, "DODAVATEL",
                [sup.name, sup.address,
                 f"IČO: {sup.ico}", f"DIČ: {sup.dic}", f"IBAN: {sup.iban}"])
    box_section(15*mm + bw + 5*mm, "ODBĚRATEL",
                [cust["name"], cust["street"], cust["city"],
                 f"IČO: {cust['ico']}", f"DIČ: {cust['dic']}"])

    # ── EAN / smlouva pás ─────────────────────────────────────────────────────
    ean = f"859182400000{cust['ico'][:8]}"
    y_ean = by - 8*mm
    c_obj.setFillColorRGB(0.96, 0.93, 0.88)
    c_obj.rect(15*mm, y_ean - 6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.2, 0.2, 0.2); c_obj.setFont(FONT, 7.5)
    c_obj.drawString(17*mm, y_ean - 4*mm,
                     f"EAN: {ean}   Napěťová úroveň: VN   "
                     f"Smlouva č.: {cust['ico'][:6]}VN   "
                     f"Fakturační období: {inv['period_start']} – {inv['period_end']}")

    # ── tabulka vícetarifních položek ─────────────────────────────────────────
    y3 = y_ean - 18*mm
    col_x = [15*mm, 88*mm, 113*mm, 137*mm, 160*mm]
    hdrs  = ["Položka", "Množství", "Jedn.", "Cena bez DPH", "Celkem Kč"]

    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(15*mm, y3 - 6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
    for hdr, x in zip(hdrs, col_x):
        c_obj.drawString(x + 1*mm, y3 - 4*mm, hdr)

    months = 6
    rows = [
        (f"Distribuce – rezervovaný příkon {inv.get('demand_kw', '—')} kW × {months} měs.",
         str(inv.get('demand_kw', '—')), "kW",
         f"{_fmt(inv.get('demand_price', 0))} Kč/kW/měs",
         _fmt(inv.get('demand_total', 0))),
        (f"Činná energie VT",
         str(inv.get('kWh_vt', '—')), "kWh",
         f"{_fmt(inv.get('price_vt', 0))} Kč/kWh",
         _fmt(inv.get('vt_total', 0))),
        (f"Činná energie NT",
         str(inv.get('kWh_nt', '—')), "kWh",
         f"{_fmt(inv.get('price_nt', 0))} Kč/kWh",
         _fmt(inv.get('nt_total', 0))),
        (f"Systémové služby",
         str(inv['consumption']), "kWh",
         f"{_fmt(inv.get('price_ss', 0))} Kč/kWh",
         _fmt(inv.get('ss_total', 0))),
    ]
    for ri, row in enumerate(rows):
        bg = (0.97, 0.95, 0.92) if ri % 2 == 0 else (1, 1, 1)
        c_obj.setFillColorRGB(*bg)
        c_obj.rect(15*mm, y3 - 13*mm - ri*6.5*mm, W - 30*mm, 6.5*mm, fill=1, stroke=0)
        c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
        for val, x in zip(row, col_x):
            c_obj.drawString(x + 1*mm, y3 - 10*mm - ri*6.5*mm, str(val)[:32])

    # ── DPH souhrn ────────────────────────────────────────────────────────────
    y4 = y3 - 48*mm
    bx, bw2, bh2 = W - 82*mm, 67*mm, 44*mm
    c_obj.setFillColorRGB(0.97, 0.97, 0.97)
    c_obj.setStrokeColorRGB(R, G, B)
    c_obj.roundRect(bx, y4 - bh2, bw2, bh2, 2.5*mm, fill=1, stroke=1)
    c_obj.setFillColorRGB(0.2, 0.2, 0.2); c_obj.setFont(FONT, 8)
    for i, (lbl, val) in enumerate([
        ("Základ daně (DPH 21 %):", _fmt(inv["amount_ex_vat"]) + " Kč"),
        ("DPH 21 %:",               _fmt(inv["vat_amount"])    + " Kč"),
        ("Celkem s DPH:",           _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Přijaté zálohy:",        f"–{_fmt(inv['advances'])} Kč"),
    ]):
        c_obj.drawString(bx + 3*mm,      y4 - 7*mm - i*7*mm, lbl)
        c_obj.drawRightString(bx + bw2 - 3*mm, y4 - 7*mm - i*7*mm, val)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    lbl = "K ÚHRADĚ:" if inv["balance"] > 0 else "PŘEPLATEK:"
    c_obj.drawString(bx + 3*mm,          y4 - 38*mm, lbl)
    c_obj.drawRightString(bx + bw2 - 3*mm, y4 - 38*mm,
                          _fmt(abs(inv["balance"])) + " Kč")

    # ── patička ───────────────────────────────────────────────────────────────
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, 0, W, 10*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT, 6.5)
    c_obj.drawCentredString(W/2, 3.5*mm,
        f"{sup.name}  ·  IČO {sup.ico}  ·  DIČ {sup.dic}  ·  "
        f"IBAN {sup.iban}  ·  {sup.email}")


def render_bohemiagas(c_obj: canvas.Canvas, sup: Supplier,
                      cust: dict, inv: dict) -> None:
    """
    Layout D — BohemiaGas (plyn MO):
    Záhlaví pás, čtyři informační boxy, tabulka spotřeby s převodem m³→kWh.
    """
    W, H = A4
    R, G, B = sup.color

    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, H - 22*mm, W, 22*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1)
    c_obj.setFont(FONT_BOLD, 13)
    c_obj.drawString(15*mm, H - 10*mm, sup.name)
    c_obj.setFont(FONT, 8)
    c_obj.drawString(15*mm, H - 17*mm,
                     f"{sup.address}  ·  IČO: {sup.ico}  ·  DIČ: {sup.dic}")
    c_obj.drawRightString(W - 15*mm, H - 10*mm, sup.phone)
    c_obj.drawRightString(W - 15*mm, H - 17*mm, sup.email)

    c_obj.setFillColorRGB(R, G, B)
    c_obj.setFont(FONT_BOLD, 11)
    c_obj.drawString(15*mm, H - 31*mm, "VYÚČTOVÁNÍ ZA DODÁVKU ZEMNÍHO PLYNU")
    c_obj.setFont(FONT, 8.5)
    c_obj.setFillColorRGB(0.3, 0.3, 0.3)
    c_obj.drawString(15*mm, H - 38*mm,
                     f"Faktura č. {inv['invoice_number']}   "
                     f"Vystaveno: {inv['invoice_date']}   "
                     f"Splatnost: {inv['due_date']}")

    y_row = H - 58*mm
    bw4 = (W - 35*mm) / 4
    boxes = [
        ("ZÁKAZNÍK",      [cust["name"][:28], cust["street"], cust["city"]]),
        ("ODBĚRNÉ MÍSTO", [f"EIC: 27ZG{cust['ico'][:8]}P",
                           cust["street"], cust["city"]]),
        ("SMLOUVA",       [f"Č. smlouvy: {cust['ico'][:6]}00",
                           f"Zákaznické č.: {cust['ico']}",
                           "Kategorie: MO"]),
        ("PLATBA",        [f"IBAN: {sup.iban[:22]}",
                           f"VS: {inv['variable_symbol'][:10]}",
                           f"Účet: {sup.account}"]),
    ]
    c_obj.setStrokeColorRGB(0.75, 0.75, 0.75)
    c_obj.setLineWidth(0.4)
    for i, (title, lines) in enumerate(boxes):
        bx = 15*mm + i * (bw4 + 5/3*mm)
        c_obj.setFillColorRGB(R, G, B)
        c_obj.rect(bx, y_row - 22*mm, bw4, 22*mm, fill=0, stroke=1)
        c_obj.rect(bx, y_row - 5*mm,  bw4,  5*mm, fill=1, stroke=0)
        c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7)
        c_obj.drawString(bx + 1.5*mm, y_row - 3.3*mm, title)
        c_obj.setFillColorRGB(0.15, 0.15, 0.15); c_obj.setFont(FONT, 6.8)
        for j, ln in enumerate(lines):
            c_obj.drawString(bx + 1.5*mm, y_row - 9*mm - j*4.2*mm, ln[:28])

    y3 = y_row - 32*mm
    col_x = [15*mm, 60*mm, 85*mm, 110*mm, 135*mm, 160*mm]
    hdrs  = ["Položka", "Spotřeba kWh", "Kč/kWh", f"DPH {inv['vat_rate']} %",
             "Základ DPH", "Celkem Kč"]
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(15*mm, y3 - 6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
    for hdr, x in zip(hdrs, col_x):
        c_obj.drawString(x + 1*mm, y3 - 4*mm, hdr)

    m3_approx = round(inv["consumption"] / 10.55, 1)
    rows = [
        (f"Zemní plyn {inv['period_start']}–{inv['period_end']}",
         str(inv["consumption"]), _fmt(inv["unit_price"]),
         f"{inv['vat_rate']} %", _fmt(inv["amount_ex_vat"]),
         _fmt(inv["amount_with_vat"])),
        (f"Přepočet: {m3_approx} m³ × 10,55 = {inv['consumption']} kWh",
         "", "", "", "", ""),
    ]
    for ri, row in enumerate(rows):
        bg = (0.95, 0.94, 0.98) if ri == 0 else (1, 1, 1)
        c_obj.setFillColorRGB(*bg)
        c_obj.rect(15*mm, y3 - 13*mm - ri*6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
        c_obj.setFillColorRGB(0.1, 0.1, 0.1)
        c_obj.setFont(FONT, ri == 0 and 7.5 or 6.8)
        for val, x in zip(row, col_x):
            c_obj.drawString(x + 1*mm, y3 - 10*mm - ri*6*mm, val)

    y4 = y3 - 32*mm
    c_obj.setFillColorRGB(0.1, 0.1, 0.1)
    for i, (lbl, val) in enumerate([
        ("Celková fakturovaná částka bez DPH:", _fmt(inv["amount_ex_vat"]) + " Kč"),
        (f"DPH {inv['vat_rate']} %:",           _fmt(inv["vat_amount"])    + " Kč"),
        ("Celkem včetně DPH:",                  _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Uhrazené zálohy:",                   f"–{_fmt(inv['advances'])} Kč"),
    ]):
        c_obj.setFont(FONT, 8)
        c_obj.drawRightString(W - 60*mm, y4 - i*5.5*mm, lbl)
        c_obj.setFont(FONT_BOLD, 8)
        c_obj.drawRightString(W - 15*mm, y4 - i*5.5*mm, val)

    y5 = y4 - 28*mm
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(1.2)
    c_obj.line(W - 80*mm, y5 + 4*mm, W - 15*mm, y5 + 4*mm)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 12)
    lbl = "K ÚHRADĚ:" if inv["balance"] > 0 else "PŘEPLATEK:"
    c_obj.drawRightString(W - 60*mm, y5, lbl)
    c_obj.drawRightString(W - 15*mm, y5, _fmt(abs(inv["balance"])) + " Kč")

    c_obj.setFillColorRGB(0.88, 0.88, 0.92)
    c_obj.rect(15*mm, y5 - 18*mm, 28*mm, 18*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.4, 0.4, 0.5); c_obj.setFont(FONT, 6)
    c_obj.drawCentredString(29*mm, y5 - 10*mm, "QR platba")

    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, 0, W, 9*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT, 6.5)
    c_obj.drawCentredString(W/2, 3*mm,
        f"{sup.name}  ·  IČO {sup.ico}  ·  DIČ {sup.dic}  ·  "
        f"IBAN {sup.iban}  ·  {sup.email}")


def render_indugas(c_obj: canvas.Canvas, sup: Supplier,
                   cust: dict, inv: dict) -> None:
    """
    Layout F — InduGas (plyn VO):
    Profesionální záhlaví, EIC kód pro VO, vícesložková fakturace
    (komodita + distribuce pevná + distribuce variabilní).
    """
    W, H = A4
    R, G, B = sup.color

    # ── záhlaví ───────────────────────────────────────────────────────────────
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, H - 8*mm, W, 8*mm, fill=1, stroke=0)   # horní pruh
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7)
    c_obj.drawCentredString(W/2, H - 5*mm,
                            "DAŇOVÝ DOKLAD – VYÚČTOVÁNÍ ZEMNÍHO PLYNU VO")

    c_obj.setFillColorRGB(0.1, 0.1, 0.1)
    c_obj.setFont(FONT_BOLD, 14)
    c_obj.drawString(15*mm, H - 18*mm, sup.name)
    c_obj.setFont(FONT, 8)
    c_obj.setFillColorRGB(0.3, 0.3, 0.3)
    for i, ln in enumerate([sup.address,
                             f"IČO: {sup.ico}  ·  DIČ: {sup.dic}",
                             f"{sup.phone}  ·  {sup.email}"]):
        c_obj.drawString(15*mm, H - 24*mm - i*4.5*mm, ln)

    # ── rámeček zákazníka ─────────────────────────────────────────────────────
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(1.5)
    c_obj.line(15*mm, H - 40*mm, W - 15*mm, H - 40*mm)
    c_obj.setLineWidth(0.5)

    eic = f"27ZG{cust['ico'][:8]}V"
    c_obj.setFillColorRGB(0.1, 0.1, 0.1)
    for col, items in [
        (15*mm,       [("Zákazník:",      cust["name"]),
                       ("IČO zákazníka:", cust["ico"]),
                       ("Adresa odběru:", f"{cust['street']}, {cust['city']}")]),
        (W/2 + 5*mm,  [("EIC odběrného místa:", eic),
                       ("Smlouva č.:",           f"{cust['ico'][:6]}VO"),
                       ("Faktura č.:",           inv["invoice_number"])]),
    ]:
        for i, (lbl, val) in enumerate(items):
            c_obj.setFont(FONT_BOLD, 8)
            c_obj.drawString(col, H - 47*mm - i*5.5*mm, lbl)
            c_obj.setFont(FONT, 8)
            c_obj.drawString(col + 42*mm, H - 47*mm - i*5.5*mm, val)

    # ── řádek dat ─────────────────────────────────────────────────────────────
    y_strip = H - 68*mm
    c_obj.setFillColorRGB(0.92, 0.96, 0.95)
    c_obj.rect(15*mm, y_strip - 6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
    c_obj.drawString(17*mm, y_strip - 4*mm,
                     f"Fakturační období: {inv['period_start']} – {inv['period_end']}   "
                     f"Datum vystavení: {inv['invoice_date']}   "
                     f"Splatnost: {inv['due_date']}   "
                     f"VS: {inv['variable_symbol']}")

    # ── vícesložková tabulka ──────────────────────────────────────────────────
    y3 = y_strip - 18*mm
    col_x = [15*mm, 78*mm, 100*mm, 123*mm, 150*mm, 174*mm]
    hdrs  = ["Položka", "Jedn.", "Množství", "Sazba Kč", "Základ DPH", "Celkem Kč"]
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(15*mm, y3 - 6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
    for hdr, x in zip(hdrs, col_x):
        c_obj.drawString(x + 1*mm, y3 - 4*mm, hdr)

    commodity_total   = inv.get("commodity_total",    round(inv["consumption"] * inv["unit_price"], 2))
    distrib_fixed_tot = inv.get("distrib_fixed_total", 0.0)
    distrib_var_tot   = inv.get("distrib_var_total",   0.0)
    distrib_fixed_mo  = inv.get("distrib_fixed_mo",    0.0)
    distrib_var_price = inv.get("distrib_var_price",   0.0)

    tbl_rows = [
        ("Komodita – zemní plyn",
         "kWh", str(inv["consumption"]),
         f"{_fmt(inv['unit_price'])} Kč/kWh",
         _fmt(round(commodity_total / 1.21, 2)),
         _fmt(commodity_total)),
        ("Distribuce – pevná měsíční platba",
         "měs.", "6",
         f"{_fmt(distrib_fixed_mo)} Kč/měs",
         _fmt(round(distrib_fixed_tot / 1.21, 2)),
         _fmt(distrib_fixed_tot)),
        ("Distribuce – variabilní složka",
         "kWh", str(inv["consumption"]),
         f"{_fmt(distrib_var_price)} Kč/kWh",
         _fmt(round(distrib_var_tot / 1.21, 2)),
         _fmt(distrib_var_tot)),
    ]
    for ri, row in enumerate(tbl_rows):
        bg = (0.94, 0.98, 0.97) if ri % 2 == 0 else (1, 1, 1)
        c_obj.setFillColorRGB(*bg)
        c_obj.rect(15*mm, y3 - 13*mm - ri*6.5*mm, W - 30*mm, 6.5*mm, fill=1, stroke=0)
        c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
        for val, x in zip(row, col_x):
            c_obj.drawString(x + 1*mm, y3 - 10*mm - ri*6.5*mm, str(val)[:28])

    # ── DPH souhrn ────────────────────────────────────────────────────────────
    y4 = y3 - 42*mm
    bx, bw2, bh2 = W - 85*mm, 70*mm, 48*mm
    c_obj.setFillColorRGB(0.96, 0.98, 0.97)
    c_obj.setStrokeColorRGB(R, G, B)
    c_obj.roundRect(bx, y4 - bh2, bw2, bh2, 2*mm, fill=1, stroke=1)
    c_obj.setFillColorRGB(0.15, 0.15, 0.15); c_obj.setFont(FONT, 8)
    for i, (lbl, val) in enumerate([
        ("Základ daně (DPH 21 %):", _fmt(inv["amount_ex_vat"])  + " Kč"),
        ("DPH 21 %:",               _fmt(inv["vat_amount"])     + " Kč"),
        ("Celkem s DPH:",           _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Zálohy:",                f"–{_fmt(inv['advances'])} Kč"),
    ]):
        c_obj.drawString(bx + 3*mm,          y4 - 8*mm - i*7*mm, lbl)
        c_obj.drawRightString(bx + bw2 - 3*mm, y4 - 8*mm - i*7*mm, val)
    c_obj.setFillColorRGB(R, G, B)
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(0.8)
    c_obj.line(bx + 3*mm, y4 - 39*mm, bx + bw2 - 3*mm, y4 - 39*mm)
    c_obj.setFont(FONT_BOLD, 11)
    lbl = "K ÚHRADĚ:" if inv["balance"] > 0 else "PŘEPLATEK:"
    c_obj.drawString(bx + 3*mm,          y4 - 44*mm, lbl)
    c_obj.drawRightString(bx + bw2 - 3*mm, y4 - 44*mm,
                          _fmt(abs(inv["balance"])) + " Kč")

    # ── IBAN box vlevo ────────────────────────────────────────────────────────
    c_obj.setFillColorRGB(0.94, 0.98, 0.97)
    c_obj.setStrokeColorRGB(R, G, B)
    c_obj.roundRect(15*mm, y4 - bh2, 80*mm, 20*mm, 2*mm, fill=1, stroke=1)
    c_obj.setFillColorRGB(0.15, 0.15, 0.15); c_obj.setFont(FONT, 7.5)
    c_obj.drawString(17*mm, y4 - 12*mm, f"IBAN: {sup.iban}")
    c_obj.drawString(17*mm, y4 - 18*mm, f"Číslo účtu: {sup.account}")

    # ── patička ───────────────────────────────────────────────────────────────
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, 0, W, 9*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT, 6.5)
    c_obj.drawCentredString(W/2, 3*mm,
        f"{sup.name}  ·  IČO {sup.ico}  ·  DIČ {sup.dic}  ·  "
        f"IBAN {sup.iban}  ·  {sup.email}")


def render_thermocity(c_obj: canvas.Canvas, sup: Supplier,
                      cust: dict, inv: dict) -> None:
    """
    Layout C — ThermoCity (teplo):
    Formulářový styl s DPH rekapitulací.
    """
    W, H = A4
    R, G, B = sup.color

    def box(x, y, w, h, title=None):
        c_obj.setStrokeColorRGB(0.6, 0.6, 0.6); c_obj.setLineWidth(0.5)
        c_obj.rect(x, y, w, h, fill=0, stroke=1)
        if title:
            c_obj.setFillColorRGB(R, G, B)
            c_obj.rect(x, y + h - 5*mm, w, 5*mm, fill=1, stroke=0)
            c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
            c_obj.drawString(x + 2*mm, y + h - 3.5*mm, title)

    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 14)
    c_obj.drawString(15*mm, H - 15*mm, "FAKTURA – DAŇOVÝ DOKLAD")
    c_obj.setFont(FONT, 9); c_obj.setFillColorRGB(0.3, 0.3, 0.3)
    c_obj.drawString(15*mm, H - 22*mm, f"Číslo dokladu: {inv['invoice_number']}")
    c_obj.drawRightString(W - 15*mm, H - 15*mm,
                          f"Datum vystavení: {inv['invoice_date']}")
    c_obj.drawRightString(W - 15*mm, H - 22*mm,
                          f"Datum splatnosti: {inv['due_date']}")

    bw = (W - 35*mm) / 2; bh = 38*mm; by = H - 65*mm
    box(15*mm, by, bw, bh, "DODAVATEL")
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT_BOLD, 8)
    c_obj.drawString(17*mm, by + bh - 10*mm, sup.name)
    c_obj.setFont(FONT, 7.5)
    for i, ln in enumerate([sup.address, f"IČO: {sup.ico}",
                             f"DIČ: {sup.dic}", f"IBAN: {sup.iban}", sup.email]):
        c_obj.drawString(17*mm, by + bh - 16*mm - i*4.5*mm, ln)

    box(15*mm + bw + 5*mm, by, bw, bh, "ODBĚRATEL")
    x2 = 17*mm + bw + 5*mm
    c_obj.setFont(FONT_BOLD, 8)
    c_obj.drawString(x2, by + bh - 10*mm, cust["name"])
    c_obj.setFont(FONT, 7.5)
    for i, ln in enumerate([cust["street"], cust["city"],
                             f"IČO: {cust['ico']}", f"DIČ: {cust['dic']}",
                             f"Č. účtu: {cust['account']}"]):
        c_obj.drawString(x2, by + bh - 16*mm - i*4.5*mm, ln)

    y2 = by - 5*mm; bh2 = 22*mm
    box(15*mm, y2 - bh2, W - 30*mm, bh2, "FAKTURAČNÍ ÚDAJE")
    c_obj.setFillColorRGB(0.1, 0.1, 0.1)
    for i, (lbl, val) in enumerate([
        ("Fakturační období:", f"{inv['period_start']} – {inv['period_end']}"),
        ("Variabilní symbol:", inv["variable_symbol"]),
        ("Způsob platby:",     "Bezhotovostní převod"),
        ("Bankovní účet:",     sup.account),
    ]):
        col = 17*mm if i < 2 else W/2 + 5*mm
        ry  = y2 - 10*mm - (i % 2)*5.5*mm
        c_obj.setFont(FONT_BOLD, 7.5); c_obj.drawString(col, ry, lbl)
        c_obj.setFont(FONT, 7.5);      c_obj.drawString(col + 38*mm, ry, val)

    y3 = y2 - bh2 - 8*mm
    cols_x  = [15*mm, 80*mm, 103*mm, 122*mm, 143*mm, 163*mm]
    cols_hdr = ["Popis plnění", "Jedn.", "Množ.",
                "Cena/j.", f"DPH {inv['vat_rate']} %", "Celkem"]
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(15*mm, y3 - 6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
    for hdr, x in zip(cols_hdr, cols_x):
        c_obj.drawString(x + 1*mm, y3 - 4*mm, hdr)
    c_obj.setFillColorRGB(0.96, 0.96, 0.96)
    c_obj.rect(15*mm, y3 - 13*mm, W - 30*mm, 7*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
    desc = f"Dodávka {sup.commodity} – {inv['period_start']} – {inv['period_end']}"
    for val, x in zip([desc[:38], inv["unit"], str(inv["consumption"]),
                       _fmt(inv["unit_price"]) + " Kč",
                       _fmt(inv["vat_amount"]) + " Kč",
                       _fmt(inv["amount_with_vat"]) + " Kč"], cols_x):
        c_obj.drawString(x + 1*mm, y3 - 10*mm, val)

    y4 = y3 - 22*mm; bh3 = 28*mm
    box(15*mm, y4 - bh3, 85*mm, bh3, "DPH REKAPITULACE")
    c_obj.setFillColorRGB(0.1, 0.1, 0.1)
    for i, (lbl, val) in enumerate([
        ("Základ daně:", _fmt(inv["amount_ex_vat"]) + " Kč"),
        (f"DPH {inv['vat_rate']} %:", _fmt(inv["vat_amount"]) + " Kč"),
        ("Celkem:",     _fmt(inv["amount_with_vat"]) + " Kč"),
    ]):
        c_obj.setFont(FONT, 7.5)
        c_obj.drawString(17*mm, y4 - 10*mm - i*5.5*mm, lbl)
        c_obj.setFont(FONT_BOLD, 7.5)
        c_obj.drawRightString(98*mm, y4 - 10*mm - i*5.5*mm, val)

    box(W - 80*mm, y4 - bh3, 65*mm, bh3, "K ÚHRADĚ")
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 16)
    c_obj.drawCentredString(W - 47.5*mm, y4 - 18*mm,
                            _fmt(abs(inv["balance"])) + " Kč")
    c_obj.setFont(FONT, 8); c_obj.setFillColorRGB(0.4, 0.4, 0.4)
    c_obj.drawCentredString(W - 47.5*mm, y4 - 25*mm,
                            "(" + ("nedoplatek" if inv["balance"] > 0 else "přeplatek") + ")")

    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, 0, W, 10*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT, 7)
    c_obj.drawCentredString(W/2, 3.5*mm,
        f"{sup.name}  ·  IČO {sup.ico}  ·  {sup.iban}  ·  {sup.email}")


def render_aquaregion(c_obj: canvas.Canvas, sup: Supplier,
                      cust: dict, inv: dict) -> None:
    """Layout B — AquaRegion (voda): klasický dopisní formát."""
    W, H = A4
    R, G, B = sup.color

    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 14)
    c_obj.drawString(15*mm, H - 18*mm, sup.name)
    c_obj.setFont(FONT, 8); c_obj.setFillColorRGB(0.2, 0.2, 0.2)
    for i, ln in enumerate([sup.address,
                             f"IČO: {sup.ico}  DIČ: {sup.dic}",
                             f"Tel: {sup.phone}", f"E-mail: {sup.email}"]):
        c_obj.drawString(15*mm, H - 24*mm - i*4.5*mm, ln)

    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(1.5)
    c_obj.line(15*mm, H - 52*mm, W - 15*mm, H - 52*mm); c_obj.setLineWidth(0.5)

    c_obj.setFillColorRGB(0.1, 0.1, 0.1)
    c_obj.setFont(FONT_BOLD, 9); c_obj.drawString(15*mm, H - 60*mm, "Odběratel:")
    c_obj.setFont(FONT, 9)
    for i, ln in enumerate([cust["name"], cust["street"], cust["city"],
                             f"IČO: {cust['ico']}  DIČ: {cust['dic']}"]):
        c_obj.drawString(15*mm, H - 66*mm - i*5*mm, ln)

    y = H - 92*mm
    for label, val in [
        ("Číslo dokladu:", inv["invoice_number"]),
        ("Datum vystavení:", inv["invoice_date"]),
        ("Datum splatnosti:", inv["due_date"]),
        ("Fakturační období:", f"{inv['period_start']} – {inv['period_end']}"),
        ("Variabilní symbol:", inv["variable_symbol"]),
        ("Bankovní účet:", sup.account),
        ("IBAN:", sup.iban),
    ]:
        c_obj.setFont(FONT_BOLD, 8.5); c_obj.drawString(15*mm, y, label)
        c_obj.setFont(FONT, 8.5);      c_obj.drawString(80*mm, y, val)
        y -= 5.5*mm

    y -= 6*mm
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(15*mm, y - 6*mm, W - 30*mm, 7*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 8)
    for txt, xpos in [("Položka", 16*mm), ("Spotřeba", 95*mm),
                       ("Sazba bez DPH", 120*mm), ("DPH", 150*mm),
                       ("Celkem s DPH", 168*mm)]:
        c_obj.drawString(xpos, y - 3.5*mm, txt)

    c_obj.setFillColorRGB(0.93, 0.97, 0.95)
    c_obj.rect(15*mm, y - 14*mm, W - 30*mm, 8*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 8)
    desc = f"Dodávka {sup.commodity} — {inv['period_start']} – {inv['period_end']}"
    for txt, xpos in [
        (desc[:45], 16*mm),
        (f"{inv['consumption']} {inv['unit']}", 95*mm),
        (_fmt(inv["unit_price"]) + " Kč/" + inv["unit"], 120*mm),
        (f"{inv['vat_rate']} %", 155*mm),
        (_fmt(inv["amount_with_vat"]) + " Kč", 168*mm),
    ]:
        c_obj.drawString(xpos, y - 11*mm, txt)

    y2 = y - 28*mm
    for lbl, val in [
        ("Fakturovaná částka bez DPH:", _fmt(inv["amount_ex_vat"]) + " Kč"),
        (f"DPH {inv['vat_rate']} %:",   _fmt(inv["vat_amount"])    + " Kč"),
        ("Celkem včetně DPH:",           _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Zálohy:",                     f"–{_fmt(inv['advances'])} Kč"),
    ]:
        c_obj.setFont(FONT, 8.5);      c_obj.drawRightString(W - 60*mm, y2, lbl)
        c_obj.setFont(FONT_BOLD, 8.5); c_obj.drawRightString(W - 15*mm, y2, val)
        y2 -= 5.5*mm

    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(1.0)
    c_obj.line(W - 80*mm, y2 + 2*mm, W - 15*mm, y2 + 2*mm)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 10)
    lbl = "K ÚHRADĚ:" if inv["balance"] > 0 else "PŘEPLATEK:"
    c_obj.drawRightString(W - 60*mm, y2 - 3*mm, lbl)
    c_obj.drawRightString(W - 15*mm, y2 - 3*mm,
                          _fmt(abs(inv["balance"])) + " Kč")

    c_obj.setFillColorRGB(0.5, 0.5, 0.5); c_obj.setFont(FONT, 7)
    c_obj.drawCentredString(W/2, 12*mm,
        f"Společnost zapsána v obchodním rejstříku. IČO: {sup.ico}  "
        f"DIČ: {sup.dic}  IBAN: {sup.iban}")


def render_energyplus(c_obj: canvas.Canvas, sup: Supplier,
                      cust: dict, inv: dict) -> None:
    """Layout G — EnergyPlus (NN): thin bar + EP circle, letter-style, borderless table."""
    W, H = A4; R, G, B = sup.color

    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, H - 13*mm, W, 13*mm, fill=1, stroke=0)
    # Circle logo
    c_obj.setFillColorRGB(1, 1, 1)
    c_obj.circle(W - 20*mm, H - 6.5*mm, 5*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 8)
    c_obj.drawCentredString(W - 20*mm, H - 9*mm, "EP")
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 11)
    c_obj.drawString(15*mm, H - 9*mm, sup.name)

    c_obj.setFillColorRGB(0.3, 0.3, 0.3); c_obj.setFont(FONT, 7.5)
    c_obj.drawString(15*mm, H - 18*mm,
                     f"{sup.address}  ·  IČO: {sup.ico}  ·  {sup.phone}  ·  {sup.email}")
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(0.8)
    c_obj.line(15*mm, H - 21*mm, W - 15*mm, H - 21*mm)

    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    c_obj.drawString(15*mm, H - 30*mm, "FAKTURA – DAŇOVÝ DOKLAD")
    c_obj.setFillColorRGB(0.3, 0.3, 0.3); c_obj.setFont(FONT, 8.5)
    c_obj.drawString(15*mm, H - 37*mm,
                     f"č. {inv['invoice_number']}   Datum: {inv['invoice_date']}   "
                     f"Splatnost: {inv['due_date']}")

    # Customer block (left) + invoice details (right, tinted)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT_BOLD, 8)
    c_obj.drawString(15*mm, H - 48*mm, "Odběratel:")
    c_obj.setFont(FONT, 8)
    for i, ln in enumerate([cust["name"], cust["street"], cust["city"],
                             f"IČO: {cust['ico']}   DIČ: {cust['dic']}"]):
        c_obj.drawString(15*mm, H - 54*mm - i*5*mm, ln)

    c_obj.setFillColorRGB(0.92, 0.97, 0.95)
    c_obj.rect(W/2, H - 73*mm, W/2 - 15*mm, 30*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1)
    for i, (lbl, val) in enumerate([
        ("Fakturační období:", f"{inv['period_start']} – {inv['period_end']}"),
        ("Variabilní symbol:", inv["variable_symbol"]),
        ("IBAN:", sup.iban), ("Číslo účtu:", sup.account),
    ]):
        c_obj.setFont(FONT_BOLD, 7.5); c_obj.drawString(W/2 + 3*mm, H - 77*mm - i*5.5*mm, lbl)
        c_obj.setFont(FONT, 7.5);      c_obj.drawString(W/2 + 38*mm, H - 77*mm - i*5.5*mm, val)

    # Borderless table with divider lines
    y3 = H - 82*mm
    hdrs = ["Popis dodávky", "Spotřeba", "Jedn.", "Cena/j.", "DPH %", "Celkem Kč"]
    xs   = [15*mm, 82*mm, 100*mm, 118*mm, 148*mm, 163*mm]
    c_obj.setStrokeColorRGB(0.65, 0.65, 0.65); c_obj.setLineWidth(0.3)
    c_obj.line(15*mm, y3 + 1*mm, W - 15*mm, y3 + 1*mm)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 7.5)
    for h2, x in zip(hdrs, xs): c_obj.drawString(x, y3 - 4*mm, h2)
    c_obj.line(15*mm, y3 - 6*mm, W - 15*mm, y3 - 6*mm)
    desc = f"Dodávka elektřiny NN – {inv['period_start']} – {inv['period_end']}"
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
    for val, x in zip([desc, str(inv["consumption"]), "kWh",
                       _fmt(inv["unit_price"]) + " Kč", f"{inv['vat_rate']} %",
                       _fmt(inv["amount_with_vat"])], xs):
        c_obj.drawString(x, y3 - 12*mm, str(val)[:28])
    c_obj.line(15*mm, y3 - 14*mm, W - 15*mm, y3 - 14*mm)

    # Underline totals (no box)
    y4 = y3 - 22*mm
    for i, (lbl, val) in enumerate([
        ("Základ daně:", _fmt(inv["amount_ex_vat"]) + " Kč"),
        (f"DPH {inv['vat_rate']} %:", _fmt(inv["vat_amount"]) + " Kč"),
        ("Celkem s DPH:", _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Zálohy:", f"–{_fmt(inv['advances'])} Kč"),
    ]):
        c_obj.setFont(FONT, 8); c_obj.drawRightString(W - 60*mm, y4 - i*5.5*mm, lbl)
        c_obj.setFont(FONT_BOLD, 8); c_obj.drawRightString(W - 15*mm, y4 - i*5.5*mm, val)
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(1.2)
    c_obj.line(W - 80*mm, y4 - 25*mm, W - 15*mm, y4 - 25*mm)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    lbl = "K ÚHRADĚ:" if inv["balance"] > 0 else "PŘEPLATEK:"
    c_obj.drawRightString(W - 60*mm, y4 - 31*mm, lbl)
    c_obj.drawRightString(W - 15*mm, y4 - 31*mm, _fmt(abs(inv["balance"])) + " Kč")

    c_obj.setFillColorRGB(0.55, 0.55, 0.55); c_obj.setFont(FONT, 6.5)
    c_obj.drawString(15*mm, 11*mm, f"{sup.name}  ·  IČO: {sup.ico}  ·  DIČ: {sup.dic}")
    c_obj.drawString(15*mm, 7*mm, f"IBAN: {sup.iban}  ·  {sup.email}  ·  {sup.phone}")


def render_sparkelit(c_obj: canvas.Canvas, sup: Supplier,
                     cust: dict, inv: dict) -> None:
    """Layout H — SparkElit (NN): no header bar, large name, bordered FAKTURA box,
    gray-header table, right-aligned underline totals."""
    W, H = A4; R, G, B = sup.color

    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 20)
    c_obj.drawString(15*mm, H - 16*mm, sup.name)
    c_obj.setFillColorRGB(0.35, 0.35, 0.35); c_obj.setFont(FONT, 7.5)
    c_obj.drawString(15*mm, H - 22*mm,
                     f"{sup.address}   IČO: {sup.ico}   DIČ: {sup.dic}")
    c_obj.drawString(15*mm, H - 27*mm,
                     f"{sup.phone}   {sup.email}")
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(15*mm, H - 29*mm, W - 30*mm, 1*mm, fill=1, stroke=0)

    # Bordered "FAKTURA" box
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(1.0)
    fx, fy, fw, fh = W/2 - 52*mm, H - 45*mm, 104*mm, 13*mm
    c_obj.rect(fx, fy, fw, fh, fill=0, stroke=1)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    c_obj.drawCentredString(W/2, fy + 4*mm,
                            f"FAKTURA – DAŇOVÝ DOKLAD č. {inv['invoice_number']}")

    # Two-col info
    y = H - 63*mm; bw = (W - 35*mm) / 2
    for col, lbl, lines in [
        (15*mm, "DODAVATEL:", [sup.name, sup.address,
                               f"IČO: {sup.ico}", f"IBAN: {sup.iban}"]),
        (15*mm + bw + 5*mm, "ODBĚRATEL:", [cust["name"], cust["street"],
                                            cust["city"], f"IČO: {cust['ico']}"]),
    ]:
        c_obj.setFont(FONT_BOLD, 7.5); c_obj.setFillColorRGB(R, G, B)
        c_obj.drawString(col, y, lbl)
        c_obj.setFont(FONT, 7.5); c_obj.setFillColorRGB(0.15, 0.15, 0.15)
        for i, ln in enumerate(lines):
            c_obj.drawString(col, y - 5*mm - i*4.5*mm, ln)

    # Invoice meta inline
    y2 = y - 30*mm
    for lbl, val in [
        ("Datum vystavení:", inv["invoice_date"]),
        ("Datum splatnosti:", inv["due_date"]),
        ("Fakturační období:", f"{inv['period_start']} – {inv['period_end']}"),
        ("Variabilní symbol:", inv["variable_symbol"]),
    ]:
        c_obj.setFont(FONT_BOLD, 8); c_obj.drawString(15*mm, y2, lbl)
        c_obj.setFont(FONT, 8);      c_obj.drawString(65*mm, y2, val)
        y2 -= 5.5*mm

    # Gray-header table
    y3 = y2 - 6*mm
    hdrs = ["Popis", "Množství", "Jedn.", "Bez DPH", "DPH", "S DPH"]
    xs   = [15*mm, 82*mm, 100*mm, 118*mm, 148*mm, 163*mm]
    c_obj.setFillColorRGB(0.82, 0.82, 0.82)
    c_obj.rect(15*mm, y3 - 6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.12, 0.12, 0.12); c_obj.setFont(FONT_BOLD, 7.5)
    for h2, x in zip(hdrs, xs): c_obj.drawString(x + 1*mm, y3 - 4*mm, h2)
    row = [f"El. NN – {inv['period_start']} – {inv['period_end']}",
           str(inv["consumption"]), "kWh",
           _fmt(inv["amount_ex_vat"]) + " Kč",
           f"{inv['vat_rate']} %", _fmt(inv["amount_with_vat"]) + " Kč"]
    c_obj.setFillColorRGB(0.95, 0.95, 0.95)
    c_obj.rect(15*mm, y3 - 13*mm, W - 30*mm, 7*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
    for val, x in zip(row, xs): c_obj.drawString(x + 1*mm, y3 - 10*mm, str(val)[:28])

    y4 = y3 - 22*mm
    for i, (lbl, val) in enumerate([
        ("Základ daně:", _fmt(inv["amount_ex_vat"]) + " Kč"),
        (f"DPH {inv['vat_rate']} %:", _fmt(inv["vat_amount"]) + " Kč"),
        ("Celkem s DPH:", _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Zálohy:", f"–{_fmt(inv['advances'])} Kč"),
    ]):
        c_obj.setFont(FONT, 8); c_obj.drawRightString(W - 60*mm, y4 - i*5.5*mm, lbl)
        c_obj.setFont(FONT_BOLD, 8); c_obj.drawRightString(W - 15*mm, y4 - i*5.5*mm, val)
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(1.5)
    c_obj.line(W - 75*mm, y4 - 25*mm, W - 15*mm, y4 - 25*mm)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 12)
    lbl = "CELKEM K ÚHRADĚ:" if inv["balance"] > 0 else "PŘEPLATEK:"
    c_obj.drawRightString(W - 75*mm, y4 - 31*mm, lbl)
    c_obj.drawRightString(W - 15*mm, y4 - 31*mm, _fmt(abs(inv["balance"])) + " Kč")

    c_obj.setFillColorRGB(0.6, 0.6, 0.6); c_obj.setFont(FONT, 7)
    c_obj.drawCentredString(W/2, 8*mm,
                            f"{sup.name}  ·  {sup.email}  ·  {sup.phone}")


def render_highvolt(c_obj: canvas.Canvas, sup: Supplier,
                    cust: dict, inv: dict) -> None:
    """Layout I — HighVolt (VN): double bar header, EAN strip, 2-tariff table."""
    W, H = A4; R, G, B = sup.color

    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, H - 20*mm, W, 20*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1)
    c_obj.setFont(FONT_BOLD, 14); c_obj.drawString(15*mm, H - 10*mm, sup.name)
    c_obj.setFont(FONT, 8); c_obj.drawString(15*mm, H - 17*mm, sup.address)
    c_obj.drawRightString(W - 15*mm, H - 10*mm,
                          f"IČO: {sup.ico}   DIČ: {sup.dic}")
    c_obj.drawRightString(W - 15*mm, H - 17*mm,
                          f"{sup.phone}   {sup.email}")
    # Thin secondary bar
    c_obj.setFillColorRGB(*[x*0.65 for x in (R, G, B)])
    c_obj.rect(0, H - 24*mm, W, 4*mm, fill=1, stroke=0)

    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 12)
    c_obj.drawString(15*mm, H - 34*mm,
                     f"VN FAKTURA – DAŇOVÝ DOKLAD č. {inv['invoice_number']}")
    c_obj.setFillColorRGB(0.3, 0.3, 0.3); c_obj.setFont(FONT, 8.5)
    c_obj.drawString(15*mm, H - 41*mm,
                     f"Datum: {inv['invoice_date']}   Splatnost: {inv['due_date']}   "
                     f"VS: {inv['variable_symbol']}")

    # Two-column boxes
    bw = (W - 35*mm) / 2; bh = 32*mm; by = H - 78*mm
    for bx, title, lines in [
        (15*mm, "DODAVATEL",
         [sup.name, sup.address, f"IČO: {sup.ico}", f"IBAN: {sup.iban}"]),
        (15*mm + bw + 5*mm, "ODBĚRATEL",
         [cust["name"], cust["street"], cust["city"], f"IČO: {cust['ico']}"]),
    ]:
        c_obj.setStrokeColorRGB(0.65, 0.65, 0.65); c_obj.setLineWidth(0.4)
        c_obj.rect(bx, by, bw, bh, fill=0, stroke=1)
        c_obj.setFillColorRGB(R, G, B)
        c_obj.rect(bx, by + bh - 5*mm, bw, 5*mm, fill=1, stroke=0)
        c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
        c_obj.drawString(bx + 2*mm, by + bh - 3.5*mm, title)
        c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
        for i, ln in enumerate(lines):
            c_obj.drawString(bx + 2*mm, by + bh - 10*mm - i*4.5*mm, ln)

    # EAN strip
    ean = f"859182400001{cust['ico'][:8]}"
    y_ean = by - 6*mm
    c_obj.setFillColorRGB(0.93, 0.95, 0.98)
    c_obj.rect(15*mm, y_ean - 5*mm, W - 30*mm, 5*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.2, 0.2, 0.2); c_obj.setFont(FONT, 7.5)
    c_obj.drawString(17*mm, y_ean - 3.5*mm,
                     f"EAN: {ean}   Napěťová úroveň: VN   "
                     f"Smlouva: {cust['ico'][:6]}VN   "
                     f"Období: {inv['period_start']} – {inv['period_end']}")

    # Simplified 2-tariff VN table (VT + NT)
    y3 = y_ean - 16*mm
    hdrs = ["Položka", "Množství kWh", "Cena Kč/kWh", "Celkem Kč"]
    xs   = [15*mm, 80*mm, 120*mm, 160*mm]
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(15*mm, y3 - 6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
    for h2, x in zip(hdrs, xs): c_obj.drawString(x + 1*mm, y3 - 4*mm, h2)

    kWh_vt = inv.get("kWh_vt", round(inv["consumption"] * 0.62))
    kWh_nt = inv["consumption"] - kWh_vt
    p_vt   = inv.get("price_vt", inv["unit_price"] * 1.25)
    p_nt   = inv.get("price_nt", inv["unit_price"] * 0.60)
    rows   = [
        ("Činná energie VT", kWh_vt, _fmt(p_vt), _fmt(round(kWh_vt * p_vt, 2))),
        ("Činná energie NT", kWh_nt, _fmt(p_nt), _fmt(round(kWh_nt * p_nt, 2))),
    ]
    for ri, row in enumerate(rows):
        bg = (0.95, 0.96, 0.98) if ri == 0 else (1, 1, 1)
        c_obj.setFillColorRGB(*bg)
        c_obj.rect(15*mm, y3 - 13*mm - ri*6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
        c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
        for val, x in zip(row, xs): c_obj.drawString(x + 1*mm, y3 - 10*mm - ri*6*mm, str(val))

    # Right summary box
    y4 = y3 - 28*mm
    bx2, bw2, bh2 = W - 82*mm, 67*mm, 42*mm
    c_obj.setFillColorRGB(0.96, 0.96, 0.98)
    c_obj.setStrokeColorRGB(R, G, B)
    c_obj.roundRect(bx2, y4 - bh2, bw2, bh2, 2*mm, fill=1, stroke=1)
    c_obj.setFillColorRGB(0.2, 0.2, 0.2); c_obj.setFont(FONT, 8)
    for i, (lbl, val) in enumerate([
        ("Základ daně:", _fmt(inv["amount_ex_vat"]) + " Kč"),
        (f"DPH {inv['vat_rate']} %:", _fmt(inv["vat_amount"]) + " Kč"),
        ("Celkem s DPH:", _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Zálohy:", f"–{_fmt(inv['advances'])} Kč"),
    ]):
        c_obj.drawString(bx2 + 3*mm, y4 - 7*mm - i*7*mm, lbl)
        c_obj.drawRightString(bx2 + bw2 - 3*mm, y4 - 7*mm - i*7*mm, val)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    lbl = "K ÚHRADĚ:" if inv["balance"] > 0 else "PŘEPLATEK:"
    c_obj.drawString(bx2 + 3*mm, y4 - 37*mm, lbl)
    c_obj.drawRightString(bx2 + bw2 - 3*mm, y4 - 37*mm,
                          _fmt(abs(inv["balance"])) + " Kč")

    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, 0, W, 9*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT, 6.5)
    c_obj.drawCentredString(W/2, 3*mm,
        f"{sup.name}  ·  IČO {sup.ico}  ·  IBAN {sup.iban}  ·  {sup.email}")


def render_elnord(c_obj: canvas.Canvas, sup: Supplier,
                  cust: dict, inv: dict) -> None:
    """Layout J — ElNord (VN): left colored sidebar strip, right main content."""
    W, H = A4; R, G, B = sup.color
    SW = 22*mm  # sidebar width

    # Sidebar
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, 0, SW, H, fill=1, stroke=0)
    # Rotated invoice number in sidebar
    c_obj.saveState()
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 8)
    c_obj.translate(SW/2, H/2)
    c_obj.rotate(90)
    c_obj.drawCentredString(0, 0, f"FAKTURA  {inv['invoice_number']}")
    c_obj.restoreState()
    # VS in sidebar
    c_obj.saveState()
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT, 6.5)
    c_obj.translate(SW/2, H/4)
    c_obj.rotate(90)
    c_obj.drawCentredString(0, 0, f"VS: {inv['variable_symbol']}")
    c_obj.restoreState()

    # Main content starts at SW + 5mm
    MX = SW + 5*mm  # main x start

    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 13)
    c_obj.drawString(MX, H - 13*mm, sup.name)
    c_obj.setFillColorRGB(0.3, 0.3, 0.3); c_obj.setFont(FONT, 7.5)
    c_obj.drawString(MX, H - 19*mm,
                     f"{sup.address}  ·  IČO: {sup.ico}  ·  {sup.email}")
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(1.0)
    c_obj.line(MX, H - 22*mm, W - 10*mm, H - 22*mm)

    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    c_obj.drawString(MX, H - 31*mm, "VN FAKTURA – DAŇOVÝ DOKLAD")
    c_obj.setFillColorRGB(0.3, 0.3, 0.3); c_obj.setFont(FONT, 8)
    c_obj.drawString(MX, H - 38*mm,
                     f"Vystaveno: {inv['invoice_date']}   Splatnost: {inv['due_date']}")

    # Customer
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT_BOLD, 8)
    c_obj.drawString(MX, H - 48*mm, "Odběratel:")
    c_obj.setFont(FONT, 8)
    for i, ln in enumerate([cust["name"], cust["street"], cust["city"],
                             f"IČO: {cust['ico']}"]):
        c_obj.drawString(MX, H - 54*mm - i*5*mm, ln)

    ean = f"859182400002{cust['ico'][:8]}"
    c_obj.setFont(FONT, 7.5); c_obj.setFillColorRGB(0.3, 0.3, 0.3)
    c_obj.drawString(MX, H - 76*mm,
                     f"EAN: {ean}   Napěťová úroveň: VN   "
                     f"Období: {inv['period_start']} – {inv['period_end']}")

    # Compact VN table (total consumption, no tariff split)
    y3 = H - 84*mm
    hdrs = ["Popis plnění", "Spotřeba (kWh)", "Cena/kWh", "Základ DPH", "Celkem Kč"]
    xs   = [MX, MX + 60*mm, MX + 88*mm, MX + 116*mm, MX + 142*mm]
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(MX, y3 - 6*mm, W - MX - 10*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
    for h2, x in zip(hdrs, xs): c_obj.drawString(x + 1*mm, y3 - 4*mm, h2)
    row = [f"Elektřina VN – {inv['period_start']}–{inv['period_end']}",
           str(inv["consumption"]), _fmt(inv["unit_price"]) + " Kč",
           _fmt(inv["amount_ex_vat"]) + " Kč", _fmt(inv["amount_with_vat"]) + " Kč"]
    c_obj.setFillColorRGB(0.94, 0.94, 0.96)
    c_obj.rect(MX, y3 - 13*mm, W - MX - 10*mm, 7*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
    for val, x in zip(row, xs): c_obj.drawString(x + 1*mm, y3 - 10*mm, str(val)[:28])

    y4 = y3 - 26*mm
    for i, (lbl, val) in enumerate([
        ("Základ daně (DPH 21 %):", _fmt(inv["amount_ex_vat"]) + " Kč"),
        ("DPH 21 %:",               _fmt(inv["vat_amount"]) + " Kč"),
        ("Celkem s DPH:",           _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Zálohy:",                f"–{_fmt(inv['advances'])} Kč"),
    ]):
        c_obj.setFont(FONT, 8); c_obj.drawRightString(W - 55*mm, y4 - i*5.5*mm, lbl)
        c_obj.setFont(FONT_BOLD, 8); c_obj.drawRightString(W - 10*mm, y4 - i*5.5*mm, val)
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(1.2)
    c_obj.line(W - 75*mm, y4 - 25*mm, W - 10*mm, y4 - 25*mm)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    lbl = "K ÚHRADĚ:" if inv["balance"] > 0 else "PŘEPLATEK:"
    c_obj.drawRightString(W - 55*mm, y4 - 31*mm, lbl)
    c_obj.drawRightString(W - 10*mm, y4 - 31*mm, _fmt(abs(inv["balance"])) + " Kč")


def render_gaspraha(c_obj: canvas.Canvas, sup: Supplier,
                    cust: dict, inv: dict) -> None:
    """Layout K — GasPraha (MO): classic letter, company top-right, FAKTURA centered."""
    W, H = A4; R, G, B = sup.color

    # Company info top-right
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 12)
    c_obj.drawRightString(W - 15*mm, H - 14*mm, sup.name)
    c_obj.setFillColorRGB(0.3, 0.3, 0.3); c_obj.setFont(FONT, 8)
    for i, ln in enumerate([sup.address, f"IČO: {sup.ico}   DIČ: {sup.dic}",
                             sup.phone, sup.email]):
        c_obj.drawRightString(W - 15*mm, H - 20*mm - i*4.5*mm, ln)
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(0.8)
    c_obj.line(15*mm, H - 40*mm, W - 15*mm, H - 40*mm)

    # Customer address block left
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT_BOLD, 8.5)
    c_obj.drawString(15*mm, H - 17*mm, "Příjemce:")
    c_obj.setFont(FONT, 8.5)
    for i, ln in enumerate([cust["name"], cust["street"], cust["city"],
                             f"IČO: {cust['ico']}"]):
        c_obj.drawString(15*mm, H - 23*mm - i*5.5*mm, ln)

    # "FAKTURA" centered
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 13)
    c_obj.drawCentredString(W/2, H - 50*mm,
                            f"FAKTURA č. {inv['invoice_number']}")
    c_obj.setFillColorRGB(0.3, 0.3, 0.3); c_obj.setFont(FONT, 8.5)
    c_obj.drawCentredString(W/2, H - 57*mm,
                            f"Datum vystavení: {inv['invoice_date']}   "
                            f"Splatnost: {inv['due_date']}   "
                            f"VS: {inv['variable_symbol']}")
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(0.5)
    c_obj.line(15*mm, H - 60*mm, W - 15*mm, H - 60*mm)

    # Fakturační období + EIC
    c_obj.setFillColorRGB(0.15, 0.15, 0.15); c_obj.setFont(FONT, 8)
    c_obj.drawString(15*mm, H - 68*mm,
                     f"Fakturační období: {inv['period_start']} – {inv['period_end']}   "
                     f"EIC: 27ZG{cust['ico'][:8]}P   "
                     f"IBAN: {sup.iban}")

    # Simple 3-col gas table
    y3 = H - 78*mm
    hdrs = ["Popis", "Spotřeba", "Cena/kWh", f"DPH {inv['vat_rate']} %", "Celkem s DPH"]
    xs   = [15*mm, 85*mm, 112*mm, 140*mm, 163*mm]
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(15*mm, y3 - 6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
    for h2, x in zip(hdrs, xs): c_obj.drawString(x + 1*mm, y3 - 4*mm, h2)
    row = [f"Zemní plyn MO – {inv['period_start']} – {inv['period_end']}",
           f"{inv['consumption']} kWh", _fmt(inv["unit_price"]) + " Kč",
           f"{inv['vat_rate']} %", _fmt(inv["amount_with_vat"]) + " Kč"]
    c_obj.setFillColorRGB(0.97, 0.95, 0.91)
    c_obj.rect(15*mm, y3 - 13*mm, W - 30*mm, 7*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
    for val, x in zip(row, xs): c_obj.drawString(x + 1*mm, y3 - 10*mm, str(val)[:32])

    # Colored "K ÚHRADĚ" box bottom-right
    bx, bw2, bh2 = W - 85*mm, 70*mm, 42*mm
    y4 = y3 - 22*mm
    c_obj.setFillColorRGB(0.97, 0.95, 0.91)
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(0.8)
    c_obj.roundRect(bx, y4 - bh2, bw2, bh2, 2*mm, fill=1, stroke=1)
    c_obj.setFillColorRGB(0.2, 0.2, 0.2); c_obj.setFont(FONT, 8)
    for i, (lbl, val) in enumerate([
        ("Základ daně:", _fmt(inv["amount_ex_vat"]) + " Kč"),
        (f"DPH {inv['vat_rate']} %:", _fmt(inv["vat_amount"]) + " Kč"),
        ("Celkem s DPH:", _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Zálohy:", f"–{_fmt(inv['advances'])} Kč"),
    ]):
        c_obj.drawString(bx + 3*mm, y4 - 7*mm - i*7*mm, lbl)
        c_obj.drawRightString(bx + bw2 - 3*mm, y4 - 7*mm - i*7*mm, val)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 12)
    lbl = "K ÚHRADĚ:" if inv["balance"] > 0 else "PŘEPLATEK:"
    c_obj.drawString(bx + 3*mm, y4 - 37*mm, lbl)
    c_obj.drawRightString(bx + bw2 - 3*mm, y4 - 37*mm,
                          _fmt(abs(inv["balance"])) + " Kč")

    c_obj.setFillColorRGB(0.6, 0.6, 0.6); c_obj.setFont(FONT, 7)
    c_obj.drawString(15*mm, 10*mm, f"{sup.name}  ·  IČO: {sup.ico}  ·  IBAN: {sup.iban}")


def render_termoplyn(c_obj: canvas.Canvas, sup: Supplier,
                     cust: dict, inv: dict) -> None:
    """Layout L — TermoPlyn (MO): centered company, two bordered boxes, centered summary."""
    W, H = A4; R, G, B = sup.color

    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(1.0)
    c_obj.line(15*mm, H - 10*mm, W - 15*mm, H - 10*mm)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 14)
    c_obj.drawCentredString(W/2, H - 19*mm, sup.name)
    c_obj.setFont(FONT, 8); c_obj.setFillColorRGB(0.35, 0.35, 0.35)
    c_obj.drawCentredString(W/2, H - 25*mm,
                            f"{sup.address}   IČO: {sup.ico}   {sup.email}")
    c_obj.setStrokeColorRGB(R, G, B)
    c_obj.line(15*mm, H - 28*mm, W - 15*mm, H - 28*mm)

    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 12)
    c_obj.drawCentredString(W/2, H - 37*mm, "VYÚČTOVÁNÍ ZA ZEMNÍ PLYN")
    c_obj.setFillColorRGB(0.3, 0.3, 0.3); c_obj.setFont(FONT, 8.5)
    c_obj.drawCentredString(W/2, H - 44*mm,
                            f"Faktura č. {inv['invoice_number']}   "
                            f"Datum: {inv['invoice_date']}   Splatnost: {inv['due_date']}")

    # Two bordered boxes
    bw = (W - 35*mm) / 2; bh = 36*mm; by = H - 85*mm
    for bx, title, lines in [
        (15*mm, "DODAVATEL",
         [sup.name, sup.address, f"IČO: {sup.ico}", f"DIČ: {sup.dic}", f"IBAN: {sup.iban}"]),
        (15*mm + bw + 5*mm, "ODBĚRATEL",
         [cust["name"], cust["street"], cust["city"],
          f"IČO: {cust['ico']}", f"EIC: 27ZG{cust['ico'][:8]}P"]),
    ]:
        c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(0.6)
        c_obj.rect(bx, by, bw, bh, fill=0, stroke=1)
        c_obj.setFillColorRGB(R, G, B)
        c_obj.rect(bx, by + bh - 5.5*mm, bw, 5.5*mm, fill=1, stroke=0)
        c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 8)
        c_obj.drawString(bx + 2*mm, by + bh - 3.8*mm, title)
        c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
        for i, ln in enumerate(lines):
            c_obj.drawString(bx + 2*mm, by + bh - 11*mm - i*4.5*mm, ln)

    y_meta = by - 7*mm
    c_obj.setFillColorRGB(0.92, 0.97, 0.93)
    c_obj.rect(15*mm, y_meta - 5.5*mm, W - 30*mm, 5.5*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
    m3_approx = round(inv["consumption"] / 10.55, 1)
    c_obj.drawString(17*mm, y_meta - 3.8*mm,
                     f"Fakturační období: {inv['period_start']} – {inv['period_end']}   "
                     f"VS: {inv['variable_symbol']}   "
                     f"Spotřeba: {inv['consumption']} kWh ≈ {m3_approx} m³")

    # Table
    y3 = y_meta - 16*mm
    hdrs = ["Položka", "kWh", "Kč/kWh", f"DPH {inv['vat_rate']} %", "Základ", "Celkem"]
    xs   = [15*mm, 68*mm, 92*mm, 116*mm, 140*mm, 163*mm]
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(15*mm, y3 - 6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
    for h2, x in zip(hdrs, xs): c_obj.drawString(x + 1*mm, y3 - 4*mm, h2)
    row = [f"Plyn MO – {inv['period_start']}–{inv['period_end']}",
           str(inv["consumption"]), _fmt(inv["unit_price"]),
           f"{inv['vat_rate']} %", _fmt(inv["amount_ex_vat"]),
           _fmt(inv["amount_with_vat"])]
    c_obj.setFillColorRGB(0.94, 0.98, 0.95)
    c_obj.rect(15*mm, y3 - 13*mm, W - 30*mm, 7*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
    for val, x in zip(row, xs): c_obj.drawString(x + 1*mm, y3 - 10*mm, str(val)[:25])

    # Centered summary (different from right-aligned boxes)
    y4 = y3 - 24*mm
    bx2 = W/2 - 45*mm; bw3 = 90*mm; bh3 = 42*mm
    c_obj.setFillColorRGB(0.95, 0.98, 0.96)
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(0.8)
    c_obj.roundRect(bx2, y4 - bh3, bw3, bh3, 2*mm, fill=1, stroke=1)
    c_obj.setFillColorRGB(0.15, 0.15, 0.15); c_obj.setFont(FONT, 8)
    for i, (lbl, val) in enumerate([
        ("Základ daně:", _fmt(inv["amount_ex_vat"]) + " Kč"),
        (f"DPH {inv['vat_rate']} %:", _fmt(inv["vat_amount"]) + " Kč"),
        ("Celkem s DPH:", _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Zálohy:", f"–{_fmt(inv['advances'])} Kč"),
    ]):
        c_obj.drawString(bx2 + 3*mm, y4 - 7*mm - i*7*mm, lbl)
        c_obj.drawRightString(bx2 + bw3 - 3*mm, y4 - 7*mm - i*7*mm, val)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    lbl = "K ÚHRADĚ:" if inv["balance"] > 0 else "PŘEPLATEK:"
    c_obj.drawCentredString(W/2, y4 - 37*mm,
                            f"{lbl}  {_fmt(abs(inv['balance']))} Kč")

    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, 0, W, 9*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT, 6.5)
    c_obj.drawCentredString(W/2, 3*mm,
        f"{sup.name}  ·  IČO {sup.ico}  ·  IBAN {sup.iban}  ·  {sup.email}")


def render_progas_vo(c_obj: canvas.Canvas, sup: Supplier,
                     cust: dict, inv: dict) -> None:
    """Layout M — ProGas VO (VO): navy header + VO badge, 2×2 info grid, VO table."""
    W, H = A4; R, G, B = sup.color

    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, H - 24*mm, W, 24*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 14)
    c_obj.drawString(15*mm, H - 11*mm, sup.name)
    c_obj.setFont(FONT, 8)
    c_obj.drawString(15*mm, H - 18*mm, sup.address)
    c_obj.drawRightString(W - 15*mm, H - 11*mm,
                          f"IČO: {sup.ico}  ·  DIČ: {sup.dic}")
    c_obj.drawRightString(W - 15*mm, H - 18*mm,
                          f"{sup.phone}  ·  {sup.email}")
    # VO badge
    c_obj.setFillColorRGB(1, 1, 1)
    c_obj.roundRect(W - 40*mm, H - 6*mm, 25*mm, 5*mm, 1*mm, fill=0, stroke=1)
    c_obj.setFont(FONT_BOLD, 7.5)
    c_obj.drawCentredString(W - 27.5*mm, H - 3.8*mm, "PRŮMYSLOVÝ ZÁKAZNÍK VO")

    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    c_obj.drawString(15*mm, H - 33*mm,
                     f"DAŇOVÝ DOKLAD – PLYN VO č. {inv['invoice_number']}")
    c_obj.setFillColorRGB(0.3, 0.3, 0.3); c_obj.setFont(FONT, 8)
    c_obj.drawString(15*mm, H - 40*mm,
                     f"Datum: {inv['invoice_date']}   Splatnost: {inv['due_date']}   "
                     f"VS: {inv['variable_symbol']}")

    # 2×2 info grid
    eic = f"27ZG{cust['ico'][:8]}V"
    gw = (W - 35*mm) / 2; gh = 20*mm; gy = H - 65*mm
    cells = [
        (15*mm,           gy, "ZÁKAZNÍK",       [cust["name"], f"IČO: {cust['ico']}"]),
        (15*mm + gw + 5*mm, gy, "ODBĚRNÉ MÍSTO", [f"EIC: {eic}", cust["street"]]),
        (15*mm,           gy - gh - 3*mm, "SMLOUVA",
         [f"Č.: {cust['ico'][:6]}VO", f"Období: {inv['period_start']}–{inv['period_end']}"]),
        (15*mm + gw + 5*mm, gy - gh - 3*mm, "PLATBA",
         [f"IBAN: {sup.iban[:20]}", f"Účet: {sup.account}"]),
    ]
    c_obj.setStrokeColorRGB(0.65, 0.65, 0.65); c_obj.setLineWidth(0.4)
    for bx2, by2, title, lines in cells:
        c_obj.rect(bx2, by2 - gh, gw, gh, fill=0, stroke=1)
        c_obj.setFillColorRGB(R, G, B)
        c_obj.rect(bx2, by2 - 5*mm, gw, 5*mm, fill=1, stroke=0)
        c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7)
        c_obj.drawString(bx2 + 1.5*mm, by2 - 3.5*mm, title)
        c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7)
        for j, ln in enumerate(lines):
            c_obj.drawString(bx2 + 1.5*mm, by2 - 9*mm - j*4.5*mm, ln[:30])

    # VO table
    y3 = gy - 2*gh - 14*mm
    commodity_total   = inv.get("commodity_total",   round(inv["consumption"] * inv["unit_price"], 2))
    distrib_fixed_tot = inv.get("distrib_fixed_total", 0.0)
    distrib_var_tot   = inv.get("distrib_var_total",   0.0)
    distrib_fixed_mo  = inv.get("distrib_fixed_mo",    0.0)
    distrib_var_p     = inv.get("distrib_var_price",   0.0)
    hdrs = ["Položka", "Jedn.", "Množství", "Sazba", "Celkem Kč"]
    xs   = [15*mm, 82*mm, 100*mm, 125*mm, 163*mm]
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(15*mm, y3 - 6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
    for h2, x in zip(hdrs, xs): c_obj.drawString(x + 1*mm, y3 - 4*mm, h2)
    tbl = [
        ("Komodita – zemní plyn", "kWh", str(inv["consumption"]),
         _fmt(inv["unit_price"]) + " Kč/kWh", _fmt(commodity_total)),
        ("Distribuce pevná (6 měs.)", "měs.", "6",
         _fmt(distrib_fixed_mo) + " Kč/měs", _fmt(distrib_fixed_tot)),
        ("Distribuce variabilní", "kWh", str(inv["consumption"]),
         _fmt(distrib_var_p) + " Kč/kWh", _fmt(distrib_var_tot)),
    ]
    for ri, row in enumerate(tbl):
        bg = (0.93, 0.94, 0.97) if ri % 2 == 0 else (1, 1, 1)
        c_obj.setFillColorRGB(*bg)
        c_obj.rect(15*mm, y3 - 13*mm - ri*6.5*mm, W - 30*mm, 6.5*mm, fill=1, stroke=0)
        c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
        for val, x in zip(row, xs): c_obj.drawString(x + 1*mm, y3 - 10*mm - ri*6.5*mm, str(val)[:28])

    y4 = y3 - 40*mm
    bx3, bw4, bh4 = W - 82*mm, 67*mm, 42*mm
    c_obj.setFillColorRGB(0.96, 0.96, 0.98)
    c_obj.setStrokeColorRGB(R, G, B); c_obj.roundRect(bx3, y4 - bh4, bw4, bh4, 2*mm, fill=1, stroke=1)
    c_obj.setFillColorRGB(0.2, 0.2, 0.2); c_obj.setFont(FONT, 8)
    for i, (lbl, val) in enumerate([
        ("Základ daně:", _fmt(inv["amount_ex_vat"]) + " Kč"),
        (f"DPH {inv['vat_rate']} %:", _fmt(inv["vat_amount"]) + " Kč"),
        ("Celkem s DPH:", _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Zálohy:", f"–{_fmt(inv['advances'])} Kč"),
    ]):
        c_obj.drawString(bx3 + 3*mm, y4 - 7*mm - i*7*mm, lbl)
        c_obj.drawRightString(bx3 + bw4 - 3*mm, y4 - 7*mm - i*7*mm, val)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    lbl = "K ÚHRADĚ:" if inv["balance"] > 0 else "PŘEPLATEK:"
    c_obj.drawString(bx3 + 3*mm, y4 - 37*mm, lbl)
    c_obj.drawRightString(bx3 + bw4 - 3*mm, y4 - 37*mm, _fmt(abs(inv["balance"])) + " Kč")

    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, 0, W, 9*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT, 6.5)
    c_obj.drawCentredString(W/2, 3*mm,
        f"{sup.name}  ·  IČO {sup.ico}  ·  IBAN {sup.iban}  ·  {sup.email}")


def render_gasind(c_obj: canvas.Canvas, sup: Supplier,
                  cust: dict, inv: dict) -> None:
    """
    Layout N — GasInd (VO): FORMULÁŘOVÝ GRID styl.
    Celá faktura je jeden velký grid s labelovanými políčky — bez tabulky položek.
    Strukturálně zcela odlišné od všech ostatních layoutů: žádný banner,
    žádná tabulka, pouze série orámovaných polí v mřížce 2× sloupce.
    """
    W, H = A4; R, G, B = sup.color

    # ── Záhlaví: tenká identifikační lišta s číslem faktury ──────────────────
    c_obj.setStrokeColorRGB(0, 0, 0); c_obj.setLineWidth(1.2)
    c_obj.rect(15*mm, H - 20*mm, W - 30*mm, 20*mm, fill=0, stroke=1)
    c_obj.setFillColorRGB(0, 0, 0); c_obj.setFont(FONT_BOLD, 11)
    c_obj.drawString(17*mm, H - 9*mm, "DAŇOVÝ DOKLAD – FAKTURA")
    c_obj.setFont(FONT_BOLD, 9)
    c_obj.drawRightString(W - 17*mm, H - 9*mm,
                          f"č. {inv['invoice_number']}")
    c_obj.setFont(FONT, 8); c_obj.setFillColorRGB(0.3, 0.3, 0.3)
    c_obj.drawString(17*mm, H - 15*mm,
                     f"{sup.name}   IČO: {sup.ico}   DIČ: {sup.dic}")
    c_obj.drawRightString(W - 17*mm, H - 15*mm,
                          f"{sup.address}")

    # ── Pomocná funkce: políčko gridu ────────────────────────────────────────
    def field(x: float, y: float, w: float, h: float,
              label: str, value: str, bold_val: bool = False) -> None:
        c_obj.setStrokeColorRGB(0.5, 0.5, 0.5)
        c_obj.setLineWidth(0.35)
        c_obj.rect(x, y - h, w, h, fill=0, stroke=1)
        c_obj.setFillColorRGB(0.92, 0.92, 0.92)
        c_obj.rect(x, y - 4.5*mm, w, 4.5*mm, fill=1, stroke=0)
        c_obj.setFillColorRGB(0, 0, 0); c_obj.setFont(FONT_BOLD, 6.2)
        c_obj.drawString(x + 1.5*mm, y - 3.2*mm, label.upper())
        font = FONT_BOLD if bold_val else FONT
        c_obj.setFont(font, 8)
        c_obj.drawString(x + 1.5*mm, y - h + 2*mm, str(value)[:38])

    FW = (W - 35*mm) / 2   # šířka políčka (2 sloupce)
    FH = 11*mm              # výška políčka
    CL = 15*mm              # levý sloupec
    CR = 15*mm + FW + 5*mm  # pravý sloupec

    # ── Řádek 1: dodavatel / odběratel ───────────────────────────────────────
    y = H - 26*mm
    field(CL, y, FW, FH + 4*mm, "Dodavatel",
          f"{sup.name}, {sup.address}")
    field(CR, y, FW, FH + 4*mm, "Odběratel",
          f"{cust['name']}, {cust['street']}, {cust['city']}")

    # ── Řádek 2: IČO dod. / IČO odb. ─────────────────────────────────────────
    y -= FH + 4*mm + 1*mm
    field(CL, y, FW/2, FH, "IČO dodavatele", sup.ico)
    field(CL + FW/2, y, FW/2, FH, "DIČ dodavatele", sup.dic)
    field(CR, y, FW/2, FH, "IČO odběratele", cust["ico"])
    eic = f"27ZG{cust['ico'][:8]}V"
    field(CR + FW/2, y, FW/2, FH, "EIC odběrného místa", eic)

    # ── Řádek 3: Fakturační období / datum vystavení / splatnost ─────────────
    y -= FH + 1*mm
    field(CL, y, FW*0.6, FH, "Fakturační období",
          f"{inv['period_start']} – {inv['period_end']}")
    field(CL + FW*0.6, y, FW*0.4, FH, "Datum vystavení", inv["invoice_date"])
    field(CR, y, FW/2, FH, "Datum splatnosti", inv["due_date"])
    field(CR + FW/2, y, FW/2, FH, "Variabilní symbol", inv["variable_symbol"])

    # ── Řádek 4: IBAN / Účet ─────────────────────────────────────────────────
    y -= FH + 1*mm
    field(CL, y, FW, FH, "IBAN dodavatele", sup.iban)
    field(CR, y, FW, FH, "Číslo účtu", sup.account)

    # ── Řádek 5: spotřeba / sazba ────────────────────────────────────────────
    y -= FH + 1*mm
    commodity_total   = inv.get("commodity_total",
                                round(inv["consumption"] * inv["unit_price"], 2))
    distrib_fixed_tot = inv.get("distrib_fixed_total", 0.0)
    distrib_var_tot   = inv.get("distrib_var_total",   0.0)
    distrib_fixed_mo  = inv.get("distrib_fixed_mo",    0.0)
    distrib_var_p     = inv.get("distrib_var_price",   0.0)

    field(CL, y, FW/3, FH, "Spotřeba ZP [kWh]",
          str(inv["consumption"]))
    field(CL + FW/3, y, FW/3, FH, "Sazba komodity [Kč/kWh]",
          _fmt(inv["unit_price"]))
    field(CL + FW*2/3, y, FW/3, FH, "Komodita celkem [Kč]",
          _fmt(commodity_total))
    field(CR, y, FW/2, FH, "Distribuce pevná (6 měs.)",
          _fmt(distrib_fixed_mo) + " Kč/měs = " + _fmt(distrib_fixed_tot) + " Kč")
    field(CR + FW/2, y, FW/2, FH, "Distribuce variabilní",
          _fmt(distrib_var_p) + " Kč/kWh = " + _fmt(distrib_var_tot) + " Kč")

    # ── Řádek 6: DPH základ / DPH / celkem ──────────────────────────────────
    y -= FH + 1*mm
    field(CL, y, FW/3, FH, "Základ DPH [Kč]",
          _fmt(inv["amount_ex_vat"]))
    field(CL + FW/3, y, FW/3, FH, f"DPH {inv['vat_rate']} % [Kč]",
          _fmt(inv["vat_amount"]))
    field(CL + FW*2/3, y, FW/3, FH, "Celkem s DPH [Kč]",
          _fmt(inv["amount_with_vat"]))
    field(CR, y, FW/2, FH, "Uhrazené zálohy [Kč]",
          f"–{_fmt(inv['advances'])}")
    lbl_balance = "NEDOPLATEK" if inv["balance"] > 0 else "PŘEPLATEK"
    field(CR + FW/2, y, FW/2, FH, lbl_balance + " [Kč]",
          _fmt(abs(inv["balance"])), bold_val=True)

    # ── Patička ───────────────────────────────────────────────────────────────
    c_obj.setStrokeColorRGB(0, 0, 0); c_obj.setLineWidth(0.5)
    c_obj.line(15*mm, 14*mm, W - 15*mm, 14*mm)
    c_obj.setFillColorRGB(0.4, 0.4, 0.4); c_obj.setFont(FONT, 6.5)
    c_obj.drawString(15*mm, 10*mm,
                     f"Dodavatel: {sup.name}  ·  IČO: {sup.ico}  ·  IBAN: {sup.iban}  ·  {sup.email}")
    c_obj.drawString(15*mm, 6.5*mm,
                     "Tento doklad byl vystaven elektronicky a je daňově platný bez podpisu.")


def render_termoplus(c_obj: canvas.Canvas, sup: Supplier,
                     cust: dict, inv: dict) -> None:
    """Layout O — TermoPlus (teplo): DOPISNÍ / PROZOVÝ styl.

    Celá faktura je formátovaná jako obchodní dopis — vyúčtování je vetkáno
    do odstavců plynulého textu. Žádné tabulky, žádné barevné záhlaví sekcí,
    žádný banner. Zcela odlišná struktura od všech ostatních layoutů.
    """
    W, H = A4
    R, G, B = sup.color
    LM  = 22*mm          # left margin
    RM  = W - 22*mm      # right edge for right-aligned text
    CW  = RM - LM        # content width
    MID = LM + CW / 2    # column divider for summary box

    # ── Hlavičkový papír — odesílatel vpravo ───────────────────────────────
    # Thin vertical accent bar on the right
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(W - 8*mm, H - 55*mm, 2.5*mm, 55*mm, fill=1, stroke=0)

    c_obj.setFillColorRGB(R, G, B)
    c_obj.setFont(FONT_BOLD, 11)
    c_obj.drawRightString(W - 11*mm, H - 14*mm, sup.name)
    c_obj.setFillColorRGB(0.35, 0.35, 0.35)
    c_obj.setFont(FONT, 7.5)
    for i, ln in enumerate([sup.address,
                             f"IČO: {sup.ico}   DIČ: CZ{sup.ico}",
                             f"IBAN: {sup.iban}",
                             sup.email]):
        c_obj.drawRightString(W - 11*mm, H - 20*mm - i * 5*mm, ln)

    # Thin horizontal rule under sender block
    c_obj.setStrokeColorRGB(R, G, B)
    c_obj.setLineWidth(0.35)
    c_obj.line(W / 2, H - 43*mm, W - 11*mm, H - 43*mm)

    # ── Datum — pravý sloupec ───────────────────────────────────────────────
    c_obj.setFillColorRGB(0.25, 0.25, 0.25)
    c_obj.setFont(FONT, 8)
    c_obj.drawRightString(W - 11*mm, H - 50*mm,
                          f"Praha, {inv['invoice_date']}")

    # ── Adresát — levý blok ─────────────────────────────────────────────────
    c_obj.setFont(FONT_BOLD, 8.5)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1)
    c_obj.drawString(LM, H - 38*mm, cust["name"])
    c_obj.setFont(FONT, 8)
    c_obj.setFillColorRGB(0.25, 0.25, 0.25)
    for i, ln in enumerate([cust["street"], cust["city"],
                             f"IČO: {cust['ico']}"]):
        c_obj.drawString(LM, H - 43.5*mm - i * 4.8*mm, ln)

    # ── Věc / reference ─────────────────────────────────────────────────────
    c_obj.setStrokeColorRGB(0.75, 0.75, 0.75)
    c_obj.setLineWidth(0.25)
    c_obj.line(LM, H - 62*mm, RM, H - 62*mm)

    c_obj.setFillColorRGB(0.1, 0.1, 0.1)
    c_obj.setFont(FONT_BOLD, 9)
    c_obj.drawString(LM, H - 69*mm,
                     f"Věc:  Vyúčtování tepelné energie č. {inv['invoice_number']}")
    c_obj.setFont(FONT, 7.5)
    c_obj.setFillColorRGB(0.4, 0.4, 0.4)
    c_obj.drawString(LM, H - 75*mm,
                     f"Fakturační období: {inv['period_start']} – {inv['period_end']}"
                     f"   ·   Variabilní symbol: {inv['variable_symbol']}"
                     f"   ·   Splatnost: {inv['due_date']}")

    c_obj.setStrokeColorRGB(0.75, 0.75, 0.75)
    c_obj.line(LM, H - 78*mm, RM, H - 78*mm)

    # ── Oslovení ────────────────────────────────────────────────────────────
    c_obj.setFillColorRGB(0.12, 0.12, 0.12)
    c_obj.setFont(FONT, 9)
    c_obj.drawString(LM, H - 86*mm, "Vážený zákazníku,")

    # ── Odstavec 1 — odečet měřidla ─────────────────────────────────────────
    gj_start = round(inv["consumption"] * 2.5 + 100, 2)
    gj_end   = round(gj_start + inv["consumption"], 2)
    c_obj.setFont(FONT, 9)
    y = H - 93*mm
    for ln in [
        f"na základě odečtu tepelného měřidla za fakturační období {inv['period_start']} –",
        f"{inv['period_end']} Vám vyúčtováváme dodávku tepelné energie. Počáteční stav",
        f"měřidla činil {_fmt(gj_start)} GJ, koncový stav {_fmt(gj_end)} GJ. Spotřeba",
        f"za sledované období tak dosáhla {_fmt(inv['consumption'])} GJ při smluvní",
        f"jednotkové ceně {_fmt(inv['unit_price'])} Kč/GJ.",
    ]:
        c_obj.drawString(LM, y, ln)
        y -= 5.2*mm

    # ── Odstavec 2 — daňový výpočet ─────────────────────────────────────────
    y -= 2*mm
    for ln in [
        f"Základ daně z přidané hodnoty (bez DPH) činí {_fmt(inv['amount_ex_vat'])} Kč.",
        f"Daň z přidané hodnoty ve výši {inv['vat_rate']} % představuje",
        f"{_fmt(inv['vat_amount'])} Kč. Celková fakturovaná částka včetně DPH",
        f"je {_fmt(inv['amount_with_vat'])} Kč.",
    ]:
        c_obj.drawString(LM, y, ln)
        y -= 5.2*mm

    # ── Odstavec 3 — zálohy a výsledné saldo ────────────────────────────────
    y -= 2*mm
    balance = inv["balance"]
    if balance > 0:
        bal_lines = [
            f"Po zohlednění uhrazených záloh ve výši {_fmt(inv['advances'])} Kč",
            f"zbývá k doplatku částka {_fmt(balance)} Kč. Prosíme o její úhradu",
            f"nejpozději do {inv['due_date']} na účet č. {sup.iban},",
            f"variabilní symbol {inv['variable_symbol']}.",
        ]
    else:
        bal_lines = [
            f"Po zohlednění uhrazených záloh ve výši {_fmt(inv['advances'])} Kč",
            f"vznikl přeplatek {_fmt(abs(balance))} Kč. Tento přeplatek bude",
            f"vrácen na Váš bankovní účet do 14 pracovních dnů od data vystavení",
            f"tohoto vyúčtování.",
        ]
    for ln in bal_lines:
        c_obj.drawString(LM, y, ln)
        y -= 5.2*mm

    # ── Rozlučka ────────────────────────────────────────────────────────────
    y -= 4*mm
    c_obj.drawString(LM, y,          "V případě dotazů se prosím obraťte na naše zákaznické centrum.")
    y -= 5.2*mm
    c_obj.drawString(LM, y,          f"Děkujeme za Vaši přízeň a těšíme se na další spolupráci.")
    y -= 8*mm
    c_obj.setFont(FONT_BOLD, 9)
    c_obj.drawString(LM, y, f"S pozdravem,   {sup.name}")

    # ── Platební shrnutí — jednoduchý box u spodního okraje ─────────────────
    BH   = 45*mm
    BY   = 14*mm
    BW   = CW
    c_obj.setStrokeColorRGB(R, G, B)
    c_obj.setLineWidth(0.5)
    c_obj.rect(LM, BY, BW, BH, fill=0, stroke=1)

    # Title strip
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(LM, BY + BH - 7.5*mm, BW, 7.5*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1)
    c_obj.setFont(FONT_BOLD, 8)
    c_obj.drawString(LM + 4*mm, BY + BH - 5.2*mm, "PLATEBNÍ SHRNUTÍ")

    # Two columns inside the box
    vy = BY + BH - 13.5*mm
    c_obj.setFont(FONT, 8)
    left_pairs = [
        ("Základ daně:",          _fmt(inv["amount_ex_vat"]) + " Kč"),
        (f"DPH {inv['vat_rate']} %:", _fmt(inv["vat_amount"]) + " Kč"),
        ("Celkem s DPH:",         _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Zálohy celkem:",        f"–{_fmt(inv['advances'])} Kč"),
    ]
    right_pairs = [
        ("Číslo faktury:",  inv["invoice_number"]),
        ("Datum vystavení:", inv["invoice_date"]),
        ("Datum splatnosti:", inv["due_date"]),
        ("Číslo účtu:",     sup.iban),
    ]
    c_obj.setFillColorRGB(0.12, 0.12, 0.12)
    for (ll, lv), (rl, rv) in zip(left_pairs, right_pairs):
        c_obj.drawString(LM + 4*mm, vy, ll)
        c_obj.drawRightString(MID - 4*mm, vy, lv)
        c_obj.drawString(MID + 4*mm, vy, rl)
        c_obj.drawRightString(LM + BW - 4*mm, vy, rv)
        vy -= 5.5*mm

    # Vertical divider
    c_obj.setStrokeColorRGB(0.8, 0.8, 0.8)
    c_obj.setLineWidth(0.25)
    c_obj.line(MID, BY + 1*mm, MID, BY + BH - 8*mm)

    # K ÚHRADĚ / PŘEPLATEK
    lbl_pay = "K ÚHRADĚ:" if balance > 0 else "PŘEPLATEK:"
    c_obj.setFillColorRGB(R, G, B)
    c_obj.setFont(FONT_BOLD, 10.5)
    c_obj.drawString(LM + 4*mm, BY + 5*mm, lbl_pay)
    c_obj.drawRightString(MID - 4*mm, BY + 5*mm,
                          _fmt(abs(balance)) + " Kč")


def render_heatworks(c_obj: canvas.Canvas, sup: Supplier,
                     cust: dict, inv: dict) -> None:
    """Layout P — HeatWorks (teplo): 3 summary CARDS shown first, details below."""
    W, H = A4; R, G, B = sup.color

    # Compact header (no big band)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 13)
    c_obj.drawString(15*mm, H - 12*mm, sup.name)
    c_obj.setFillColorRGB(0.3, 0.3, 0.3); c_obj.setFont(FONT, 7.5)
    c_obj.drawString(15*mm, H - 18*mm,
                     f"{sup.address}   IČO: {sup.ico}   {sup.phone}   {sup.email}")
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(1.5)
    c_obj.line(15*mm, H - 21*mm, W - 15*mm, H - 21*mm)

    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    c_obj.drawString(15*mm, H - 29*mm,
                     f"FAKTURA – TEPELNÁ ENERGIE   č. {inv['invoice_number']}")

    # 3 AMOUNT CARDS — shown prominently before details
    cw = (W - 40*mm) / 3; ch = 22*mm; cy = H - 56*mm
    cards = [
        ("ZÁKLAD DPH", _fmt(inv["amount_ex_vat"]) + " Kč"),
        (f"DPH {inv['vat_rate']} %", _fmt(inv["vat_amount"]) + " Kč"),
        ("K ÚHRADĚ" if inv["balance"] > 0 else "PŘEPLATEK",
         _fmt(abs(inv["balance"])) + " Kč"),
    ]
    for i, (title, amount) in enumerate(cards):
        cx2 = 15*mm + i*(cw + 5*mm)
        intensity = 0.85 + i*0.05
        c_obj.setFillColorRGB(R*intensity, G*intensity, B*intensity)
        c_obj.roundRect(cx2, cy - ch, cw, ch, 2*mm, fill=1, stroke=0)
        c_obj.setFillColorRGB(1, 1, 1)
        c_obj.setFont(FONT, 7.5); c_obj.drawCentredString(cx2 + cw/2, cy - 8*mm, title)
        c_obj.setFont(FONT_BOLD, 10); c_obj.drawCentredString(cx2 + cw/2, cy - 16*mm, amount)

    # Supplier + customer blocks below cards
    y_info = cy - ch - 8*mm
    for col, title, lines in [
        (15*mm, "DODAVATEL:",
         [sup.name, sup.address, f"IČO: {sup.ico}", f"IBAN: {sup.iban}"]),
        (W/2 + 5*mm, "ODBĚRATEL:",
         [cust["name"], cust["street"], cust["city"], f"IČO: {cust['ico']}"]),
    ]:
        c_obj.setFont(FONT_BOLD, 8); c_obj.setFillColorRGB(R, G, B)
        c_obj.drawString(col, y_info, title)
        c_obj.setFont(FONT, 8); c_obj.setFillColorRGB(0.1, 0.1, 0.1)
        for j, ln in enumerate(lines):
            c_obj.drawString(col, y_info - 5*mm - j*4.5*mm, ln)

    y_meta = y_info - 28*mm
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 8)
    c_obj.drawString(15*mm, y_meta,
                     f"Fakturační období: {inv['period_start']} – {inv['period_end']}   "
                     f"Datum: {inv['invoice_date']}   Splatnost: {inv['due_date']}   "
                     f"VS: {inv['variable_symbol']}")

    # Table
    y3 = y_meta - 9*mm
    hdrs = ["Popis plnění", "GJ", "Kč/GJ", f"DPH {inv['vat_rate']} %", "Celkem Kč"]
    xs   = [15*mm, 90*mm, 112*mm, 142*mm, 163*mm]
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(15*mm, y3 - 6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
    for h2, x in zip(hdrs, xs): c_obj.drawString(x + 1*mm, y3 - 4*mm, h2)
    row = [f"Tepelná energie – {inv['period_start']} – {inv['period_end']}",
           str(inv["consumption"]), _fmt(inv["unit_price"]) + " Kč",
           f"{inv['vat_rate']} %", _fmt(inv["amount_with_vat"]) + " Kč"]
    c_obj.setFillColorRGB(0.97, 0.95, 0.92)
    c_obj.rect(15*mm, y3 - 13*mm, W - 30*mm, 7*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
    for val, x in zip(row, xs): c_obj.drawString(x + 1*mm, y3 - 10*mm, str(val)[:30])

    c_obj.setFillColorRGB(0.6, 0.6, 0.6); c_obj.setFont(FONT, 7)
    c_obj.drawCentredString(W/2, 9*mm,
        f"{sup.name}  ·  IČO {sup.ico}  ·  IBAN {sup.iban}  ·  {sup.email}")


def render_aquatown(c_obj: canvas.Canvas, sup: Supplier,
                    cust: dict, inv: dict) -> None:
    """Layout Q — AquaTown (voda): utility bill, company top-center, meter readings."""
    W, H = A4; R, G, B = sup.color

    # Top-center company header
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, H - 22*mm, W, 22*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 14)
    c_obj.drawCentredString(W/2, H - 11*mm, sup.name)
    c_obj.setFont(FONT, 8)
    c_obj.drawCentredString(W/2, H - 18*mm,
                            f"{sup.address}   IČO: {sup.ico}   {sup.phone}   {sup.email}")

    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    c_obj.drawCentredString(W/2, H - 30*mm, "FAKTURA ZA DODÁVKU PITNÉ VODY")
    c_obj.setFillColorRGB(0.3, 0.3, 0.3); c_obj.setFont(FONT, 8.5)
    c_obj.drawCentredString(W/2, H - 37*mm,
                            f"č. {inv['invoice_number']}   "
                            f"Datum: {inv['invoice_date']}   "
                            f"Splatnost: {inv['due_date']}")

    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(0.5)
    c_obj.line(15*mm, H - 40*mm, W - 15*mm, H - 40*mm)

    # Customer + meta two columns
    c_obj.setFillColorRGB(0.1, 0.1, 0.1)
    for col, header, lines in [
        (15*mm, "Odběratel:",
         [cust["name"], cust["street"], cust["city"], f"IČO: {cust['ico']}"]),
        (W/2 + 5*mm, "Platební údaje:",
         [f"IBAN: {sup.iban}", f"Účet: {sup.account}",
          f"VS: {inv['variable_symbol']}",
          f"Období: {inv['period_start']} – {inv['period_end']}"]),
    ]:
        c_obj.setFont(FONT_BOLD, 8); c_obj.drawString(col, H - 47*mm, header)
        c_obj.setFont(FONT, 8)
        for i, ln in enumerate(lines):
            c_obj.drawString(col, H - 53*mm - i*5*mm, ln)

    # Meter readings section (distinctive for water utility)
    y_mr = H - 77*mm
    c_obj.setFillColorRGB(0.90, 0.95, 0.98)
    c_obj.rect(15*mm, y_mr - 22*mm, W - 30*mm, 22*mm, fill=1, stroke=0)
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(0.5)
    c_obj.rect(15*mm, y_mr - 22*mm, W - 30*mm, 22*mm, fill=0, stroke=1)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 8)
    c_obj.drawString(17*mm, y_mr - 5*mm, "ODEČET VODOMĚRU")
    m3_start = round(inv["consumption"] * 3.8 + 200, 1)
    m3_end   = round(m3_start + inv["consumption"], 1)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 8)
    c_obj.drawString(17*mm, y_mr - 11*mm,
                     f"Stav začátek: {_fmt(m3_start)} m³      "
                     f"Stav konec: {_fmt(m3_end)} m³      "
                     f"Spotřeba: {inv['consumption']} m³")
    c_obj.drawString(17*mm, y_mr - 17*mm,
                     f"Cena vody: {_fmt(inv['unit_price'])} Kč/m³   "
                     f"DPH: {inv['vat_rate']} %   "
                     f"Č. smlouvy: {cust['ico'][:6]}VOD")

    # Table
    y3 = y_mr - 30*mm
    hdrs = ["Položka", "m³", "Kč/m³", f"DPH {inv['vat_rate']} %", "Celkem Kč"]
    xs   = [15*mm, 88*mm, 110*mm, 140*mm, 163*mm]
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(15*mm, y3 - 6*mm, W - 30*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
    for h2, x in zip(hdrs, xs): c_obj.drawString(x + 1*mm, y3 - 4*mm, h2)
    row = [f"Pitná voda – {inv['period_start']} – {inv['period_end']}",
           str(inv["consumption"]), _fmt(inv["unit_price"]) + " Kč",
           f"{inv['vat_rate']} %", _fmt(inv["amount_with_vat"]) + " Kč"]
    c_obj.setFillColorRGB(0.93, 0.96, 0.99)
    c_obj.rect(15*mm, y3 - 13*mm, W - 30*mm, 7*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
    for val, x in zip(row, xs): c_obj.drawString(x + 1*mm, y3 - 10*mm, str(val)[:32])

    # Summary
    y4 = y3 - 22*mm
    bx, bw2, bh2 = W - 82*mm, 67*mm, 42*mm
    c_obj.setFillColorRGB(0.94, 0.97, 0.99)
    c_obj.setStrokeColorRGB(R, G, B); c_obj.roundRect(bx, y4 - bh2, bw2, bh2, 2*mm, fill=1, stroke=1)
    c_obj.setFillColorRGB(0.2, 0.2, 0.2); c_obj.setFont(FONT, 8)
    for i, (lbl, val) in enumerate([
        ("Základ daně:", _fmt(inv["amount_ex_vat"]) + " Kč"),
        (f"DPH {inv['vat_rate']} %:", _fmt(inv["vat_amount"]) + " Kč"),
        ("Celkem s DPH:", _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Zálohy:", f"–{_fmt(inv['advances'])} Kč"),
    ]):
        c_obj.drawString(bx + 3*mm, y4 - 7*mm - i*7*mm, lbl)
        c_obj.drawRightString(bx + bw2 - 3*mm, y4 - 7*mm - i*7*mm, val)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    lbl = "K ÚHRADĚ:" if inv["balance"] > 0 else "PŘEPLATEK:"
    c_obj.drawString(bx + 3*mm, y4 - 37*mm, lbl)
    c_obj.drawRightString(bx + bw2 - 3*mm, y4 - 37*mm, _fmt(abs(inv["balance"])) + " Kč")

    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, 0, W, 9*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT, 6.5)
    c_obj.drawCentredString(W/2, 3*mm,
        f"{sup.name}  ·  IČO {sup.ico}  ·  IBAN {sup.iban}  ·  {sup.email}")


def render_clearwater(c_obj: canvas.Canvas, sup: Supplier,
                      cust: dict, inv: dict) -> None:
    """Layout R — ClearWater (voda): left sidebar with rotated invoice#, clean modern."""
    W, H = A4; R, G, B = sup.color
    SW = 20*mm  # sidebar width

    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(0, 0, SW, H, fill=1, stroke=0)
    # Rotated invoice number
    c_obj.saveState()
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 8)
    c_obj.translate(SW/2, H/2)
    c_obj.rotate(90)
    c_obj.drawCentredString(0, 0, f"FAKTURA  {inv['invoice_number']}")
    c_obj.restoreState()
    c_obj.saveState()
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT, 6.5)
    c_obj.translate(SW/2, H/3)
    c_obj.rotate(90)
    c_obj.drawCentredString(0, 0, f"VS: {inv['variable_symbol']}")
    c_obj.restoreState()

    MX = SW + 6*mm
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 13)
    c_obj.drawString(MX, H - 12*mm, sup.name)
    c_obj.setFillColorRGB(0.3, 0.3, 0.3); c_obj.setFont(FONT, 7.5)
    c_obj.drawString(MX, H - 18*mm,
                     f"{sup.address}  ·  IČO: {sup.ico}  ·  {sup.email}")
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(1.0)
    c_obj.line(MX, H - 21*mm, W - 10*mm, H - 21*mm)

    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    c_obj.drawString(MX, H - 30*mm, "FAKTURA – PITNÁ VODA")
    c_obj.setFillColorRGB(0.3, 0.3, 0.3); c_obj.setFont(FONT, 8)
    c_obj.drawString(MX, H - 37*mm,
                     f"Datum: {inv['invoice_date']}   Splatnost: {inv['due_date']}")

    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT_BOLD, 8)
    c_obj.drawString(MX, H - 47*mm, "Odběratel:")
    c_obj.setFont(FONT, 8)
    for i, ln in enumerate([cust["name"], cust["street"], cust["city"],
                             f"IČO: {cust['ico']}"]):
        c_obj.drawString(MX, H - 53*mm - i*5*mm, ln)

    c_obj.setFont(FONT, 7.5); c_obj.setFillColorRGB(0.3, 0.3, 0.3)
    c_obj.drawString(MX, H - 75*mm,
                     f"Fakturační období: {inv['period_start']} – {inv['period_end']}   "
                     f"IBAN: {sup.iban}")
    m3_start = round(inv["consumption"] * 4.1 + 150, 1)
    m3_end   = round(m3_start + inv["consumption"], 1)
    c_obj.drawString(MX, H - 81*mm,
                     f"Vodoměr: {_fmt(m3_start)} → {_fmt(m3_end)} m³   "
                     f"Spotřeba: {inv['consumption']} m³")

    y3 = H - 90*mm
    hdrs = ["Popis plnění", "Spotřeba m³", "Cena/m³", "DPH", "Celkem Kč"]
    xs   = [MX, MX + 60*mm, MX + 88*mm, MX + 118*mm, MX + 142*mm]
    c_obj.setFillColorRGB(R, G, B)
    c_obj.rect(MX, y3 - 6*mm, W - MX - 10*mm, 6*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(1, 1, 1); c_obj.setFont(FONT_BOLD, 7.5)
    for h2, x in zip(hdrs, xs): c_obj.drawString(x + 1*mm, y3 - 4*mm, h2)
    row = [f"Voda – {inv['period_start']}–{inv['period_end']}",
           str(inv["consumption"]), _fmt(inv["unit_price"]) + " Kč",
           f"{inv['vat_rate']} %", _fmt(inv["amount_with_vat"]) + " Kč"]
    c_obj.setFillColorRGB(0.92, 0.97, 0.98)
    c_obj.rect(MX, y3 - 13*mm, W - MX - 10*mm, 7*mm, fill=1, stroke=0)
    c_obj.setFillColorRGB(0.1, 0.1, 0.1); c_obj.setFont(FONT, 7.5)
    for val, x in zip(row, xs): c_obj.drawString(x + 1*mm, y3 - 10*mm, str(val)[:28])

    y4 = y3 - 26*mm
    for i, (lbl, val) in enumerate([
        ("Základ daně:", _fmt(inv["amount_ex_vat"]) + " Kč"),
        (f"DPH {inv['vat_rate']} %:", _fmt(inv["vat_amount"]) + " Kč"),
        ("Celkem s DPH:", _fmt(inv["amount_with_vat"]) + " Kč"),
        ("Zálohy:", f"–{_fmt(inv['advances'])} Kč"),
    ]):
        c_obj.setFont(FONT, 8); c_obj.drawRightString(W - 12*mm, y4 - i*5.5*mm, val)
        c_obj.setFont(FONT, 8); c_obj.drawRightString(W - 55*mm, y4 - i*5.5*mm, lbl)
    c_obj.setStrokeColorRGB(R, G, B); c_obj.setLineWidth(1.2)
    c_obj.line(W - 72*mm, y4 - 25*mm, W - 12*mm, y4 - 25*mm)
    c_obj.setFillColorRGB(R, G, B); c_obj.setFont(FONT_BOLD, 11)
    lbl = "K ÚHRADĚ:" if inv["balance"] > 0 else "PŘEPLATEK:"
    c_obj.drawRightString(W - 55*mm, y4 - 31*mm, lbl)
    c_obj.drawRightString(W - 12*mm, y4 - 31*mm, _fmt(abs(inv["balance"])) + " Kč")


RENDERERS = {
    "novatech":   render_novatech,
    "energyplus": render_energyplus,
    "sparkelit":  render_sparkelit,
    "voltpro":    render_voltpro,
    "highvolt":   render_highvolt,
    "elnord":     render_elnord,
    "bohemiagas": render_bohemiagas,
    "gaspraha":   render_gaspraha,
    "termoplyn":  render_termoplyn,
    "indugas":    render_indugas,
    "progas_vo":  render_progas_vo,
    "gasind":     render_gasind,
    "thermocity": render_thermocity,
    "termoplus":  render_termoplus,
    "heatworks":  render_heatworks,
    "aquaregion": render_aquaregion,
    "aquatown":   render_aquatown,
    "clearwater": render_clearwater,
}


# ── PDF → obrázek → degradace → PDF ──────────────────────────────────────────

def _pdf_to_image(pdf_bytes: bytes, dpi: int) -> Image.Image:
    doc = pdfium.PdfDocument(pdf_bytes)
    bm  = doc[0].render(scale=dpi / 72)
    img = bm.to_pil().convert("RGB")
    doc.close()
    return img


def _degrade_q1(img: Image.Image) -> Image.Image:
    """Čistý scan (Q1) — jen velmi jemný šum a papírový tón, žádné artefakty."""
    arr = np.array(img, dtype=np.float32)
    arr += np.random.normal(0, 2, arr.shape).astype(np.float32)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _degrade_q2(img: Image.Image, rng: random.Random) -> Image.Image:
    """Průměrný scan (Q2) — tři varianty degradace, náhodně vybrána."""
    mode = rng.choice(["blur_skew", "scanlines", "warm_tint"])

    if mode == "blur_skew":
        img  = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.5, 0.9)))
        arr  = np.array(img, dtype=np.float32)
        arr += np.random.normal(0, rng.uniform(4, 8), arr.shape).astype(np.float32)
        arr  = np.clip(arr, 0, 255).astype(np.uint8)
        img  = Image.fromarray(arr)
        img  = ImageEnhance.Contrast(img).enhance(rng.uniform(0.87, 0.96))
        skew = rng.uniform(0.3, 1.0) * (1 if rng.random() > 0.5 else -1)
        img  = img.rotate(skew, expand=False, fillcolor=(242, 242, 242))

    elif mode == "scanlines":
        img  = img.filter(ImageFilter.GaussianBlur(radius=0.4))
        arr  = np.array(img, dtype=np.float32)
        arr += np.random.normal(0, 4, arr.shape).astype(np.float32)
        # horizontal scan lines artifact
        step = rng.randint(7, 14)
        for row in range(0, arr.shape[0], step):
            arr[row:row+1, :] *= rng.uniform(0.80, 0.93)
        arr  = np.clip(arr, 0, 255).astype(np.uint8)
        img  = Image.fromarray(arr)

    else:  # warm_tint (old paper, slight yellowing)
        arr  = np.array(img, dtype=np.float32)
        arr += np.random.normal(0, 5, arr.shape).astype(np.float32)
        arr[:, :, 0] = np.clip(arr[:, :, 0] * rng.uniform(1.02, 1.07), 0, 255)
        arr[:, :, 1] = np.clip(arr[:, :, 1] * rng.uniform(1.00, 1.04), 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] * rng.uniform(0.86, 0.94), 0, 255)
        arr  = np.clip(arr, 0, 255).astype(np.uint8)
        img  = Image.fromarray(arr)
        img  = ImageEnhance.Contrast(img).enhance(rng.uniform(0.88, 0.95))

    return img


def _add_stamp(img: Image.Image, rng: random.Random) -> Image.Image:
    """Přidá razítko (kruh s textem) — typický artefakt skenů."""
    draw   = ImageDraw.Draw(img)
    w, h   = img.size
    corner = rng.choice(["tl", "tr", "bl", "br"])
    r      = rng.randint(55, 90)
    margin = rng.randint(30, 60)
    if corner == "tl":   cx, cy = margin + r, margin + r
    elif corner == "tr": cx, cy = w - margin - r, margin + r
    elif corner == "bl": cx, cy = margin + r, h - margin - r
    else:                cx, cy = w - margin - r, h - margin - r
    col = rng.choice([(85, 105, 155), (135, 50, 50), (45, 105, 70)])
    lw  = rng.randint(3, 6)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],     outline=col, width=lw)
    draw.ellipse([cx - r + 8, cy - r + 8, cx + r - 8, cy + r - 8],
                 outline=col, width=1)
    txt   = rng.choice(["ZAPLACENO", "UHRAZENO", "PŘIJATO", "ARCHIV", "KONTROLA"])
    txt_w = len(txt) * 7
    draw.text((cx - txt_w // 2, cy - 5), txt, fill=col)
    return img


def _degrade_q3(img: Image.Image, rng: random.Random) -> Image.Image:
    """
    Špatný scan (Q3) — tři varianty: razítko / ohyb stránky / vignet.
    Text je čitelný, ale plný artefaktů.
    """
    mode = rng.choice(["stamp", "fold", "vignette"])

    # Common base degradation
    img  = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.7, 1.1)))
    arr  = np.array(img, dtype=np.float32)
    arr += np.random.normal(0, rng.uniform(8, 13), arr.shape).astype(np.float32)
    arr  = np.clip(arr, 0, 255).astype(np.uint8)
    img  = Image.fromarray(arr)
    img  = ImageEnhance.Contrast(img).enhance(rng.uniform(0.78, 0.86))

    if mode == "stamp":
        img = _add_stamp(img, rng)
        skew = rng.uniform(1.5, 3.0) * (1 if rng.random() > 0.5 else -1)
        img = img.rotate(skew, expand=False, fillcolor=(218, 218, 218))

    elif mode == "fold":
        arr2 = np.array(img, dtype=np.float32)
        h = arr2.shape[0]
        fold_y = rng.randint(h // 5, 4 * h // 5)
        fold_w = rng.randint(3, 7)
        arr2[fold_y:fold_y + fold_w, :] *= rng.uniform(0.48, 0.68)
        arr2[max(0, fold_y - 2):fold_y, :] = np.clip(
            arr2[max(0, fold_y - 2):fold_y, :] * 1.12, 0, 255)
        arr2 = np.clip(arr2, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr2)
        skew = rng.uniform(1.0, 2.5) * (1 if rng.random() > 0.5 else -1)
        img = img.rotate(skew, expand=False, fillcolor=(220, 220, 220))

    else:  # vignette — dark corners
        arr2 = np.array(img, dtype=np.float32)
        h2, w2 = arr2.shape[:2]
        yy, xx = np.ogrid[:h2, :w2]
        cy2, cx2 = h2 / 2.0, w2 / 2.0
        dist = np.sqrt(((yy - cy2) / cy2) ** 2 + ((xx - cx2) / cx2) ** 2)
        mask = np.clip(1.0 - 0.38 * np.clip(dist - 0.55, 0, 1) / 0.45, 0.58, 1.0)
        arr2 = arr2 * mask[:, :, np.newaxis]
        arr2 = np.clip(arr2, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr2)
        img = _add_stamp(img, rng)
        skew = rng.uniform(2.0, 3.5) * (1 if rng.random() > 0.5 else -1)
        img = img.rotate(skew, expand=False, fillcolor=(215, 215, 215))

    return img


def _image_to_pdf_bytes(img: Image.Image) -> bytes:
    """Obalí PIL obrázek jako jednostránkové PDF (bez textové vrstvy)."""
    import os, tempfile
    buf = io.BytesIO()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tmp_path = tf.name
    try:
        img.save(tmp_path, format="JPEG", quality=88)
        c_obj = canvas.Canvas(buf, pagesize=A4)
        c_obj.drawImage(tmp_path, 0, 0, width=A4[0], height=A4[1],
                        preserveAspectRatio=False)
        c_obj.save()
    finally:
        os.unlink(tmp_path)
    return buf.getvalue()


# ── Hlavní smyčka ─────────────────────────────────────────────────────────────

def generate_all(n_per_tier: int = 3) -> None:
    """
    Vygeneruje n_per_tier faktur na každou úroveň kvality pro každého dodavatele.
    Celkem: 18 dodavatelů × 3 úrovně × n_per_tier = 162 PDF (při n=3).
    Všechny PDF jsou rastrované skeny (bez textové vrstvy).
    Vedlejší soubor: data/synthetic/ground_truth.json
    """
    import json
    rng = random.Random(42)
    ground_truth: dict = {}

    TIER_DPI = {"Q1": 200, "Q2": 150, "Q3": 120}

    for sup in SUPPLIERS:
        print(f"\n── {sup.name} ({sup.commodity_key}) ──────────────────────────────")
        renderer = RENDERERS[sup.key]

        combos = [(c, p) for c in CUSTOMERS for p in PERIODS]
        rng.shuffle(combos)
        selected = combos[: n_per_tier * 3]

        groups = {
            "Q1": selected[:n_per_tier],
            "Q2": selected[n_per_tier: n_per_tier*2],
            "Q3": selected[n_per_tier*2:],
        }

        for tier, cases in groups.items():
            out_dir = OUT_DIR / sup.commodity_key / sup.key / tier
            out_dir.mkdir(parents=True, exist_ok=True)

            for idx, (cust, (p_start, p_end)) in enumerate(cases, 1):
                inv   = _invoice_amounts(sup, p_start, p_end, rng)
                fname = f"faktura_{inv['invoice_number']}.pdf"

                # vykresli čisté PDF
                buf   = io.BytesIO()
                c_obj = canvas.Canvas(buf, pagesize=A4)
                renderer(c_obj, sup, cust, inv)
                c_obj.save()
                clean_bytes = buf.getvalue()

                # rasterizuj → degraduj → zabal zpět jako scan PDF
                dpi = TIER_DPI[tier]
                img = _pdf_to_image(clean_bytes, dpi)
                if tier == "Q1":
                    img = _degrade_q1(img)
                elif tier == "Q2":
                    img = _degrade_q2(img, rng)
                else:
                    img = _degrade_q3(img, rng)
                scan_pdf = _image_to_pdf_bytes(img)
                (out_dir / fname).write_bytes(scan_pdf)

                # ground truth
                rel = f"data/synthetic/{sup.commodity_key}/{sup.key}/{tier}/{fname}"
                ground_truth[rel] = {
                    "supplier_name":    sup.name,
                    "supplier_ico":     sup.ico,
                    "commodity":        sup.commodity,
                    "commodity_key":    sup.commodity_key,
                    "quality_tier":     tier,
                    "invoice_number":   inv["invoice_number"],
                    "invoice_date":     inv["invoice_date"],
                    "due_date":         inv["due_date"],
                    "period_start":     inv["period_start"],
                    "period_end":       inv["period_end"],
                    "consumption":      inv["consumption"],
                    "unit":             inv["unit"],
                    "unit_price":       round(float(inv["unit_price"]), 4),
                    "vat_rate":         inv["vat_rate"],
                    "amount_ex_vat":    inv["amount_ex_vat"],
                    "vat_amount":       inv["vat_amount"],
                    "amount_with_vat":  inv["amount_with_vat"],
                    "advances":         inv["advances"],
                    "balance":          inv["balance"],
                    "variable_symbol":  inv["variable_symbol"],
                    "customer_name":    cust["name"],
                    "customer_ico":     cust["ico"],
                    "customer_street":  cust["street"],
                    "customer_city":    cust["city"],
                }

                print(f"  {tier} [{idx}/{n_per_tier}]  {fname}")

    # ── uložení ground truth ──────────────────────────────────────────────────
    gt_path = OUT_DIR / "ground_truth.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)
    print(f"\nGround truth: {gt_path}  ({len(ground_truth)} záznamů)")

    # ── přehled ───────────────────────────────────────────────────────────────
    total = sum(1 for _ in OUT_DIR.rglob("*.pdf"))
    print(f"\n{'='*60}")
    print(f"Syntetické faktury vygenerovány: {total} PDF (všechny jsou skeny)")
    print(f"Složka: {OUT_DIR}")
    print()
    for sup in SUPPLIERS:
        for tier in ["Q1", "Q2", "Q3"]:
            d = OUT_DIR / sup.commodity_key / sup.key / tier
            n = sum(1 for _ in d.glob("*.pdf")) if d.exists() else 0
            print(f"  {sup.commodity_key}/{sup.key}/{tier}: {n} PDF")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=3,
                   help="Počet faktur na úroveň kvality (default: 3, celkem 6×3×N PDF)")
    args = p.parse_args()
    generate_all(n_per_tier=args.n)
