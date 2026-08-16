"""LangChain-based extraction strategy using LLMs with LCEL.

Uses LangChain Expression Language (LCEL) to call language models
(OpenAI, Anthropic) for structured data extraction from invoice text.
Returns parsed Pydantic models directly.

Extrakční strategie založená na LangChain a LLM s LCEL.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from src.core.extraction.base import BaseExtractionStrategy, ExtractionContext
from src.core.vat_utils import (
    apply_vat_derivation_to_invoice,
    apply_vat_inc_correction_to_invoice,
)
from src.domain.entities import (
    BillingPeriod,
    CommodityType,
    ElectricityNNData,
    ElectricityVNData,
    ExtractionResult,
    GasMOData,
    GasVOData,
    HeatData,
    InvoiceData,
    InvoiceType,
    SupplyPoint,
    WaterData,
    parse_czech_date,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


logger = logging.getLogger(__name__)

# Fix 3: deterministic amount_inc_vat correction (recompute from ex_vat × VAT
# rate and replace when the model's value disagrees by > tolerance). Toggle via
# ENABLE_VAT_INC_CORRECTION so before/after comparisons stay possible.
ENABLE_VAT_INC_CORRECTION = True


# ════════════════════════════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ════════════════════════════════════════════════════════════════════════════


SYSTEM_PROMPT = """You are a specialist in extracting structured data from Czech utility invoices (elektřina, plyn, voda, teplo).
Return ONLY valid JSON matching the requested fields. Use null for any field not clearly present in the text.

══ FORMÁTY ČÍSEL A DATUMŮ ══════════════════════════════════════════════════════
• Čísla: mezera nebo tečka jako oddělovač tisíců, čárka jako desetinná — "1 200,50" = 1200.50
• Datum: DD.MM.YYYY (např. "31.10.2023"). Akceptuj i YYYY-MM-DD.
• Částky: vždy v CZK (Kč).

══ IDENTIFIKACE DOKLADU ═══════════════════════════════════════════════════════
• Číslo faktury / Číslo dokladu / Faktura č. → invoice_number
• Variabilní symbol (VS) → variable_symbol (může být stejné jako číslo faktury)
• Odběrné místo / EAN / EIC / Číslo OM → consumption_point_code
  – Elektřina EAN: 18 číslic začínajících 859
  – Plyn EIC: 16 znaků začínajících 27ZG
• IČO = daňové identifikační číslo (8 číslic, doplnit nulami zleva):
  – IČO odběratele / IČ zákazníka → customer_tax_id
  – IČO dodavatele / IČ firmy → supplier_tax_id
• DIČ odběratele → customer_vat_id; DIČ dodavatele → supplier_vat_id

══ KLÍČOVÁ DATA ════════════════════════════════════════════════════════════════
• Období od / Fakturační období od → period_from
• Období do / Fakturační období do → period_to
• Datum vystavení → issue_date (VŽDY nastane PO datu period_to)
• Datum splatnosti / Splatno do → due_date
• DUZP / Datum uskutečnění zdanitelného plnění / Datum UZP → tax_point_date
  (DUZP bývá shodné s period_to nebo posledním dnem měsíce fakturace)

══ PRAVIDLA PRO CELKOVÉ ČÁSTKY (KRITICKÉ) ══════════════════════════════════
Hledej v sekci "Rekapitulace DPH" nebo "Celková cena" (obvykle poslední strana):
• total_amount_ex_vat = "Základ daně" / "Cena celkem bez DPH" / "Celkem bez DPH" / "Základ"
• total_amount_inc_vat = "Cena celkem s DPH" / "Celkem s DPH" / "Celkem vč. DPH" / "Celková cena"
• vat_rate = sazba DPH v % (číslo: 21.0, 12.0, 10.0 nebo 0.0)

NIKDY nepoužívej pro total_amount_* tato pole:
• "K úhradě" / "Doplatek" / "Nedoplatek" = částka k zaplacení PO odečtení záloh → IGNORUJ
• "Záloha" / "Zálohová platba" / "Uhrazené zálohy" → IGNORUJ
• "Přeplatek" / "Zůstatek" / "Celkový dluh" → IGNORUJ
• Dílčí základy DPH pro jednotlivé sazby (21 % × základ) → IGNORUJ

Pokud faktura obsahuje více sazeb DPH (např. 21 % a 10 %): pro vat_rate použij sazbu
platnou pro hlavní komoditní složku. total_amount_ex_vat a total_amount_inc_vat
jsou SOUČTY za celou fakturu.

Opravná faktura / Dobropis / Vrubopis: částky mohou být záporné — extrahuj je záporné.

══ STRUKTURA TABULEK ═══════════════════════════════════════════════════════════
Vícekolonové řádky jsou formátovány jako Markdown pipe-tabulky:
  | Popis položky | Jednotka | Množství | Cena/j | Celkem Kč |

Řádek PŘED tabulkou = název sekce (např. "DISTRIBUCE ELEKTŘINY").
Při hledání hodnoty:
• Sloupec "Celkem" / "Celkem Kč" / "Částka" → peněžní hodnota položky (VŽDY poslední sloupec)
• Sloupec "Množství" / "Spotřeba" / "Hodnota" → fyzikální množství
• Sloupec "Cena/j" / "Jedn. cena" / "Sazba" → jednotková cena

Subtotal sekce: řádek s "Celkem" nebo "Cena celkem" UVNITŘ sekce — není to grand total!
Pro každé pole hledej konkrétní řádek ve správné sekci — viz detaily v commodity polích."""


EXTRACTION_PROMPT_TEMPLATE = """Extrahuj strukturovaná data z této faktury za {commodity_type}.

POVINNÁ POLE (vyplň vždy, pokud jsou na faktuře):
- invoice_number: string — číslo faktury / číslo dokladu
- period_from: date (DD.MM.YYYY) — začátek fakturačního období
- period_to: date (DD.MM.YYYY) — konec fakturačního období
- issue_date: date (DD.MM.YYYY) — datum vystavení
- due_date: date (DD.MM.YYYY) — datum splatnosti
- total_amount_ex_vat: number (CZK) — celková cena BEZ DPH (základ daně)
- total_amount_inc_vat: number (CZK) — celková cena S DPH

VOLITELNÁ OBECNÁ POLE:
- tax_point_date: date (DD.MM.YYYY) — DUZP, datum uskutečnění zdanitelného plnění
- vat_rate: number — sazba DPH v % (např. 21.0, 12.0, 10.0, 0.0)
- variable_symbol: string — variabilní symbol (VS)
- consumption_point_code: string — číslo odběrného místa (EAN/EIC/OM)
- customer_tax_id: string (8 číslic) — IČO odběratele
- supplier_tax_id: string (8 číslic) — IČO dodavatele

{commodity_specific_fields}

Text faktury:
```
{invoice_text}
```

Vrať platný JSON objekt se všemi výše uvedenými klíči. Chybějící hodnoty = null."""


ELECTRICITY_NN_FIELDS = """ELEKTŘINA NN — specifická pole spotřeby:

Spotřeba (kWh) — hledej v tabulce s hlavičkou "Spotřeba", "Odečet" nebo "Vyúčtování spotřeby":
- consumption_low_tariff: number (kWh)
    → řádek "Spotřeba NT" / "Nízký tarif" / "NT" — sloupec MNOŽSTVÍ
    → POZOR: není u jednosazbových tarifů (D01d, D02d) — v tom případě null
- consumption_high_tariff: number (kWh)
    → řádek "Spotřeba VT" / "Vysoký tarif" / "VT" / "Spotřeba" (jednosazbový → celá spotřeba)
    → u jednosazbového tarifu (D01d, D02d): CELÁ spotřeba jde sem
- total_consumption: number (kWh)
    → "Celková spotřeba" / "Spotřeba celkem" — součet NT + VT
    → pokud není uvedeno, nepokoušej se počítat — vrať null

PŘÍKLAD správné extrakce (dvojsazbový tarif D25d):
Úsek faktury:
  | Spotřeba VT | kWh | 834 |
  | Spotřeba NT | kWh | 421 |
  | Celková spotřeba | kWh | 1 255 |
Správná extrakce:
  consumption_high_tariff = 834
  consumption_low_tariff = 421
  total_consumption = 1255

PŘÍKLAD (jednosazbový tarif D01d — pouze jedna hodnota spotřeby):
  | Spotřeba | kWh | 612 |
Správná extrakce:
  consumption_high_tariff = 612
  consumption_low_tariff = null
  total_consumption = null"""


ELECTRICITY_VN_FIELDS = """ELEKTŘINA VN — specifická pole (vysoké napětí).

Faktura VN má TABULKOVOU STRUKTURU: každá sekce = nadpis + tabulka řádků.
Každý řádek tabulky: | Popis | Jednotka | Množství/Hodnota | Cena/j | Celkem Kč |
Pro každé pole níže: hledej PŘESNÝ řádek v příslušné sekci.

── SEKCE: SILOVÁ ELEKTŘINA / DODÁVKA ELEKTŘINY ──────────────────────────────
- supply_consumption: number (MWh)
    → řádek "Spotřeba SE" / "Spotřeba elektřiny" / "Odběr" — sloupec MNOŽSTVÍ (MWh)
- supply_charge: number (CZK)
    → řádek "Silová elektřina" / "Cena silové elektřiny" / "Komodita SE" — sloupec CELKEM
- supply_tax_charge: number (CZK)
    → řádek "Daň ze silové elektřiny" / "Daň SE" / "Spotřební daň" — sloupec CELKEM

── SEKCE: PŘENOS ELEKTŘINY / PŘENOSOVÁ SOUSTAVA ─────────────────────────────
- grid_usage_rate: number (CZK/MWh)
    → řádek "Přenos elektřiny" / "Použití přenosové soustavy" — sloupec CENA/J
- grid_usage_charge: number (CZK)
    → řádek "Přenos elektřiny" / "Použití přenosové soustavy" / "Přenos VN" — sloupec CELKEM
- monthly_reserved_capacity: number (MW nebo kW — ulož v původní jednotce)
    → řádek "Rezervovaná kapacita" / "RK měsíční" / "Měsíční RK" — sloupec MNOŽSTVÍ
- monthly_reserved_capacity_charge: number (CZK)
    → stejný řádek jako výše — sloupec CELKEM
- annual_reserved_capacity: number (MW)
    → řádek "Roční rezervovaná kapacita" / "RK roční" — sloupec MNOŽSTVÍ
- annual_reserved_capacity_charge: number (CZK)
    → stejný řádek — sloupec CELKEM
- reserved_capacity_excess: number (MW)
    → řádek "Překročení rezervované kapacity" / "Překročení RK" — sloupec MNOŽSTVÍ
    → pokud překročení nenastalo, bývá 0 nebo řádek chybí → null nebo 0
- reserved_capacity_excess_rate: number (CZK/MW)
    → stejný řádek — sloupec CENA/J
- reserved_capacity_excess_charge: number (CZK)
    → stejný řádek — sloupec CELKEM

── SEKCE: ČTVRTHODINOVÉ MAXIMUM ─────────────────────────────────────────────
- quarter_hour_max: number (kW nebo MW)
    → řádek "Čtvrthodinové maximum" / "ČHM" / "Max. čtvrthodinový výkon" — sloupec HODNOTA/MNOŽSTVÍ
- eru_rate: number (CZK/kW nebo CZK/MW)
    → řádek "Sazba ERÚ" / "Regulovaná složka ERÚ" — sloupec CENA/J nebo HODNOTA

── SEKCE: JALOVÁ ENERGIE / KOMPENZACE JALOVÉ ENERGIE ────────────────────────
- power_factor: number (bezrozměrné, tg φ nebo cos φ)
    → řádek "tg φ" / "cos φ" / "Účiník" — sloupec HODNOTA
- reactive_power_quantity: number (kVArh)
    → řádek "Jalová energie" / "Množství jalové energie" — sloupec MNOŽSTVÍ
- reactive_power_rate: number (CZK/kVArh)
    → stejný řádek — sloupec CENA/J
- reactive_power_charge: number (CZK)
    → řádek "Jalová energie" / "Kompenzace jalové energie" — sloupec CELKEM

── SEKCE: SYSTÉMOVÉ SLUŽBY / PROVOZ SÍTĚ ────────────────────────────────────
- service_price: number (CZK)
    → řádek "Cena služby" / "Systémové služby" / "Cena za systém. služby" — sloupec CELKEM
- operating_price: number (CZK)
    → řádek "Cena za provoz" / "Provozní složka" / "Cena provozu sítě" — sloupec CELKEM

── SEKCE: POZE / OZE / OBNOVITELNÉ ZDROJE ───────────────────────────────────
- renewable_energy_fee: number (CZK)
    → řádek "POZE" / "Podpora obnovitelných zdrojů" / "OZE + KVET" — sloupec CELKEM

POZNÁMKA: Pokud sekce nebo řádek na faktuře neexistuje, vrať null. Nevymýšlej hodnoty.

PŘÍKLAD VN faktury — úsek distribučních tabulek:
  Sekce: SILOVÁ ELEKTŘINA
  | Spotřeba elektřiny | MWh | 45,800 | 1 250,00 | 57 250,00 |
  | Silová elektřina | Kč | | | 57 250,00 |
  | Daň ze silové elektřiny | Kč | | | 1 145,00 |

  Sekce: PŘENOS ELEKTŘINY
  | Přenos elektřiny | MWh | 45,800 | 89,50 | 4 099,10 |
  | Rezervovaná kapacita měsíční | kW | 120 | 45,20 | 5 424,00 |
  | Překročení rezervované kapacity | kW | 0 | 500,00 | 0,00 |

  Sekce: JALOVÁ ENERGIE
  | tg φ | — | 0,124 | | |
  | Jalová energie | kVArh | 2 400 | 0,042 | 100,80 |

  Sekce: POZE
  | POZE | Kč | | | 3 200,00 |

Správná extrakce:
  supply_consumption = 45.8
  supply_charge = 57250.0
  supply_tax_charge = 1145.0
  grid_usage_charge = 4099.10
  grid_usage_rate = 89.5
  monthly_reserved_capacity = 120.0
  monthly_reserved_capacity_charge = 5424.0
  reserved_capacity_excess = 0.0
  reserved_capacity_excess_charge = 0.0
  power_factor = 0.124
  reactive_power_quantity = 2400.0
  reactive_power_rate = 0.042
  reactive_power_charge = 100.8
  renewable_energy_fee = 3200.0"""


GAS_MO_FIELDS = """PLYN MO (maloodběr) — specifická pole:

── SPOTŘEBA A PŘEPOČET ───────────────────────────────────────────────────────
- consumption_m3: number (m³)
    → řádek "Spotřeba plynu" / "Naměřená spotřeba" / "Objem" — hodnota v m³
    → POZOR: v letních měsících může být 0 m³ — extrahuj 0.0, ne null
- consumption_mwh: number (MWh)
    → řádek "Spotřeba v MWh" / "Energie" — hodnota v MWh
    → ALTERNATIVA: někdy uvedeno jako "Spotřeba" × koeficient přepočtu × spalné teplo / 3,6
- combustion_heat: number (MJ/m³)
    → řádek "Spalné teplo" — hodnota v MJ/m³ (typicky 30–38 MJ/m³)
    → POZOR: NE kWh/m³! Pokud je v kWh, přepočítej: MJ/m³ = kWh/m³ × 3,6
- conversion_factor: number (bezrozměrné, obvykle 1,0–1,05)
    → řádek "Koeficient přepočtu" / "Přepočtový koeficient"
- period_months: integer
    → počet měsíců fakturačního období (např. 12 pro roční, 1 pro měsíční)

── KOMODITNÍ SLOŽKA ─────────────────────────────────────────────────────────
- commodity_unit_price: number (CZK/MWh)
    → řádek "Komoditní složka" / "Cena za plyn" / "Silový plyn" — sloupec CENA/J nebo JEDN. CENA
- commodity_total_price: number (CZK)
    → stejný řádek — sloupec CELKEM

── DISTRIBUČNÍ SLOŽKA ───────────────────────────────────────────────────────
- fixed_monthly_fee_unit_price: number (CZK/měs)
    → řádek "Stálý měsíční plat" / "Pevná měsíční platba" — sloupec CENA/J
- fixed_monthly_fee: number (CZK)
    → stejný řádek — sloupec CELKEM (= jedn. cena × počet měsíců)
- distribution_unit_price: number (CZK/MWh)
    → řádek "Distribuce plynu" / "Variabilní složka distribuce" — sloupec CENA/J
- distribution_fixed_price: number (CZK)
    → řádek "Pevná cena distribuce" / "Stálý plat distribuce" — sloupec CELKEM
- reserved_capacity_unit_price: number (CZK/m³/h)
    → řádek "Přistavená kapacita" / "Rezervovaná kapacita" — sloupec CENA/J
- reserved_capacity_price: number (CZK)
    → stejný řádek — sloupec CELKEM
- market_operator_price: number (CZK)
    → řádek "Činnost operátora trhu" / "OTE" / "OPZ" — sloupec CELKEM

── DAŇ A OSTATNÍ ─────────────────────────────────────────────────────────────
- natural_gas_tax_total: number (CZK)
    → řádek "Daň ze zemního plynu" / "Daň ZP" / "Energetická daň" — sloupec CELKEM

PŘÍKLAD plyn MO faktury:
  | Spotřeba plynu | m³ | 248 | | |
  | Spalné teplo | MJ/m³ | 34,18 | | |
  | Koeficient přepočtu | — | 1,015 | | |
  | Spotřeba v MWh | MWh | 2,421 | | |
  | Komoditní složka | Kč/MWh | 2,421 | 1 850,00 | 4 478,85 |
  | Stálý měsíční plat | Kč/měs | 1 | 162,00 | 162,00 |
  | Pevná cena distribuce | Kč | | | 320,00 |
  | Přistavená kapacita | Kč/m³/h | 4,2 | 38,50 | 161,70 |
  | Činnost OTE | Kč | | | 14,52 |
  | Daň ze zemního plynu | Kč | | | 72,63 |
  Základ daně: 5 209,70 Kč    DPH 21 %: 1 094,04 Kč    Celkem s DPH: 6 303,74 Kč

Správná extrakce:
  consumption_m3 = 248.0
  combustion_heat = 34.18
  conversion_factor = 1.015
  consumption_mwh = 2.421
  commodity_total_price = 4478.85
  fixed_monthly_fee = 162.0
  distribution_fixed_price = 320.0
  reserved_capacity_price = 161.70
  market_operator_price = 14.52
  natural_gas_tax_total = 72.63
  total_amount_ex_vat = 5209.70
  total_amount_inc_vat = 6303.74
  vat_rate = 21.0"""


GAS_VO_FIELDS = """PLYN VO (velkoodběr) — specifická pole:

── SPOTŘEBA A PŘEPOČET ───────────────────────────────────────────────────────
- consumption_m3: number (m³)
    → řádek "Spotřeba" / "Objem plynu" — hodnota v m³
    → V letním měsíci může být 0 m³ — extrahuj 0.0, ne null
- consumption_mwh: number (MWh)
    → řádek "Energie" / "Spotřeba v MWh" — hodnota v MWh
- combustion_heat: number (MJ/m³)
    → řádek "Spalné teplo" — hodnota v MJ/m³
- conversion_factor: number
    → řádek "Koeficient přepočtu" / "Přepočtový koeficient"
- daily_reserved_capacity: number (m³/h)
    → řádek "Denní přistavená kapacita" / "Rezervovaná kapacita" / "Technická kapacita" — hodnota v m³/h

── OBCHODNÍ SLOŽKA (DODÁVKA PLYNU) ──────────────────────────────────────────
- other_supply_services_price: number (CZK)
    → řádek "Ostatní služby dodávky" / "Ostatní obchodní služby" — sloupec CELKEM
- trade_reserved_capacity_unit_price: number (CZK/m³/h/den nebo CZK/MWh)
    → řádek "Obchodní rezervovaná kapacita" / "Kapacita obchod" — sloupec CENA/J
- trade_reserved_capacity_price: number (CZK)
    → stejný řádek — sloupec CELKEM

── DISTRIBUČNÍ SLOŽKA ───────────────────────────────────────────────────────
- distribution_service_price: number (CZK)
    → řádek "Cena za službu distribuce" / "Přeprava plynu" / "Přenos" — sloupec CELKEM
- distribution_system_unit_price: number (CZK/MWh)
    → řádek "Distribuce soustavy" / "Variabilní složka distribuce" — sloupec CENA/J
- distribution_reserved_capacity_unit_price: number (CZK/m³/h/den)
    → řádek "Distribuční rezervovaná kapacita" / "Kapacita distribuce" — sloupec CENA/J
- distribution_reserved_capacity_price: number (CZK)
    → stejný řádek — sloupec CELKEM
- market_operator_price: number (CZK)
    → řádek "Činnost OTE" / "Operátor trhu" / "OPZ" — sloupec CELKEM

── DAŇ ──────────────────────────────────────────────────────────────────────
- natural_gas_tax_total: number (CZK)
    → řádek "Daň ze zemního plynu" / "Daň ZP" / "Energetická daň" — sloupec CELKEM

PŘÍKLAD plyn VO faktury:
  | Spotřeba plynu | m³ | 4 820 | | |
  | Spalné teplo | MJ/m³ | 33,85 | | |
  | Spotřeba v MWh | MWh | 45,35 | | |
  | Denní přistavená kapacita | m³/h | 680 | | |
  | Obchodní rezervovaná kapacita | Kč/m³/h/den | 680 | 0,42 | 8 618,40 |
  | Cena za službu distribuce | Kč | | | 12 450,00 |
  | Distribuce soustavy | Kč/MWh | 45,35 | 85,20 | 3 863,82 |
  | Distribuční rezervovaná kapacita | Kč/m³/h/den | 680 | 1,15 | 23 598,00 |
  | Ostatní služby dodávky | Kč | | | 1 200,00 |
  | Činnost OTE | Kč | | | 272,10 |
  | Daň ze zemního plynu | Kč | | | 1 361,00 |

Správná extrakce:
  consumption_m3 = 4820.0
  combustion_heat = 33.85
  consumption_mwh = 45.35
  daily_reserved_capacity = 680.0
  trade_reserved_capacity_price = 8618.40
  distribution_service_price = 12450.0
  distribution_system_unit_price = 85.2
  distribution_reserved_capacity_price = 23598.0
  other_supply_services_price = 1200.0
  market_operator_price = 272.10
  natural_gas_tax_total = 1361.0"""


WATER_FIELDS = """VODA — specifická pole:

- consumption_point_code: string
    → číslo odběrného místa / číslo zákazníka / číslo smlouvy (libovolný formát)
- consumption_m3: number (m³)
    → "Spotřeba vody" / "Naměřená spotřeba" / "Fakturované množství" — hodnota v m³
- water_rate: number (CZK)
    → "Vodné" / "Cena za dodávku pitné vody" / "Vodné celkem" — CELKOVÁ ČÁSTKA v CZK
    → POZOR: hledej celkovou ČÁSTKU (CZK), ne jednotkovou cenu (CZK/m³)!
    → Výpočet: jednotková cena (Kč/m³) × spotřeba (m³) = vodné (CZK)
- sewage_rate: number (CZK)
    → "Stočné" / "Cena za odkanalizování" / "Stočné celkem" — CELKOVÁ ČÁSTKA v CZK
    → POZOR: hledej celkovou ČÁSTKU (CZK), ne jednotkovou cenu (CZK/m³)!
- precipitation_water: number (CZK)
    → "Srážkové vody" / "Odvádění srážkových vod" — celková částka (nemusí být)
- wastewater_charge: number (CZK)
    → "Odpadní vody" / "Čištění odpadních vod" — celková částka (nemusí být)

UPOZORNĚNÍ: total_amount_ex_vat a total_amount_inc_vat jsou SOUČTY všech složek (vodné + stočné + případně srážkové).

PŘÍKLAD voda faktury:
  Fakturované množství: 42 m³
  | Vodné | m³ | 42 | 46,28 | 1 943,76 |
  | Stočné | m³ | 42 | 39,15 | 1 644,30 |
  Základ daně (10 %): 3 588,06 Kč    Celkem s DPH: 3 946,87 Kč

Správná extrakce:
  consumption_m3 = 42.0
  water_rate = 1943.76        ← CELKOVÁ ČÁSTKA za vodné (CZK), ne 46,28 Kč/m³!
  sewage_rate = 1644.30       ← CELKOVÁ ČÁSTKA za stočné (CZK), ne 39,15 Kč/m³!
  total_amount_ex_vat = 3588.06
  total_amount_inc_vat = 3946.87
  vat_rate = 10.0"""


HEAT_FIELDS = """TEPLO (centrální zásobování teplem) — specifická pole:

- consumption_point_code: string
    → číslo odběrného místa / číslo zákazníka / číslo smlouvy

── SPOTŘEBA ──────────────────────────────────────────────────────────────────
- consumption_gj: number (GJ)
    → "Spotřeba tepla celkem" / "Celková dodávka tepla" / "Spotřeba GJ" — celková hodnota v GJ
    → Pokud jsou na faktuře obě složky (vytápění + ohřev TV), toto je jejich SOUČET
- heat_consumption: number (GJ)
    → "Spotřeba tepla" / "Teplo na vytápění" / "Vytápění" — POUZE spotřeba pro vytápění (bez TV)
    → Pokud faktura nerozlišuje typy spotřeby, použij tuto hodnotu jako celkovou spotřebu
- hot_water_heating: number (GJ)
    → "Ohřev teplé vody" / "Příprava TV" / "Teplo na přípravu TV" / "Teplá voda (GJ)"
    → POUZE energie pro ohřev teplé vody
- total_heat_consumption: number (GJ)
    → "Celková spotřeba tepla" — pokud je explicitně uveden součet, dej ho sem
    → Pokud chybí, nekalkul — vrať null
- cold_water: number (m³)
    → "Studená voda" / "Objem studené vody" — spotřeba studené vody pro ohřev TV v m³

── KAPACITA A OSTATNÍ ────────────────────────────────────────────────────────
- reserved_capacity: number (kW)
    → "Rezervovaná kapacita" / "Smluvní výkon" / "Instalovaný výkon" — hodnota v kW
- supplementary_water: number (m³)
    → "Doplňovací voda" / "Náhradní voda" / "Voda pro topný systém" — hodnota v m³

── SLOŽKY CENY ───────────────────────────────────────────────────────────────
- fixed_monthly_fee: number (CZK)
    → "Stálý měsíční plat" / "Pevná složka" / "Paušál" — celková částka (CZK)
- variable_charge: number (CZK)
    → "Variabilní složka" / "Cena za teplo" / "Spotřební složka" — celková variabilní složka (CZK)
    → Je to SOUČET všech variabilních položek (vytápění + TV + případně voda)

PŘÍKLAD teplo faktury:
  | Spotřeba tepla (vytápění) | GJ | 18,42 | 285,40 | 5 254,97 |
  | Ohřev teplé vody | GJ | 4,15 | 285,40 | 1 184,41 |
  | Studená voda | m³ | 3,20 | 85,00 | 272,00 |
  | Stálý měsíční plat | Kč/měs | 1 | 420,00 | 420,00 |
  | Variabilní složka celkem | Kč | | | 6 711,38 |
  Základ daně (12 %): 7 131,38 Kč    Celkem s DPH: 7 987,15 Kč

Správná extrakce:
  heat_consumption = 18.42        ← jen vytápění
  hot_water_heating = 4.15        ← jen ohřev TV
  consumption_gj = 22.57          ← celkem GJ (18,42 + 4,15)
  cold_water = 3.20
  fixed_monthly_fee = 420.0
  variable_charge = 6711.38       ← variabilní celkem (CZK)
  total_amount_ex_vat = 7131.38
  total_amount_inc_vat = 7987.15
  vat_rate = 12.0"""


COMMODITY_FIELD_PROMPTS = {
    CommodityType.ELEKTRINA_NN: ELECTRICITY_NN_FIELDS,
    CommodityType.ELEKTRINA_VN: ELECTRICITY_VN_FIELDS,
    CommodityType.PLYN_MO: GAS_MO_FIELDS,
    CommodityType.PLYN_VO: GAS_VO_FIELDS,
    CommodityType.VODA: WATER_FIELDS,
    CommodityType.TEPLO: HEAT_FIELDS,
}


# ════════════════════════════════════════════════════════════════════════════
# PYDANTIC OUTPUT MODELS FOR STRUCTURED LLM OUTPUT
# ════════════════════════════════════════════════════════════════════════════


class LLMExtractedInvoice(BaseModel):
    """Structured output model for LLM extraction.

    Flat model covering all commodity-specific fields. Only the fields
    relevant to the detected commodity will be populated. Converted to
    InvoiceData by _convert_to_invoice_data().
    """

    # ── Universal fields ────────────────────────────────────────────────────
    invoice_number: str | None = Field(None, description="Číslo faktury / číslo dokladu — the invoice identifier string, e.g. '2023100123456'")
    variable_symbol: str | None = Field(None, description="Variabilní symbol (VS) — payment reference number, often same as invoice_number")
    consumption_point_code: str | None = Field(None, description="Číslo odběrného místa — EAN (electricity, 18 digits starting 859), EIC (gas, 16 chars starting 27ZG), or any provider-specific OM code")
    ean_code: str | None = Field(None, description="EAN supply point code for electricity (18 digits starting with 859)")
    eic_code: str | None = Field(None, description="EIC supply point code for gas (16 alphanumeric chars starting with 27ZG)")

    period_from: str | None = Field(None, description="Fakturační období OD — billing period start date, format DD.MM.YYYY")
    period_to: str | None = Field(None, description="Fakturační období DO — billing period end date, format DD.MM.YYYY")
    issue_date: str | None = Field(None, description="Datum vystavení — invoice issue date (always AFTER period_to), format DD.MM.YYYY")
    due_date: str | None = Field(None, description="Datum splatnosti — payment due date, format DD.MM.YYYY")
    tax_point_date: str | None = Field(None, description="DUZP — datum uskutečnění zdanitelného plnění (tax point date), format DD.MM.YYYY; often equals period_to or last day of billing month")

    customer_tax_id: str | None = Field(None, description="IČO odběratele — buyer's Czech tax ID, exactly 8 digits zero-padded (e.g. '01234567')")
    customer_vat_id: str | None = Field(None, description="DIČ odběratele — buyer's VAT ID (e.g. 'CZ01234567')")
    supplier_tax_id: str | None = Field(None, description="IČO dodavatele — supplier's Czech tax ID, exactly 8 digits zero-padded")
    supplier_vat_id: str | None = Field(None, description="DIČ dodavatele — supplier's VAT ID")

    total_amount_inc_vat: float | None = Field(None, description="Celková cena S DPH — 'Cena celkem s DPH' / 'Celkem vč. DPH' from the final DPH summary table. NEVER use 'K úhradě' (amount after advances deducted).")
    total_amount_ex_vat: float | None = Field(None, description="Celková cena BEZ DPH — 'Základ daně' / 'Cena celkem bez DPH' / 'Celkem bez DPH' from the final DPH summary table.")
    vat_rate: float | None = Field(None, description="Sazba DPH v % — VAT rate as a number: 21.0, 12.0, 10.0, or 0.0. For multiple rates use the rate on the main commodity line.")
    advance_payment: float | None = Field(None, description="Uhrazené zálohy — total prepayments already paid (CZK); used only for information, NOT for total amounts")
    amount_to_pay: float | None = Field(None, description="K úhradě / Doplatek — amount remaining to pay after subtracting advances (CZK)")

    # ── Elektřina NN ────────────────────────────────────────────────────────
    consumption_low_tariff: float | None = Field(None, description="Spotřeba NT — low-tariff consumption in kWh (Nízký tarif). Null for single-rate tariffs (D01d, D02d).")
    consumption_high_tariff: float | None = Field(None, description="Spotřeba VT — high-tariff consumption in kWh (Vysoký tarif). For single-rate tariffs (D01d, D02d) this holds the total consumption.")
    total_consumption: float | None = Field(None, description="Celková spotřeba — total electricity consumption in kWh (NT + VT combined). Null if not explicitly stated.")

    # ── Elektřina VN ────────────────────────────────────────────────────────
    supply_consumption: float | None = Field(None, description="Spotřeba SE — electricity supply consumption in MWh; row 'Spotřeba elektřiny' or 'Odběr SE', MNOŽSTVÍ column")
    supply_charge: float | None = Field(None, description="Částka SE — charge for supply electricity in CZK; row 'Silová elektřina' or 'Komodita SE', CELKEM column")
    supply_tax_charge: float | None = Field(None, description="Daň SE — energy tax on supply electricity in CZK; row 'Daň ze silové elektřiny' or 'Daň SE', CELKEM column")
    quarter_hour_max: float | None = Field(None, description="Čtvrthodinové maximum — peak quarter-hour demand in kW or MW; row 'Čtvrthodinové maximum' or 'ČHM', HODNOTA column")
    eru_rate: float | None = Field(None, description="Sazba ERÚ — ERÚ-regulated rate (CZK/kW); row 'Sazba ERÚ' or 'Regulovaná složka ERÚ'")
    annual_reserved_capacity: float | None = Field(None, description="Roční rezervovaná kapacita (RK roční) in MW; row 'Rezervovaná kapacita roční' or 'RK roční', MNOŽSTVÍ column")
    annual_reserved_capacity_charge: float | None = Field(None, description="Částka RK roční — annual reserved capacity charge in CZK; same row as annual_reserved_capacity, CELKEM column")
    monthly_reserved_capacity: float | None = Field(None, description="Měsíční rezervovaná kapacita (RK měsíční) in MW or kW; row 'Rezervovaná kapacita' or 'RK měsíční', MNOŽSTVÍ column")
    monthly_reserved_capacity_charge: float | None = Field(None, description="Částka RK měsíční — monthly reserved capacity charge in CZK; same row as monthly_reserved_capacity, CELKEM column")
    grid_usage_rate: float | None = Field(None, description="Sazba použití sítí — grid usage unit rate in CZK/MWh; row 'Přenos elektřiny' or 'Použití přenosové soustavy', CENA/J column")
    grid_usage_charge: float | None = Field(None, description="Částka použití sítí — total grid usage charge in CZK; row 'Přenos elektřiny' or 'Přenos VN', CELKEM column")
    reserved_capacity_excess: float | None = Field(None, description="Překročení RK — reserved capacity excess quantity in MW; row 'Překročení rezervované kapacity', MNOŽSTVÍ column. Null or 0 if no excess occurred.")
    reserved_capacity_excess_rate: float | None = Field(None, description="Sazba překročení RK — excess capacity rate in CZK/MW; same row as reserved_capacity_excess, CENA/J column")
    reserved_capacity_excess_charge: float | None = Field(None, description="Částka překročení RK — excess capacity charge in CZK; same row, CELKEM column. Null or 0 if no excess.")
    power_factor: float | None = Field(None, description="tg φ — power factor (dimensionless ratio, typically 0.0–0.9); row 'tg φ' or 'cos φ' or 'Účiník', HODNOTA column")
    reactive_power_quantity: float | None = Field(None, description="Množství jalové energie — reactive power quantity in kVArh; row 'Jalová energie' or 'Kompenzace jalové energie', MNOŽSTVÍ column")
    reactive_power_rate: float | None = Field(None, description="Sazba jalové energie — reactive power rate in CZK/kVArh; same row, CENA/J column")
    reactive_power_charge: float | None = Field(None, description="Částka jalová — reactive power charge in CZK; row 'Jalová energie' or 'Kompenzace jalové energie', CELKEM column")
    service_price: float | None = Field(None, description="Cena služby — system services charge in CZK; row 'Cena služby' or 'Systémové služby', CELKEM column")
    operating_price: float | None = Field(None, description="Cena provozu sítě — grid operating charge in CZK; row 'Cena za provoz' or 'Provozní složka', CELKEM column")
    renewable_energy_fee: float | None = Field(None, description="POZE — renewable energy support fee in CZK; row 'POZE' or 'Podpora obnovitelných zdrojů' or 'OZE+KVET', CELKEM column")

    # ── Plyn MO + VO — shared ───────────────────────────────────────────────
    consumption_m3: float | None = Field(None, description="Spotřeba plynu v m³ — gas consumption in cubic metres; row 'Spotřeba' or 'Objem plynu'. Can be 0.0 in summer months — extract 0.0, not null.")
    consumption_mwh: float | None = Field(None, description="Spotřeba plynu v MWh — gas consumption in MWh; row 'Energie' or 'Spotřeba v MWh'")
    conversion_factor: float | None = Field(None, description="Koeficient přepočtu — gas volume conversion factor (dimensionless, typically 1.0–1.05); row 'Koeficient přepočtu'")
    combustion_heat: float | None = Field(None, description="Spalné teplo — calorific value in MJ/m³ (typically 30–38 MJ/m³); row 'Spalné teplo'. NOTE: unit must be MJ/m³, not kWh/m³.")
    market_operator_price: float | None = Field(None, description="Cena za činnost operátora trhu OTE/OPZ — market operator fee in CZK; row 'Činnost OTE' or 'Operátor trhu', CELKEM column")
    natural_gas_tax_total: float | None = Field(None, description="Daň ze zemního plynu celkem — natural gas energy tax in CZK; row 'Daň ze zemního plynu' or 'Daň ZP' or 'Energetická daň', CELKEM column")

    # ── Plyn MO ─────────────────────────────────────────────────────────────
    period_months: int | None = Field(None, description="Počet měsíců v fakturačním období — billing period length in months (e.g. 1, 3, 12)")
    commodity_unit_price: float | None = Field(None, description="Jednotková cena komoditní složky — gas commodity unit price in CZK/MWh; row 'Komoditní složka' or 'Silový plyn', CENA/J column")
    commodity_total_price: float | None = Field(None, description="Cena za komoditní složku — total commodity charge in CZK; same row as commodity_unit_price, CELKEM column")
    fixed_monthly_fee_unit_price: float | None = Field(None, description="Jednotková cena stálého měsíčního platu — fixed monthly fee unit price in CZK/month; row 'Stálý měsíční plat', CENA/J column")
    fixed_monthly_fee: float | None = Field(None, description="Stálý měsíční plat — total fixed monthly fee for the period in CZK; row 'Stálý měsíční plat', CELKEM column")
    distribution_unit_price: float | None = Field(None, description="Jednotková cena distribuce plynu — gas distribution variable unit price in CZK/MWh; row 'Distribuce plynu' or 'Variabilní složka distribuce', CENA/J column")
    distribution_fixed_price: float | None = Field(None, description="Pevná cena distribuce — fixed distribution charge in CZK; row 'Pevná cena za distribuci' or 'Stálý plat distribuce', CELKEM column")
    reserved_capacity_unit_price: float | None = Field(None, description="Jednotková cena přistavené kapacity — reserved capacity unit price in CZK/m³/h; row 'Přistavená kapacita' or 'Rezervovaná kapacita', CENA/J column")
    reserved_capacity_price: float | None = Field(None, description="Cena za přistavenou kapacitu — total reserved capacity charge in CZK; same row, CELKEM column")

    # ── Plyn VO ─────────────────────────────────────────────────────────────
    daily_reserved_capacity: float | None = Field(None, description="Denní přistavená kapacita — daily reserved gas capacity in m³/h; row 'Denní přistavená kapacita' or 'Technická kapacita', MNOŽSTVÍ column")
    other_supply_services_price: float | None = Field(None, description="Ostatní služby dodávky — other supply service charges in CZK; row 'Ostatní služby dodávky', CELKEM column")
    trade_reserved_capacity_unit_price: float | None = Field(None, description="Jednotková cena obchodní rezervované kapacity — trade reserved capacity unit price in CZK/m³/h/day; CENA/J column")
    trade_reserved_capacity_price: float | None = Field(None, description="Cena za obchodní rezervovanou kapacitu — total trade reserved capacity charge in CZK; CELKEM column")
    distribution_service_price: float | None = Field(None, description="Cena za službu distribuce — total gas distribution service charge in CZK; row 'Cena za službu distribuce' or 'Přeprava plynu', CELKEM column")
    distribution_system_unit_price: float | None = Field(None, description="Jednotková cena distribuce soustavy — distribution system variable unit price in CZK/MWh; CENA/J column")
    distribution_reserved_capacity_unit_price: float | None = Field(None, description="Jednotková cena distribuční rezervované kapacity — distribution reserved capacity unit price in CZK/m³/h/day; CENA/J column")
    distribution_reserved_capacity_price: float | None = Field(None, description="Cena distribuční rezervované kapacity — total distribution reserved capacity charge in CZK; CELKEM column")

    # ── Voda ────────────────────────────────────────────────────────────────
    water_rate: float | None = Field(None, description="Vodné — total charge for water supply in CZK (NOT unit price per m³). Row 'Vodné' or 'Cena za dodávku pitné vody', CELKEM column.")
    sewage_rate: float | None = Field(None, description="Stočné — total charge for sewage/wastewater removal in CZK (NOT unit price per m³). Row 'Stočné' or 'Cena za odkanalizování', CELKEM column.")
    precipitation_water: float | None = Field(None, description="Srážkové vody — total charge for stormwater drainage in CZK; row 'Srážkové vody', CELKEM column. Null if not on invoice.")
    wastewater_charge: float | None = Field(None, description="Odpadní vody — total charge for wastewater treatment in CZK; row 'Odpadní vody' or 'Čištění odpadních vod', CELKEM column.")

    # ── Teplo ────────────────────────────────────────────────────────────────
    consumption_gj: float | None = Field(None, description="Celková spotřeba tepla v GJ — total heat consumption in GJ (heating + hot water combined). Row 'Spotřeba tepla celkem' or 'Celková dodávka tepla'.")
    heat_consumption: float | None = Field(None, description="Spotřeba tepla pro vytápění v GJ — space heating consumption only (NOT hot water prep). Row 'Spotřeba tepla' or 'Vytápění'. If invoice does not split types, use this for total.")
    hot_water_heating: float | None = Field(None, description="Ohřev teplé vody v GJ — energy for hot water preparation in GJ. Row 'Ohřev teplé vody' or 'Příprava TV' or 'Teplá voda (GJ)'.")
    cold_water: float | None = Field(None, description="Studená voda v m³ — cold water volume for hot water preparation. Row 'Studená voda' or 'Objem studené vody'.")
    total_heat_consumption: float | None = Field(None, description="Celková spotřeba tepla (explicit label) in GJ — use only if invoice explicitly labels this as a total. Do not compute.")
    reserved_capacity: float | None = Field(None, description="Rezervovaná kapacita tepla v kW — contracted heat power capacity. Row 'Rezervovaná kapacita' or 'Smluvní výkon'.")
    supplementary_water: float | None = Field(None, description="Doplňovací voda v m³ — supplementary water for heating system. Row 'Doplňovací voda' or 'Náhradní voda'.")
    variable_charge: float | None = Field(None, description="Variabilní složka celkem v CZK — total variable charge for heat (sum of all consumption-based charges). Row 'Variabilní složka' or 'Spotřební složka'.")


# ════════════════════════════════════════════════════════════════════════════
# LANGCHAIN EXTRACTION STRATEGY WITH LCEL
# ════════════════════════════════════════════════════════════════════════════


class LangChainExtractionStrategy(BaseExtractionStrategy):
    """LLM-based extraction using LangChain Expression Language (LCEL).

    Supports multiple LLM providers (OpenAI, Anthropic) via LangChain.
    Uses PydanticOutputParser for structured output parsing.

    LCEL chain: prompt_template | llm | output_parser
    """

    def __init__(
        self,
        model_provider: Literal["openai", "anthropic"] = "openai",
        model_name: str | None = None,
        temperature: float = 0.0,
        use_vision: bool = False,
        retriever=None,
    ) -> None:
        """Initialize the LangChain extraction strategy.

        Args:
            model_provider: LLM provider ('openai' or 'anthropic').
            model_name: Specific model name (e.g., 'gpt-4o', 'claude-3-sonnet').
            temperature: Model temperature (0.0 for deterministic).
            use_vision: If True, use vision models for image input.
            retriever: Optional FewShotRetriever. When provided, similar
                       correctly-extracted invoice examples are injected into
                       the prompt (few-shot RAG n=2 — the validated primary
                       strategy from NB05 / the DS1 final pipeline).
        """
        self._provider = model_provider
        self._model_name = model_name
        self._temperature = temperature
        self._use_vision = use_vision
        self._retriever = retriever
        self._llm: BaseChatModel | None = None
        self._chain: Any = None

    @property
    def name(self) -> str:
        """Return strategy name identifier."""
        suffix = "_vision" if self._use_vision else ""
        model = self._model_name or self._get_default_model_name()
        return f"langchain_{self._provider}_{model}{suffix}"

    @property
    def model_name(self) -> str:
        """Return the LLM model name used for extraction."""
        return self._model_name or self._get_default_model_name()

    def _get_default_model_name(self) -> str:
        """Get default model name for provider.

        The OpenAI vision fallback defaults to ``gpt-4.1-mini`` — the validated
        best Vision-LLM configuration from the NB07 benchmark (F1=0.733,
        gpt-4.1-mini + grayscale/denoise), which also outperformed plain
        gpt-4.1 (F1=0.687) at a fraction of the cost.
        """
        if self._provider == "openai":
            return "gpt-4.1-mini"
        elif self._provider == "anthropic":
            return "claude-3-5-sonnet-20241022" if self._use_vision else "claude-3-haiku-20240307"
        return "gpt-4.1-mini"

    def _initialize_llm(self) -> BaseChatModel:
        """Initialize the LangChain LLM client.

        Returns:
            Configured LangChain chat model.

        Raises:
            ImportError: If required LangChain packages are not installed.
            ValueError: If API keys are not configured.
        """
        if self._llm is not None:
            return self._llm

        model_name = self._model_name or self._get_default_model_name()

        if self._provider == "openai":
            try:
                from langchain_openai import ChatOpenAI

                from src.config.settings import get_settings

                settings = get_settings()
                if not settings.openai_api_key:
                    raise ValueError("OPENAI_API_KEY not configured in settings")

                self._llm = ChatOpenAI(
                    model=model_name,
                    temperature=self._temperature,
                    api_key=settings.openai_api_key,
                )
            except ImportError as e:
                raise ImportError(
                    "langchain-openai not installed. "
                    "Install with: pip install langchain-openai"
                ) from e

        elif self._provider == "anthropic":
            try:
                from langchain_anthropic import ChatAnthropic

                from src.config.settings import get_settings

                settings = get_settings()
                if not settings.anthropic_api_key:
                    raise ValueError("ANTHROPIC_API_KEY not configured in settings")

                self._llm = ChatAnthropic(
                    model=model_name,
                    temperature=self._temperature,
                    api_key=settings.anthropic_api_key,
                )
            except ImportError as e:
                raise ImportError(
                    "langchain-anthropic not installed. "
                    "Install with: pip install langchain-anthropic"
                ) from e

        else:
            raise ValueError(f"Unknown model provider: {self._provider}")

        return self._llm

    def _build_lcel_chain(self, commodity: CommodityType, few_shot: str = "") -> Any:
        """Build LCEL chain: prompt | llm | parser.

        Args:
            commodity: Detected commodity type for specialized prompt.
            few_shot: Optional few-shot examples block (from the retriever) to
                      inject before the invoice text.

        Returns:
            Tuple of (prompt_chain, parser) where prompt_chain = prompt | llm.
        """
        from langchain_core.output_parsers import PydanticOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        llm = self._initialize_llm()
        parser = PydanticOutputParser(pydantic_object=LLMExtractedInvoice)

        commodity_fields = COMMODITY_FIELD_PROMPTS.get(commodity, "")

        # Build prompt with format instructions
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT + "\n\n{format_instructions}"),
            ("human", """Extract the following data from this {commodity_type} invoice:

Required fields:
- invoice_number: string
- variable_symbol: string (optional)
- consumption_point_code: string (odběrné místo / EAN / EIC / any supply point code — NEVER invoice number, optional)
- ean_code: string (13-18 digit EAN supply point identifier starting 859, optional)
- eic_code: string (starts with 27ZG, optional)
- period_from: date (DD.MM.YYYY, billing period start)
- period_to: date (DD.MM.YYYY, billing period end)
- issue_date: date (DD.MM.YYYY, "Datum vystavení" — AFTER period_to, not the period end date itself)
- due_date: date (DD.MM.YYYY, datum splatnosti, optional)
- tax_point_date: date (DD.MM.YYYY, DUZP / datum uskutečnění zdanitelného plnění, optional)
- customer_tax_id: string (8 digits, IČO of Odběratel/buyer — look under "Odběratel" section, NOT supplier IČO, optional)
- supplier_tax_id: string (8 digits, IČO of Dodavatel/supplier — look under "Dodavatel" section, optional)
- total_amount_inc_vat: number (CZK, "Cena celkem s DPH" or "Celkem včetně DPH" from FINAL SUMMARY TABLE — NEVER K úhradě)
- total_amount_ex_vat: number (CZK, "Základ daně" or "Cena celkem bez DPH" from FINAL SUMMARY TABLE — NEVER K úhradě, optional)
- vat_rate: number (%, sazba DPH, e.g. 21.0, optional)

{commodity_specific_fields}

{few_shot_examples}
Invoice text:
```
{invoice_text}
```

Return a valid JSON object with the extracted data. Use null for missing fields."""),
        ])

        # Partial prompt with non-variable parts
        partial_prompt = prompt.partial(
            format_instructions=parser.get_format_instructions(),
            commodity_specific_fields=commodity_fields,
            few_shot_examples=few_shot,
        )

        # Build LCEL chains: prompt -> llm (for raw response), and parser separately
        prompt_chain = partial_prompt | llm

        return prompt_chain, parser

    def _convert_to_invoice_data(
        self,
        extracted: LLMExtractedInvoice,
        commodity: CommodityType,
        source_filename: str,
    ) -> InvoiceData:
        """Convert LLM extracted data to InvoiceData model.

        Args:
            extracted: LLM extraction result.
            commodity: Detected commodity type.
            source_filename: Original source file.

        Returns:
            Fully populated InvoiceData.
        """
        # Parse dates
        period_from = parse_czech_date(extracted.period_from) if extracted.period_from else None
        period_to = parse_czech_date(extracted.period_to) if extracted.period_to else None
        issue_date_parsed = parse_czech_date(extracted.issue_date) if extracted.issue_date else None
        due_date_parsed = parse_czech_date(extracted.due_date) if extracted.due_date else None

        # If period dates are missing, leave them as None and create a
        # best-effort period. BillingPeriod requires both dates, so we
        # skip it when either is absent (the invoice will be saved with
        # NULL period_from/period_to, which the DB allows).
        period: BillingPeriod | None = None
        if period_from and period_to:
            if period_from > period_to:
                period_from, period_to = period_to, period_from
            period = BillingPeriod(
                period_from=period_from,
                period_to=period_to,
            )
        elif period_from or period_to:
            # Only one date available — use it for both
            single = period_from or period_to
            period = BillingPeriod(period_from=single, period_to=single)

        # Create supply point — prefer explicit consumption_point_code, then EAN/EIC
        supply_point = SupplyPoint(
            consumption_point_code=extracted.consumption_point_code or extracted.ean_code or extracted.eic_code or "",
            ean_code=extracted.ean_code or "",
            eic_code=extracted.eic_code or "",
        )

        invoice_type = InvoiceType.REGULAR

        vat_date_parsed = (
            parse_czech_date(extracted.tax_point_date)
            if extracted.tax_point_date else None
        )

        # Build base invoice
        invoice = InvoiceData(
            id=uuid4(),
            source_filename=source_filename,
            invoice_number=extracted.invoice_number or "UNKNOWN",
            variable_symbol=extracted.variable_symbol,
            commodity=commodity,
            invoice_type=invoice_type,
            supply_point=supply_point,
            period=period,
            issue_date=issue_date_parsed,
            due_date=due_date_parsed,
            vat_date=vat_date_parsed,
            customer_tax_id=extracted.customer_tax_id,
            customer_vat_id=extracted.customer_vat_id,
            supplier_tax_id=extracted.supplier_tax_id,
            supplier_vat_id=extracted.supplier_vat_id,
            total_amount_ex_vat=extracted.total_amount_ex_vat,
            total_amount_inc_vat=extracted.total_amount_inc_vat,
            vat_rate=extracted.vat_rate,
            advance_payment=extracted.advance_payment,
            amount_to_pay=extracted.amount_to_pay,
        )

        # Add commodity-specific details
        self._add_commodity_details(invoice, extracted, commodity, period)

        return invoice

    def _add_commodity_details(
        self,
        invoice: InvoiceData,
        extracted: LLMExtractedInvoice,
        commodity: CommodityType,
        period: BillingPeriod,
    ) -> None:
        """Add commodity-specific detail records to invoice.

        Args:
            invoice: Invoice to populate.
            extracted: LLM extraction result.
            commodity: Detected commodity type.
            period: Billing period for details.
        """
        if commodity == CommodityType.ELEKTRINA_NN:
            detail = ElectricityNNData(
                period=period,
                consumption_low_tariff=extracted.consumption_low_tariff,
                consumption_high_tariff=extracted.consumption_high_tariff,
                total_consumption=extracted.total_consumption,
            )
            invoice.electricity_nn_details.append(detail)

        elif commodity == CommodityType.ELEKTRINA_VN:
            detail = ElectricityVNData(
                period=period,
                supply_consumption=extracted.supply_consumption,
                supply_charge=extracted.supply_charge,
                supply_tax_charge=extracted.supply_tax_charge,
                quarter_hour_max=extracted.quarter_hour_max,
                eru_rate=extracted.eru_rate,
                annual_reserved_capacity=extracted.annual_reserved_capacity,
                annual_reserved_capacity_charge=extracted.annual_reserved_capacity_charge,
                monthly_reserved_capacity=extracted.monthly_reserved_capacity,
                monthly_reserved_capacity_charge=extracted.monthly_reserved_capacity_charge,
                grid_usage_rate=extracted.grid_usage_rate,
                grid_usage_charge=extracted.grid_usage_charge,
                reserved_capacity_excess=extracted.reserved_capacity_excess,
                reserved_capacity_excess_rate=extracted.reserved_capacity_excess_rate,
                reserved_capacity_excess_charge=extracted.reserved_capacity_excess_charge,
                power_factor=extracted.power_factor,
                reactive_power_quantity=extracted.reactive_power_quantity,
                reactive_power_rate=extracted.reactive_power_rate,
                reactive_power_charge=extracted.reactive_power_charge,
                service_price=extracted.service_price,
                operating_price=extracted.operating_price,
                renewable_energy_fee=extracted.renewable_energy_fee,
            )
            invoice.electricity_vn_details.append(detail)

        elif commodity == CommodityType.PLYN_MO:
            detail = GasMOData(
                period=period,
                consumption_m3=extracted.consumption_m3,
                consumption_mwh=extracted.consumption_mwh,
                conversion_factor=extracted.conversion_factor,
                combustion_heat=extracted.combustion_heat,
                period_months=extracted.period_months,
                commodity_unit_price=extracted.commodity_unit_price,
                commodity_total_price=extracted.commodity_total_price,
                fixed_monthly_fee_unit_price=extracted.fixed_monthly_fee_unit_price,
                fixed_monthly_fee=extracted.fixed_monthly_fee,
                distribution_unit_price=extracted.distribution_unit_price,
                distribution_fixed_price=extracted.distribution_fixed_price,
                reserved_capacity_unit_price=extracted.reserved_capacity_unit_price,
                reserved_capacity_price=extracted.reserved_capacity_price,
                market_operator_price=extracted.market_operator_price,
                natural_gas_tax_total=extracted.natural_gas_tax_total,
            )
            invoice.gas_mo_details.append(detail)

        elif commodity == CommodityType.PLYN_VO:
            detail = GasVOData(
                period=period,
                consumption_m3=extracted.consumption_m3,
                consumption_mwh=extracted.consumption_mwh,
                conversion_factor=extracted.conversion_factor,
                combustion_heat=extracted.combustion_heat,
                daily_reserved_capacity=extracted.daily_reserved_capacity,
                other_supply_services_price=extracted.other_supply_services_price,
                trade_reserved_capacity_unit_price=extracted.trade_reserved_capacity_unit_price,
                trade_reserved_capacity_price=extracted.trade_reserved_capacity_price,
                distribution_service_price=extracted.distribution_service_price,
                distribution_system_unit_price=extracted.distribution_system_unit_price,
                distribution_reserved_capacity_unit_price=extracted.distribution_reserved_capacity_unit_price,
                distribution_reserved_capacity_price=extracted.distribution_reserved_capacity_price,
                market_operator_price=extracted.market_operator_price,
                natural_gas_tax_total=extracted.natural_gas_tax_total,
            )
            invoice.gas_vo_details.append(detail)

        elif commodity == CommodityType.VODA:
            detail = WaterData(
                period=period,
                consumption_m3=extracted.consumption_m3,
                water_rate=extracted.water_rate,
                sewage_rate=extracted.sewage_rate,
                precipitation_water=extracted.precipitation_water,
                wastewater_charge=extracted.wastewater_charge,
            )
            invoice.water_details.append(detail)

        elif commodity == CommodityType.TEPLO:
            detail = HeatData(
                period=period,
                consumption_gj=extracted.consumption_gj,
                heat_consumption=extracted.heat_consumption,
                hot_water_heating=extracted.hot_water_heating,
                cold_water=extracted.cold_water,
                total_heat_consumption=extracted.total_heat_consumption,
                reserved_capacity=extracted.reserved_capacity,
                supplementary_water=extracted.supplementary_water,
                fixed_monthly_fee=extracted.fixed_monthly_fee,
                variable_charge=extracted.variable_charge,
            )
            invoice.heat_details.append(detail)

    async def extract(self, context: ExtractionContext) -> ExtractionResult:
        """Extract structured data using LangChain LCEL.

        Args:
            context: ExtractionContext with raw text and metadata.

        Returns:
            ExtractionResult with parsed InvoiceData.
        """
        errors: list[str] = []
        warnings: list[str] = []
        start_time = time.time()

        try:
            # Detect commodity if not provided
            commodity = context.commodity_hint
            if not commodity:
                from src.core.extraction.regex_strategy import RegexExtractionStrategy
                regex = RegexExtractionStrategy()
                commodity = regex._detect_commodity(context.raw_text)

            if not commodity:
                errors.append("Could not detect commodity type from text")
                return ExtractionResult(
                    source_file=context.source_filename,
                    strategy_name=self.name,
                    raw_text=context.raw_text,
                    errors=errors,
                )

            # Few-shot RAG (n=2): inject similar correctly-extracted examples,
            # matching the validated primary strategy (NB05 / DS1 final pipeline).
            few_shot = ""
            if self._retriever is not None and context.raw_text:
                try:
                    few_shot = self._retriever.get_examples(
                        context.raw_text, n=2,
                        exclude_stem=Path(context.source_filename).stem
                        if context.source_filename else None,
                    )
                except Exception as fs_err:
                    logger.warning("Few-shot retrieval failed: %s", fs_err)

            # Build LCEL chain (split into prompt+llm and parser)
            prompt_chain, parser = self._build_lcel_chain(commodity, few_shot=few_shot)

            # Truncate text to avoid token limits (increased: price tables often appear late in docs)
            invoice_text = context.raw_text[:24000]

            # Invoke prompt+llm to get raw AIMessage with metadata
            logger.info(f"Invoking LangChain LCEL chain with {self._provider}")
            ai_message = await prompt_chain.ainvoke({
                "commodity_type": commodity.value,
                "invoice_text": invoice_text,
            })

            # Extract token usage from response metadata
            token_count = 0
            cost_usd = 0.0
            resp_meta = getattr(ai_message, "response_metadata", {}) or {}
            usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
            if usage:
                token_count = usage.get("total_tokens", 0)
                input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
                cost_usd = self._estimate_cost(input_tokens, output_tokens)

            # Parse the response content into structured data
            try:
                extracted: LLMExtractedInvoice = parser.parse(ai_message.content)
            except Exception as parse_err:
                logger.warning(
                    "LLM returned unparseable output, attempting JSON repair: %s",
                    str(parse_err)[:200],
                )
                # Try to extract JSON from the raw content
                raw_content = ai_message.content
                try:
                    # Find JSON object in the response
                    start = raw_content.index("{")
                    end = raw_content.rindex("}") + 1
                    json_str = raw_content[start:end]
                    extracted = LLMExtractedInvoice.model_validate_json(json_str)
                except (ValueError, Exception) as repair_err:
                    error_msg = (
                        f"Failed to parse LLM output: {str(parse_err)[:200]}; "
                        f"repair also failed: {str(repair_err)[:200]}"
                    )
                    errors.append(error_msg)
                    return ExtractionResult(
                        source_file=context.source_filename,
                        strategy_name=self.name,
                        raw_text=context.raw_text,
                        errors=errors,
                        token_count=token_count,
                        cost_usd=cost_usd,
                    )

            # Convert to InvoiceData
            invoice_data = self._convert_to_invoice_data(
                extracted,
                commodity,
                context.source_filename,
            )

            # Derive missing amount from DPH table
            invoice_data, _vat_derived = apply_vat_derivation_to_invoice(invoice_data)
            if _vat_derived:
                warnings.append("total_amount derived via DPH table (one amount was null)")

            # Fix 3: deterministic amount_inc_vat correction from the VAT rate
            invoice_data, _vat_corr = apply_vat_inc_correction_to_invoice(
                invoice_data, enabled=ENABLE_VAT_INC_CORRECTION
            )
            if _vat_corr:
                warnings.append(
                    "amount_inc_vat corrected via VAT rate "
                    f"{_vat_corr['vat_rate_used']}% "
                    f"({_vat_corr['extracted_value']} → {_vat_corr['computed_value']})"
                )

            # Validate
            validation_warnings = await self.validate(invoice_data)
            warnings.extend(validation_warnings)

            # Calculate confidence based on required fields
            confidence = self._calculate_confidence(invoice_data)

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"LangChain extraction completed in {elapsed_ms}ms, "
                f"confidence={confidence:.2f}, tokens={token_count}"
            )

            return ExtractionResult(
                source_file=context.source_filename,
                strategy_name=self.name,
                confidence=confidence,
                raw_text=context.raw_text,
                invoice_data=invoice_data,
                errors=errors,
                warnings=warnings,
                token_count=token_count,
                cost_usd=cost_usd,
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = f"LLM extraction failed after {elapsed_ms}ms: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            return ExtractionResult(
                source_file=context.source_filename,
                strategy_name=self.name,
                raw_text=context.raw_text,
                errors=errors,
            )

    def _calculate_confidence(self, invoice_data: InvoiceData) -> float:
        """Calculate extraction confidence based on required fields.

        Args:
            invoice_data: Extracted invoice data.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        score = 0.0
        max_score = 10.0

        # Core fields
        if invoice_data.invoice_number and invoice_data.invoice_number != "UNKNOWN":
            score += 2.0
        if (invoice_data.supply_point.consumption_point_code
                or invoice_data.supply_point.ean_code
                or invoice_data.supply_point.eic_code):
            score += 2.0
        if invoice_data.period and invoice_data.period.period_from and invoice_data.period.period_to:
            score += 2.0
        if invoice_data.issue_date:
            score += 1.0
        if invoice_data.total_amount_inc_vat:
            score += 2.0
        if invoice_data.customer_tax_id or invoice_data.supplier_tax_id:
            score += 1.0

        return min(score / max_score, 1.0)

    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate API cost in USD based on model pricing.

        Args:
            input_tokens: Number of input/prompt tokens.
            output_tokens: Number of output/completion tokens.

        Returns:
            Estimated cost in USD.
        """
        model = (self._model_name or self._get_default_model_name()).lower()

        # Pricing per 1M tokens (input, output)
        pricing = {
            "gpt-4.1-mini": (0.40, 1.60),
            "gpt-4.1-nano": (0.10, 0.40),
            "gpt-4.1": (2.00, 8.00),
            "gpt-4o-mini": (0.15, 0.60),
            "gpt-4o": (2.50, 10.00),
            "gpt-4-turbo": (10.00, 30.00),
            "claude-3-haiku": (0.25, 1.25),
            "claude-3-5-haiku": (0.80, 4.00),
            "claude-3-5-sonnet": (3.00, 15.00),
        }

        # Find matching pricing — check longer keys first to avoid "gpt-4.1" matching "gpt-4.1-mini"
        input_rate, output_rate = 0.40, 1.60  # default to gpt-4.1-mini
        for key in sorted(pricing, key=len, reverse=True):
            if key in model:
                input_rate, output_rate = pricing[key]
                break

        return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000

    async def validate(self, invoice_data: InvoiceData) -> list[str]:
        """Validate extracted invoice data.

        Args:
            invoice_data: Extracted data to validate.

        Returns:
            List of validation warnings.
        """
        warnings: list[str] = []

        if not invoice_data.invoice_number or invoice_data.invoice_number == "UNKNOWN":
            warnings.append("Missing invoice number")

        if (
            not invoice_data.supply_point.ean_code
            and not invoice_data.supply_point.eic_code
        ):
            warnings.append("No supply point identifier found (EAN/EIC)")

        if (invoice_data.due_date and invoice_data.issue_date
                and invoice_data.due_date < invoice_data.issue_date):
            warnings.append("Due date is before issue date")

        if invoice_data.is_cross_year():
            warnings.append("Billing period spans multiple years (cross-year invoice)")

        if invoice_data.is_correction and not invoice_data.correction_info:
            warnings.append("Correction invoice without original invoice reference")

        return warnings


class VisionLLMExtractionStrategy(LangChainExtractionStrategy):
    """LLM extraction using vision models for image input.

    Extends LangChainExtractionStrategy to support sending images
    directly to vision-capable models (GPT-4o, Claude 3.5 Sonnet).
    Uses base64-encoded images with LCEL chains.
    """

    def __init__(
        self,
        model_provider: Literal["openai", "anthropic"] = "openai",
        model_name: str | None = None,
        temperature: float = 0.0,
        retriever=None,
    ) -> None:
        """Initialize vision LLM strategy.

        Args:
            retriever: Optional FewShotRetriever. When provided, similar
                       GT invoice examples are injected into the vision prompt.
        """
        super().__init__(
            model_provider=model_provider,
            model_name=model_name,
            temperature=temperature,
            use_vision=True,
        )
        self._retriever = retriever

    @property
    def name(self) -> str:
        """Return strategy name identifier."""
        model = self._model_name or self._get_default_model_name()
        return f"vision_llm_{self._provider}_{model}"

    async def extract(self, context: ExtractionContext) -> ExtractionResult:
        """Extract using vision model with image input.

        Args:
            context: ExtractionContext with image_bytes.

        Returns:
            ExtractionResult from vision model.
        """
        import base64

        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.output_parsers import PydanticOutputParser

        errors: list[str] = []
        warnings: list[str] = []
        start_time = time.time()

        try:
            # Get image bytes
            if not context.image_bytes:
                warnings.append("No image bytes provided, falling back to text extraction")
                return await super().extract(context)

            # Detect commodity from text (if available)
            commodity = context.commodity_hint
            if not commodity and context.raw_text:
                from src.core.extraction.regex_strategy import RegexExtractionStrategy
                regex = RegexExtractionStrategy()
                commodity = regex._detect_commodity(context.raw_text)

            # Default commodity if not detected
            if not commodity:
                commodity = CommodityType.ELEKTRINA_NN
                warnings.append("Could not detect commodity, defaulting to electricity NN")

            # Initialize LLM
            llm = self._initialize_llm()
            parser = PydanticOutputParser(pydantic_object=LLMExtractedInvoice)

            # Encode image to base64
            image_b64 = base64.b64encode(context.image_bytes).decode("utf-8")
            image_media_type = "image/png"  # Assume PNG from PDF conversion

            # Build messages with image
            commodity_fields = COMMODITY_FIELD_PROMPTS.get(commodity, "")

            few_shot = (
                self._retriever.get_examples(context.raw_text, n=2)
                if self._retriever and context.raw_text
                else ""
            )

            vision_prompt = f"""Extract invoice data from this image.

{few_shot}Required fields:
- invoice_number, variable_symbol, ean_code, eic_code
- period_from, period_to, issue_date, due_date (DD.MM.YYYY format)
- customer_tax_id, supplier_tax_id
- total_amount_inc_vat, total_amount_ex_vat (CZK)

{commodity_fields}

Check if this is a correction invoice (opravná faktura) or transitional invoice (přechodová faktura).

{parser.get_format_instructions()}"""

            # Create message with image content
            if self._provider == "openai":
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=[
                            {"type": "text", "text": vision_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_media_type};base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                        ]
                    ),
                ]
            else:  # anthropic
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(
                        content=[
                            {"type": "text", "text": vision_prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_media_type,
                                    "data": image_b64,
                                },
                            },
                        ]
                    ),
                ]

            # Invoke LLM
            logger.info(f"Invoking vision LLM ({self._provider}) with image")
            response = await llm.ainvoke(messages)

            # Parse response
            extracted = parser.parse(response.content)

            # Convert to InvoiceData
            invoice_data = self._convert_to_invoice_data(
                extracted,
                commodity,
                context.source_filename,
            )

            # Derive missing amount from DPH table
            invoice_data, _vat_derived = apply_vat_derivation_to_invoice(invoice_data)
            if _vat_derived:
                warnings.append("total_amount derived via DPH table (one amount was null)")

            # Fix 3: deterministic amount_inc_vat correction from the VAT rate
            invoice_data, _vat_corr = apply_vat_inc_correction_to_invoice(
                invoice_data, enabled=ENABLE_VAT_INC_CORRECTION
            )
            if _vat_corr:
                warnings.append(
                    "amount_inc_vat corrected via VAT rate "
                    f"{_vat_corr['vat_rate_used']}% "
                    f"({_vat_corr['extracted_value']} → {_vat_corr['computed_value']})"
                )

            # Validate
            validation_warnings = await self.validate(invoice_data)
            warnings.extend(validation_warnings)

            confidence = self._calculate_confidence(invoice_data)

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"Vision extraction completed in {elapsed_ms}ms, "
                f"confidence={confidence:.2f}"
            )

            return ExtractionResult(
                source_file=context.source_filename,
                strategy_name=self.name,
                confidence=confidence,
                raw_text=context.raw_text,
                invoice_data=invoice_data,
                errors=errors,
                warnings=warnings,
            )

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            error_msg = f"Vision LLM extraction failed after {elapsed_ms}ms: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            return ExtractionResult(
                source_file=context.source_filename,
                strategy_name=self.name,
                raw_text=context.raw_text,
                errors=errors,
            )


def create_langchain_strategy(
    provider: Literal["openai", "anthropic"] = "openai",
    model_name: str | None = None,
    use_vision: bool = False,
    temperature: float = 0.0,
    retriever=None,
) -> LangChainExtractionStrategy:
    """Factory function to create LangChain extraction strategy.

    Args:
        provider: LLM provider ('openai' or 'anthropic').
        model_name: Specific model name override.
        use_vision: If True, create vision-capable strategy.
        temperature: LLM temperature (0.0 for deterministic).
        retriever: Optional FewShotRetriever for RAG-based few-shot prompting
                   (only used when use_vision=True).

    Returns:
        Configured LangChainExtractionStrategy or VisionLLMExtractionStrategy.
    """
    if use_vision:
        return VisionLLMExtractionStrategy(
            model_provider=provider,
            model_name=model_name,
            temperature=temperature,
            retriever=retriever,
        )
    return LangChainExtractionStrategy(
        model_provider=provider,
        model_name=model_name,
        temperature=temperature,
        retriever=retriever,
    )
