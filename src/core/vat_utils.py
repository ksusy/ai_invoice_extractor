"""Czech VAT (DPH) rate lookup and amount derivation.

Covers all six utility commodity types and their historical rate changes:
- Electricity/Gas: 21 % standard; 0 % COVID relief (Nov–Dec 2021)
- Water: 15 % → 10 % (May 2020) → 12 % (Jan 2024)
- Heat: 15 % → 10 % (Jan 2020) → 12 % (Jan 2024)

Use derive_missing_amounts() as a post-processing step after LLM extraction:
if the model returned only one of (amount_ex_vat, amount_inc_vat), the other
is computed from the applicable VAT rate.
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import NamedTuple


class _VatBand(NamedTuple):
    valid_from: date
    valid_to: date | None  # None = open-ended (current)
    rate: float            # percent, e.g. 21.0


# ---------------------------------------------------------------------------
# Historical VAT bands per commodity (based on Czech legislative changes)
# ---------------------------------------------------------------------------
_BANDS: dict[str, list[_VatBand]] = {
    "elektrina_nn": [
        _VatBand(date(2013, 1, 1), date(2021, 10, 31), 21.0),
        _VatBand(date(2021, 11, 1), date(2021, 12, 31), 0.0),  # COVID energy relief
        _VatBand(date(2022, 1, 1), None, 21.0),
    ],
    "elektrina_vn": [
        _VatBand(date(2013, 1, 1), date(2021, 10, 31), 21.0),
        _VatBand(date(2021, 11, 1), date(2021, 12, 31), 0.0),
        _VatBand(date(2022, 1, 1), None, 21.0),
    ],
    "plyn_mo": [
        _VatBand(date(2013, 1, 1), date(2021, 10, 31), 21.0),
        _VatBand(date(2021, 11, 1), date(2021, 12, 31), 0.0),
        _VatBand(date(2022, 1, 1), None, 21.0),
    ],
    "plyn_vo": [
        _VatBand(date(2013, 1, 1), date(2021, 10, 31), 21.0),
        _VatBand(date(2021, 11, 1), date(2021, 12, 31), 0.0),
        _VatBand(date(2022, 1, 1), None, 21.0),
    ],
    "voda": [
        _VatBand(date(2013, 1, 1), date(2020, 4, 30), 15.0),
        _VatBand(date(2020, 5, 1), date(2023, 12, 31), 10.0),
        _VatBand(date(2024, 1, 1), None, 12.0),
    ],
    "teplo": [
        _VatBand(date(2013, 1, 1), date(2019, 12, 31), 15.0),
        _VatBand(date(2020, 1, 1), date(2023, 12, 31), 10.0),
        _VatBand(date(2024, 1, 1), None, 12.0),
    ],
}

# Normalize commodity key variants (plyn_MO → plyn_mo, etc.)
_ALIASES = {
    "plyn_mo": "plyn_mo",
    "plyn_MO": "plyn_mo",
    "plyn_vo": "plyn_vo",
    "plyn_VO": "plyn_vo",
    "elektrina_nn": "elektrina_nn",
    "elektrina_vn": "elektrina_vn",
    "voda": "voda",
    "teplo": "teplo",
}


# ---------------------------------------------------------------------------
# dph.csv parsing — build the band table from the thesis data file (Fix 3)
# ---------------------------------------------------------------------------
# The historical VAT rates live in data/dph.csv. Its named rows carry the
# commodity in the "popis" column and the *inclusive end date* (obdobi_do) of
# each rate tier; the open-ended (current) tier has an empty date. We convert
# each commodity's ordered tiers into (valid_from, valid_to, rate) bands where a
# tier's valid_from = previous tier's end + 1 day.
_DPH_CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "dph.csv"

_COMMODITY_NAME_MAP = {
    "elektrina nn": "elektrina_nn",
    "elektřina nn": "elektrina_nn",
    "elektrina vn": "elektrina_vn",
    "elektřina vn": "elektrina_vn",
    "plyn mo": "plyn_mo",
    "plyn vo": "plyn_vo",
    "teplo": "teplo",
    "voda": "voda",
}


def _parse_iso_date(s: str | None) -> date | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


def load_vat_bands_from_csv(path: str | Path = _DPH_CSV_PATH) -> dict[str, list[_VatBand]] | None:
    """Parse dph.csv into ``{commodity: [(valid_from, valid_to, rate), ...]}``.

    dph.csv holds two mirrored blocks in the same commodity order:

    * *labelled* rows — ``popis`` = commodity name + a single boundary date; and
    * *explicit-range* rows — ``date_from ; date_to ; rate`` triples (an empty
      ``date_to`` meaning "still in effect").

    The explicit-range rows are the authoritative source of the
    ``(date_from, date_to, vat_rate)`` bands the pipeline needs: the labelled
    gas rows carry a data-entry error (they reuse the heat boundary
    ``2019-12-31`` instead of the energy VAT-holiday boundary ``2021-10-31``),
    whereas the explicit-range gas rows are correct. We therefore read the
    commodity order and per-commodity tier counts from the labelled block, then
    assign the explicit-range triples to commodities in that order.

    Returns None if the file is missing/unreadable or the two blocks don't line
    up (callers then fall back to the hardcoded legislative table).
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open(encoding="utf-8-sig", newline="") as fh:
            rows = [r for r in csv.reader(fh, delimiter=";") if any(c.strip() for c in r)]
    except OSError:
        return None

    commodity_order: list[str] = []
    tier_counts: dict[str, int] = {}
    triples: list[tuple[date | None, date | None, float]] = []

    for row in rows[1:]:  # skip header
        if len(row) < 5:
            continue
        popis = row[2].strip()
        commodity = _COMMODITY_NAME_MAP.get(popis.lower())
        if commodity is not None:
            # Labelled row → record commodity order + tier count.
            if commodity not in tier_counts:
                commodity_order.append(commodity)
                tier_counts[commodity] = 0
            tier_counts[commodity] += 1
        elif _parse_iso_date(popis) is not None or not popis:
            # Explicit-range row: date_from ; date_to ; rate
            try:
                rate = float(row[4])
            except (ValueError, IndexError):
                continue
            triples.append((_parse_iso_date(row[2]), _parse_iso_date(row[3]), rate))

    total_expected = sum(tier_counts.values())
    if not commodity_order or len(triples) != total_expected:
        return None  # blocks don't line up — let caller fall back

    bands: dict[str, list[_VatBand]] = {}
    cursor = 0
    for commodity in commodity_order:
        n = tier_counts[commodity]
        chunk = triples[cursor:cursor + n]
        cursor += n
        out = [
            _VatBand(df or date(2013, 1, 1), dt, rate)
            for df, dt, rate in chunk
        ]
        bands[commodity] = out
    return bands


def _bands_are_contiguous(bands: list[_VatBand]) -> bool:
    """True if bands cover the timeline without gaps and end open-ended.

    A valid VAT-rate table must leave no date uncovered. This is used to reject
    malformed dph.csv entries: the gas rows there carry a data-entry error that
    leaves 2020-01-01…2021-10-31 uncovered (they reuse the heat boundary
    2019-12-31 instead of the energy VAT-holiday boundary 2021-10-31), which
    contradicts the documented Nov–Dec 2021 energy holiday.
    """
    if not bands:
        return False
    ordered = sorted(bands, key=lambda b: b.valid_from)
    for prev, nxt in zip(ordered, ordered[1:]):
        if prev.valid_to is None:  # only the last band may be open-ended
            return False
        if nxt.valid_from > prev.valid_to + timedelta(days=1):
            return False  # gap between tiers
    return ordered[-1].valid_to is None


# Adopt bands parsed from dph.csv (the thesis source of truth) per commodity,
# but only when they fully cover the timeline — this keeps the correct,
# legislatively-validated hardcoded bands for gas, whose CSV rows are erroneous.
try:
    _csv_bands = load_vat_bands_from_csv()
    if _csv_bands:
        for _comm, _bands in _csv_bands.items():
            if _bands_are_contiguous(_bands):
                _BANDS[_comm] = _bands
except Exception:  # pragma: no cover - defensive; never break import
    pass


def get_vat_rate(commodity: str, period_to: date | str | None) -> float | None:
    """Return the applicable VAT rate (%) for a commodity on a given invoice date.

    Args:
        commodity: commodity string, e.g. "elektrina_nn", "plyn_MO", "voda"
        period_to: invoice period_to date (used to pick the correct band)

    Returns:
        VAT rate as float (e.g. 21.0) or None if commodity unknown / date unparseable.
    """
    key = _ALIASES.get(commodity)
    if key is None:
        return None

    if period_to is None:
        return None

    if isinstance(period_to, str):
        from src.domain.entities import parse_czech_date
        period_to = parse_czech_date(period_to)
        if period_to is None:
            return None

    for band in _BANDS[key]:
        if period_to < band.valid_from:
            continue
        if band.valid_to is not None and period_to > band.valid_to:
            continue
        return band.rate

    return None


def derive_missing_amounts(
    amount_ex_vat: float | None,
    amount_inc_vat: float | None,
    commodity: str,
    period_to: date | str | None,
) -> tuple[float | None, float | None, bool]:
    """Derive one amount from the other using the historical VAT rate.

    If both are present or both are None, returns them unchanged.
    If exactly one is present, computes the other.

    Args:
        amount_ex_vat:  castka_bez_dph
        amount_inc_vat: castka_s_dph
        commodity:      commodity type string
        period_to:      invoice period end date

    Returns:
        (amount_ex_vat, amount_inc_vat, was_derived)
        was_derived=True when a value was computed (not extracted directly).
    """
    # Nothing to do if both present or both absent
    if (amount_ex_vat is None) == (amount_inc_vat is None):
        return amount_ex_vat, amount_inc_vat, False

    rate = get_vat_rate(commodity, period_to)
    if rate is None:
        return amount_ex_vat, amount_inc_vat, False

    multiplier = 1.0 + rate / 100.0

    if amount_inc_vat is not None and amount_ex_vat is None:
        # Derive ex_vat from inc_vat
        derived = round(amount_inc_vat / multiplier, 2)
        return derived, amount_inc_vat, True

    if amount_ex_vat is not None and amount_inc_vat is None:
        # Derive inc_vat from ex_vat
        derived = round(amount_ex_vat * multiplier, 2)
        return amount_ex_vat, derived, True

    return amount_ex_vat, amount_inc_vat, False  # unreachable


def apply_vat_derivation_to_invoice(invoice_data: InvoiceData) -> tuple[InvoiceData, bool]:  # noqa: F821
    """Apply DPH amount derivation to an InvoiceData object in-place.

    Derives total_amount_ex_vat ↔ total_amount_inc_vat at invoice level.
    Returns (invoice_data, was_modified).
    """
    period_to = None
    if invoice_data.period is not None:
        period_to = invoice_data.period.period_to

    commodity = invoice_data.commodity.value if invoice_data.commodity else None

    ex_vat, inc_vat, derived = derive_missing_amounts(
        invoice_data.total_amount_ex_vat,
        invoice_data.total_amount_inc_vat,
        commodity or "",
        period_to,
    )

    if derived:
        invoice_data.total_amount_ex_vat = ex_vat
        invoice_data.total_amount_inc_vat = inc_vat

    return invoice_data, derived


# ---------------------------------------------------------------------------
# Deterministic amount_inc_vat correction (Fix 3)
# ---------------------------------------------------------------------------
# The model confuses amount_ex_vat, the DPH amount, and amount_inc_vat (three
# visually adjacent numbers in the invoice table). When we know the applicable
# VAT rate we can recompute inc_vat = ex_vat * (1 + rate/100) deterministically
# and replace an extracted value that disagrees by more than a small tolerance.

VAT_INC_TOLERANCE_CZK = 1.0


def correct_amount_inc_vat(
    amount_ex_vat: float | None,
    amount_inc_vat: float | None,
    commodity: str,
    ref_date: date | str | None,
    tolerance: float = VAT_INC_TOLERANCE_CZK,
) -> tuple[float | None, bool, float | None, float | None]:
    """Deterministically correct ``amount_inc_vat`` from ``amount_ex_vat``.

    Computes ``amount_inc_vat_computed = amount_ex_vat * (1 + rate/100)`` using
    the VAT rate applicable to ``commodity`` on ``ref_date``. If the extracted
    ``amount_inc_vat`` is missing or differs from the computed value by more
    than ``tolerance`` CZK, the computed value is returned instead.

    Returns ``(value, was_corrected, rate_used, computed_value)``. When the rate
    is unknown or ``amount_ex_vat`` is missing, the input is returned unchanged
    with ``was_corrected=False``.
    """
    if amount_ex_vat is None:
        return amount_inc_vat, False, None, None

    rate = get_vat_rate(commodity, ref_date)
    if rate is None:
        return amount_inc_vat, False, None, None

    computed = round(amount_ex_vat * (1.0 + rate / 100.0), 2)
    if amount_inc_vat is None or abs(amount_inc_vat - computed) > tolerance:
        return computed, True, rate, computed
    return amount_inc_vat, False, rate, computed


def apply_vat_inc_correction_to_invoice(
    invoice_data: InvoiceData,  # noqa: F821
    enabled: bool = True,
    tolerance: float = VAT_INC_TOLERANCE_CZK,
) -> tuple[InvoiceData, dict | None]:  # noqa: F821
    """Apply :func:`correct_amount_inc_vat` to an InvoiceData in place.

    Uses ``issue_date`` (falling back to ``period.period_from``) to pick the VAT
    rate, per Fix 3. Returns ``(invoice_data, audit_record_or_None)`` where the
    audit record is ``{invoice_id, extracted_value, computed_value, vat_rate_used}``
    whenever a replacement occurred.
    """
    if not enabled:
        return invoice_data, None

    ref_date = invoice_data.issue_date
    if ref_date is None and invoice_data.period is not None:
        ref_date = invoice_data.period.period_from

    commodity = invoice_data.commodity.value if invoice_data.commodity else ""
    extracted = invoice_data.total_amount_inc_vat

    corrected, was_corrected, rate_used, computed = correct_amount_inc_vat(
        invoice_data.total_amount_ex_vat,
        extracted,
        commodity,
        ref_date,
        tolerance,
    )

    if was_corrected:
        invoice_data.total_amount_inc_vat = corrected
        record = {
            "invoice_id": invoice_data.source_filename or invoice_data.invoice_number,
            "extracted_value": extracted,
            "computed_value": computed,
            "vat_rate_used": rate_used,
        }
        return invoice_data, record

    return invoice_data, None
