#!/usr/bin/env python
"""
Příprava anonymizovaných datových podkladů pro reportovací notebooky.

Surové výstupy běhu (soubory `.jsonl`, které zapisují `run_ds1_final.py`
a `run_ds3_final.py`) obsahují skutečné hodnoty z reálných faktur, a proto
nepatří do veřejného repozitáře.
Tento skript z nich odvodí dvojici tabulek, které neobsahují žádnou hodnotu
z faktury, ale zachovávají vše potřebné pro výpočet všech metrik uvedených
v kapitole 5 práce:

    <sada>_per_invoice.csv   jeden řádek na fakturu — komodita, třída kvality,
                             eskalace, tp/fp/fn, latence, náklady
    <sada>_per_field.csv     jeden řádek na (faktura, pole) — pouze status
                             tp / fp / fn, nikoli hodnota

Použité skórovací konvence odpovídají vyhodnocení v práci:

  * pole ``amount_inc_vat`` se hodnotí proti syrové extrakci (deterministická
    dopočítaná korekce DPH byla z pipeline vyřazena — na DS1 přesnost snižovala),
  * konvence ``null == 0``: chybí-li v predikci hodnota a ground truth je
    číselná nula, počítá se pole jako správně (položka na faktuře není).

Ze stejného důvodu skript odstraní citlivé sloupce (název souboru, dodavatel,
vyextrahovaná a referenční hodnota) z tabulek dílčích experimentů na sadě DS2.
Metriky se počítají z předpočítaných tp/fp/fn, takže odstraněním hodnot se
žádný výsledek uvedený v práci nemění.

Spuštění:
    python scripts/build_thesis_datasets.py
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "experiments" / "data"

# Surové výstupy běhu. Skripty run_ds1_final.py / run_ds3_final.py je ukládají
# s názvem odvozeným od značky běhu, hledá se proto poslední odpovídající soubor.
SLOZKY = {
    "ds1": DATA / "07_evaluace_ds1",
    "ds3": DATA / "08_generalizace_ds3",
}


def surovy_vystup(slozka: Path) -> Path | None:
    """Nejnovější .jsonl se surovými extrakcemi ve složce sady."""
    soubory = sorted(slozka.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return soubory[-1] if soubory else None

# Sloupce, které nesmí opustit lokální stroj: identifikace konkrétní faktury
# a jakákoli hodnota z ní.
CITLIVE_SLOUPCE = ["pdf_name", "supplier", "extracted", "gt", "gt_json",
                   "extracted_json", "doc_id"]

# Tabulky experimentů na DS2, které je nutné před commitem očistit.
DS2_TABULKY = [
    DATA / "02_predzpracovani" / "preproc_v2_doc.csv",
    DATA / "03_prostorove_kodovani" / "lie_benchmark_full.csv",
    DATA / "04_jazyk_promptu" / "prompt_benchmark_full.csv",
    DATA / "05_strategie_modely" / "nb05_results.csv",
]

INVOICE_COLS = [
    "doc_idx", "commodity", "quality", "escalated", "escalation_reason",
    "tp", "fp", "fn", "precision", "recall", "f1", "null_rate",
    "ocr_ms", "text_llm_ms", "vision_llm_ms", "full_pipeline_ms",
    "prompt_tokens", "completion_tokens", "cost_usd", "json_valid",
]


def is_zero(value) -> bool:
    """Ground truth je číselná nula (položka na faktuře chybí)."""
    try:
        return float(value) == 0.0
    except (ValueError, TypeError):
        return False


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def field_statuses(row: dict) -> dict[str, str]:
    """Status každého pole po aplikaci obou skórovacích konvencí."""
    out: dict[str, str] = {}
    for name, info in json.loads(row["per_field_json"]).items():
        status = info["status"]

        # amount_inc_vat se skóruje proti syrové extrakci
        if name == "amount_inc_vat" and row.get("aiv_gt_present"):
            if row.get("aiv_match_raw"):
                status = "tp"
            elif str(row.get("amount_inc_vat_raw") or "").strip():
                status = "fp"
            else:
                status = "fn"

        if status == "no_gt":
            continue

        # null == 0
        if status == "fn" and is_zero(info.get("gt", "")) and not info.get("pred", ""):
            status = "tp"

        out[name] = status
    return out


def build(name: str, slozka: Path) -> None:
    src = surovy_vystup(slozka)
    if src is None:
        print(f"  [preskoceno] ve slozce {slozka.relative_to(ROOT)} neni surovy vystup (.jsonl)")
        return

    rows = [json.loads(line) for line in src.open(encoding="utf-8")]
    out_dir = slozka
    inv_path = out_dir / f"{name}_per_invoice.csv"
    fld_path = out_dir / f"{name}_per_field.csv"

    with inv_path.open("w", newline="", encoding="utf-8") as fi, \
         fld_path.open("w", newline="", encoding="utf-8") as ff:
        wi = csv.DictWriter(fi, fieldnames=INVOICE_COLS)
        wf = csv.DictWriter(ff, fieldnames=["doc_idx", "field", "status"])
        wi.writeheader()
        wf.writeheader()

        for idx, row in enumerate(rows):
            statuses = field_statuses(row)
            tp = sum(1 for s in statuses.values() if s == "tp")
            fp = sum(1 for s in statuses.values() if s == "fp")
            fn = sum(1 for s in statuses.values() if s == "fn")
            p, r, f1 = prf(tp, fp, fn)

            wi.writerow({
                "doc_idx": idx,
                "commodity": row["commodity"],
                "quality": row["quality"],
                "escalated": int(row.get("escalated") or 0),
                "escalation_reason": row.get("escalation_reason") or "",
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6),
                "null_rate": row.get("null_rate"),
                "ocr_ms": row.get("ocr_ms"),
                "text_llm_ms": row.get("text_llm_ms"),
                "vision_llm_ms": row.get("vision_llm_ms"),
                "full_pipeline_ms": row.get("full_pipeline_ms"),
                "prompt_tokens": row.get("prompt_tokens"),
                "completion_tokens": row.get("completion_tokens"),
                "cost_usd": row.get("cost_usd"),
                "json_valid": int(row.get("api_error") is None),
            })
            for field, status in statuses.items():
                wf.writerow({"doc_idx": idx, "field": field, "status": status})

    macro = sum(
        prf(r_["tp"], r_["fp"], r_["fn"])[2] for r_ in
        [{"tp": sum(1 for s in field_statuses(r).values() if s == "tp"),
          "fp": sum(1 for s in field_statuses(r).values() if s == "fp"),
          "fn": sum(1 for s in field_statuses(r).values() if s == "fn")} for r in rows]
    ) / len(rows)
    print(f"  {name}: {len(rows)} faktur -> {inv_path.name}, {fld_path.name} "
          f"(makro F1 = {macro:.4f})")


def anonymni_id(hodnota: str) -> str:
    """Stabilní neidentifikující klíč dokumentu (zachová párování mezi řádky)."""
    return hashlib.sha1(str(hodnota).encode("utf-8")).hexdigest()[:10]


def sanitize(path: Path) -> None:
    """Odstraní citlivé sloupce z tabulky experimentu na DS2."""
    if not path.exists():
        print(f"  [preskoceno] chybi {path.relative_to(ROOT)}")
        return

    df = pd.read_csv(path)
    odstranene = [c for c in CITLIVE_SLOUPCE if c in df.columns]
    if not odstranene:
        print(f"  {path.name}: jiz ocisteno")
        return

    # Identifikátor dokumentu nahradíme neidentifikujícím klíčem, aby zůstalo
    # možné párovat řádky téhož dokumentu napříč variantami.
    for sloupec in ("doc_id", "pdf_name"):
        if sloupec in df.columns:
            df.insert(0, "dokument", df[sloupec].map(anonymni_id))
            break

    df = df.drop(columns=odstranene)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"  {path.name}: odstraneno {odstranene}")


def main() -> None:
    print("Odvozeni anonymizovanych tabulek:")
    for name, slozka in SLOZKY.items():
        build(name, slozka)

    print("Ocisteni tabulek experimentu na DS2:")
    for path in DS2_TABULKY:
        sanitize(path)


if __name__ == "__main__":
    main()
