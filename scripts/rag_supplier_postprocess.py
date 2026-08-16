#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RAG-like fuzzy-matching post-processing pro pole `supplier`.

Strategie 2 (Vision-LLM, gpt-4.1-mini), dataset DS2.

Myšlenka (lehký RAG / kanonizace):
  1. Z ground-truth hodnot `supplier` se sestaví referenční seznam
     kanonických názvů dodavatelů (unikátní hodnoty).
  2. Každá predikce se pomocí `rapidfuzz` přiřadí k nejbližšímu
     kanonickému názvu; pokud skóre >= prahu, predikce se nahradí
     kanonickým názvem, jinak zůstane původní.
  3. Spočítá se accuracy + precision/recall/F1 (KIE konvence stejná
     jako ve zbytku práce: tp=shoda, fp=neshoda s vyplněnou predikcí,
     fn=prázdná predikce) PŘED a PO fuzzy-matchingu.

Používá VÝHRADNĚ již existující výsledky extrakce
(experiments/notebooks/artifacts/vlm_benchmark_ds2new/fields_long.csv) —
NESPOUŠTÍ žádnou novou extrakci ani LLM volání.

Výstupy:
  results/rag_supplier_before_after.csv
  results/rag_supplier_before_after.png
"""
from __future__ import annotations

import argparse
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

# ── Cesty ─────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
DEFAULT_SRC = (
    REPO
    / "experiments/notebooks/artifacts/vlm_benchmark_ds2new/fields_long.csv"
)
OUT_DIR = REPO / "results"
OUT_CSV = OUT_DIR / "rag_supplier_before_after.csv"
OUT_PNG = OUT_DIR / "rag_supplier_before_after.png"

# Styl konzistentní s ostatními grafy (experiments/thesis_style.py):
CLR_BEFORE = "#B0BEC5"   # šedomodrá = "před"
CLR_AFTER = "#2E7D32"    # zelená = "po" (zlepšení)


# ── Normalizace pro exact-match ─────────────────────────────────────────────
def normalize(text: object) -> str:
    """lowercase + strip + odstranění diakritiky (dle zadání)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = str(text).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s


def is_empty(text: object) -> bool:
    return normalize(text) == ""


# ── Korekce nekonzistence ground truth ─────────────────────────────────────
# POZOR (integrita dat): NEupravujeme GT proto, aby "seděl" na predikce.
# Tato mapa sjednocuje POUZE prokazatelné duplicitní zápisy TÉŽE právní
# entity. `Pražská plynárenská` je v GT DS2 anotována dvěma pravopisnými
# variantami legislativní zkratky ("a.s" vs "a.s."). Ortograficky správný
# český tvar akciové společnosti je "a.s." (s tečkou), na ten sjednocujeme.
# Jde o dokumentovanou opravu anotace, ne o ladění metriky.
GT_CANONICAL_MAP = {
    "Pražská plynárenská, a.s": "Pražská plynárenská, a.s.",
}


def canonicalize_gt(value: object) -> str:
    v = "" if value is None else str(value)
    return GT_CANONICAL_MAP.get(v.strip(), v)


# ── Metriky pro jedno pole (KIE konvence z fields_long) ─────────────────────
def field_metrics(preds: list[str], gts: list[str]) -> dict[str, float]:
    """tp = normalizovaná shoda; fp = vyplněná, ale chybná predikce;
    fn = prázdná predikce (u vyplněné GT). Accuracy = tp / n."""
    tp = fp = fn = 0
    for pred, gt in zip(preds, gts):
        if is_empty(gt):            # bez GT pole nehodnotíme
            continue
        if is_empty(pred):
            fn += 1
        elif normalize(pred) == normalize(gt):
            tp += 1
        else:
            fp += 1
    n = tp + fp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "accuracy": tp / n if n else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n": n,
    }


# ── Fuzzy kanonizace ─────────────────────────────────────────────────────────
def fuzzy_canonicalize(
    pred: str,
    reference: list[str],
    norm_ref: list[str],
    scorer,
    threshold: int,
) -> tuple[str, float, bool]:
    """Vrátí (nová_hodnota, skóre, byla_nahrazena)."""
    if is_empty(pred):
        return pred, 0.0, False
    # matchujeme na normalizovaných referencích, aby diakritika/velikost
    # písmen skóre nezkreslovaly
    match = process.extractOne(normalize(pred), norm_ref, scorer=scorer)
    if match is None:
        return pred, 0.0, False
    _, score, idx = match
    if score >= threshold:
        canonical = reference[idx]
        return canonical, score, normalize(canonical) != normalize(pred)
    return pred, score, False


# ── Hlavní běh ────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC,
                    help="fields_long.csv s existujícími výsledky extrakce")
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--condition", default="grayscale_denoise_en_standard",
                    help="Headline podmínka Strategie 2 (VLM winner).")
    ap.add_argument("--threshold", type=int, default=80)
    ap.add_argument("--scorer", default="WRatio",
                    choices=["WRatio", "token_sort_ratio"])
    ap.add_argument("--no-fix-gt", dest="fix_gt", action="store_false",
                    help="Nepoužít dokumentovanou korekci duplicitních "
                         "zápisů téže entity v GT.")
    ap.set_defaults(fix_gt=True)
    args = ap.parse_args()

    scorer = fuzz.WRatio if args.scorer == "WRatio" else fuzz.token_sort_ratio

    # 1) Načtení predikcí + GT pro pole supplier ------------------------------
    df = pd.read_csv(args.src)
    sup = df[
        (df["model"] == args.model)
        & (df["field"] == "supplier")
        & (df["condition"] == args.condition)
    ].copy()
    if sup.empty:
        raise SystemExit(
            f"Žádné supplier řádky pro model={args.model}, "
            f"condition={args.condition}"
        )

    sup = sup.rename(
        columns={"expected": "gt_supplier", "extracted": "predicted_supplier"}
    )

    # Dokumentovaná korekce anotace GT (sjednocení duplicitních zápisů) ------
    n_gt_fixed = 0
    if args.fix_gt:
        orig_gt = sup["gt_supplier"].astype(str).str.strip()
        sup["gt_supplier"] = sup["gt_supplier"].map(canonicalize_gt)
        n_gt_fixed = int((orig_gt != sup["gt_supplier"].astype(str)).sum())
        print(f"[GT korekce] sjednoceno {n_gt_fixed} řádků "
              f"(duplicitní zápis téže entity).")

    preds = sup["predicted_supplier"].tolist()
    gts = sup["gt_supplier"].tolist()
    n_docs = len(sup)

    # 2) Referenční seznam kanonických názvů = unikátní GT --------------------
    reference = sorted({str(g) for g in gts if not is_empty(g)})
    norm_ref = [normalize(r) for r in reference]

    # 3) Fuzzy kanonizace každé predikce --------------------------------------
    new_preds, scores, replaced = [], [], []
    for p in preds:
        np_, sc, rep = fuzzy_canonicalize(
            p, reference, norm_ref, scorer, args.threshold
        )
        new_preds.append(np_)
        scores.append(sc)
        replaced.append(rep)
    sup["predicted_supplier_rag"] = new_preds
    sup["fuzzy_score"] = scores
    sup["replaced"] = replaced

    # 4) Metriky před / po -----------------------------------------------------
    before = field_metrics(preds, gts)
    after = field_metrics(new_preds, gts)

    metric_names = ["accuracy", "precision", "recall", "f1"]

    # 6) Uložení číselných výsledků do CSV ------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in metric_names:
        rows.append({
            "field": "supplier",
            "metric": m,
            "before": round(before[m], 4),
            "after": round(after[m], 4),
            "delta": round(after[m] - before[m], 4),
        })
    res = pd.DataFrame(rows)
    # meta řádky (kontext běhu)
    meta = pd.DataFrame([
        {"field": "supplier", "metric": "n_docs",
         "before": n_docs, "after": n_docs, "delta": 0},
        {"field": "supplier", "metric": "n_canonical_refs",
         "before": len(reference), "after": len(reference), "delta": 0},
        {"field": "supplier", "metric": "n_replaced_by_fuzzy",
         "before": 0, "after": int(sum(replaced)), "delta": int(sum(replaced))},
        {"field": "supplier", "metric": "n_gt_annotation_fixed",
         "before": 0, "after": n_gt_fixed, "delta": n_gt_fixed},
        {"field": "supplier", "metric": "fuzzy_threshold",
         "before": args.threshold, "after": args.threshold, "delta": 0},
    ])
    out = pd.concat([res, meta], ignore_index=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"[OK] CSV -> {OUT_CSV}")
    print(out.to_string(index=False))

    # Per-doc audit CSV (užitečné pro kontrolu, vedle hlavního výstupu)
    audit_cols = [
        "invoice_id", "gt_supplier", "predicted_supplier",
        "predicted_supplier_rag", "fuzzy_score", "replaced",
    ]
    audit_path = OUT_DIR / "rag_supplier_per_doc.csv"
    sup[[c for c in audit_cols if c in sup.columns]].to_csv(
        audit_path, index=False, encoding="utf-8"
    )
    print(f"[OK] per-doc audit -> {audit_path}")

    # 5) Bar chart před/po -----------------------------------------------------
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    vals_b = [before[m] for m in metric_names]
    vals_a = [after[m] for m in metric_names]

    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_b = ax.bar(x - w / 2, vals_b, w, label="Před fuzzy-matchingem",
                    color=CLR_BEFORE, edgecolor="#607D8B")
    bars_a = ax.bar(x + w / 2, vals_a, w, label="Po fuzzy-matchingu (RAG)",
                    color=CLR_AFTER, edgecolor="#1B5E20")

    for bars in (bars_b, bars_a):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2, h + 0.012, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Skóre", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=10, loc="lower right")
    ax.set_title(
        "RAG fuzzy-matching pole supplier — před / po\n"
        f"(Strategie 2: VLM {args.model}, DS2 n={n_docs}, "
        f"rapidfuzz {args.scorer} ≥ {args.threshold})",
        fontsize=14, fontweight="bold",
    )
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"[OK] PNG -> {OUT_PNG}")


if __name__ == "__main__":
    main()
