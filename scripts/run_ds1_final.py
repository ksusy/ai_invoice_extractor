"""
DS1 Final Evaluation — dávkové vyhodnocení kaskádové pipeline.

Samotnou extrakci provádí sdílený modul :mod:`src.core.cascade`, na kterém běží
i REST API. Zde zůstává jen to, co patří k vyhodnocení: načtení ground truth,
výběr vzorku, výpočet metrik a uložení výsledků.

Spuštění:
    python scripts/run_ds1_final.py
    python scripts/run_ds1_final.py --limit 20
    python scripts/run_ds1_final.py --dry-run
"""
from __future__ import annotations

import sys as _sys

if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(_sys.stderr, "reconfigure"):
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import argparse
import json
import sqlite3
import time
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(ROOT))

# Jediný zdroj extrakční logiky — táž pipeline, kterou používá REST API.
from src.core.cascade import (  # noqa: E402
    _GT_ALIASES,
    _VAT_AUDIT,
    ENABLE_VAT_INC_CORRECTION,
    EXCLUDE_COMMODITIES,
    FIELD_REGISTRY,
    OPENAI_API_KEY,
    PRIMARY_MODEL,
    USD_TO_CZK,
    VAT_INC_TOLERANCE_CZK,
    VISION_MODEL,
    _fmt_amount,
    _fmt_date,
    _normalize_stem,
    _ref_date_for_vat,
    _to_float_or_none,
    build_json_schema,
    cascade_extract,
    check_arithmetic,
    check_formats,
    load_commodity_fields,
    normalize_value,
    pdf_to_ocr_text,
)
from src.core.vat_utils import correct_amount_inc_vat  # noqa: E402

DS1_ROOT = ROOT / "data" / "ds1"
OUTPUT_DIR = ROOT / "experiments" / "data" / "07_evaluace_ds1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DB_RESULTS_PATH = ROOT / "ds1_final_evaluation.db"
OUTPUT_TAG = "v3"

def load_ground_truth(ds_root: Path) -> pd.DataFrame:
    gt_files = [f for f in ds_root.glob("ground_truth_*.csv") if " copy" not in f.stem]
    if not gt_files:
        raise FileNotFoundError(f"Žádné ground_truth_*.csv v {ds_root}")
    dfs = []
    for f in gt_files:
        df = pd.read_csv(f, sep=None, engine="python", dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        df["_source_csv"] = f.stem
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def build_gt_record(gt_row: dict, fields: list[dict]) -> dict[str, str]:
    result: dict[str, str] = {}
    for fdef in fields:
        fname = fdef["field_name"]
        ftype = fdef.get("field_type", "string")
        raw   = gt_row.get(fname, gt_row.get(_GT_ALIASES.get(fname, ""), ""))
        if ftype == "float":
            result[fname] = _fmt_amount(raw)
        elif ftype == "date":
            result[fname] = _fmt_date(raw)
        else:
            result[fname] = (str(raw).strip()
                             if raw and not (isinstance(raw, float) and pd.isna(raw)) else "")
    return result


def collect_sample(ds_root: Path, gt_df: pd.DataFrame,
                   limit: int | None = None) -> list[dict]:
    # Klíč: (commodity, stem) — zabraňuje přepisování mezi komoditami.
    # Stejný název souboru existuje v plyn_mo i plyn_vo GT CSV; bez commodity
    # by posledně načtená CSV přepsala správnou GT pro dřívější komoditu.
    gt_lookup: dict[tuple[str, str], Any] = {}
    for _, row in gt_df.iterrows():
        comm_gt = str(row.get("commodity", "")).strip().lower()
        for key in ("scan_filename", "source_filename", "file_name", "filename", "pdf_name"):
            if key in row and pd.notna(row[key]):
                gt_lookup[(comm_gt, _normalize_stem(row[key]))] = row
                break

    by_comm: dict[str, list[dict]] = {}
    for pdf_path in ds_root.glob("*/*/Q*/*.pdf"):
        comm = pdf_path.parent.parent.parent.name
        if comm in EXCLUDE_COMMODITIES:
            continue
        stem = _normalize_stem(pdf_path.name)
        # Hledej GT pomocí (commodity, stem); fallback na ("", stem) pro GT bez commodity sloupce
        _r = gt_lookup.get((comm, stem))
        row = _r if _r is not None else gt_lookup.get(("", stem))
        by_comm.setdefault(comm, []).append({
            "pdf_path":  pdf_path,
            "pdf_name":  pdf_path.name,
            "quality":   pdf_path.parent.name,
            "supplier":  pdf_path.parent.parent.name,
            "commodity": comm,
            "has_gt":    row is not None,
            "gt_row":    dict(row) if row is not None else {},
        })

    selected = []
    for comm, lst in sorted(by_comm.items()):
        pool = lst if limit is None else sorted(lst, key=lambda x: x["quality"])[:limit]
        selected.extend(pool)
        print(f"  {comm}: {len(pool)} faktur (s GT: {sum(1 for x in pool if x['has_gt'])})")
    return selected


def compute_metrics(pred: dict, gt: dict, fields: list[dict]) -> dict:
    tp = fp = fn = 0
    per_field: dict[str, dict] = {}
    for fdef in fields:
        fname = fdef["field_name"]
        ftype = fdef.get("field_type", "string")
        gt_raw = gt.get(fname, gt.get(_GT_ALIASES.get(fname, ""), ""))
        gt_val = normalize_value(fname, gt_raw, ftype)
        pr_val = normalize_value(fname, pred.get(fname), ftype)
        if not gt_val:
            per_field[fname] = {"status": "no_gt", "gt": "", "pred": pr_val}
            continue
        if pr_val == gt_val:
            tp += 1
            per_field[fname] = {"status": "tp", "gt": gt_val, "pred": pr_val}
        elif pr_val:
            fp += 1
            per_field[fname] = {"status": "fp", "gt": gt_val, "pred": pr_val}
        else:
            fn += 1
            per_field[fname] = {"status": "fn", "gt": gt_val, "pred": ""}
    n_fields  = len(fields)
    null_rate = sum(1 for f in fields if not pred.get(f["field_name"])) / n_fields if n_fields else 1.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "null_rate": round(null_rate, 4),
        "gt_present": tp + fp + fn,
        "per_field":  per_field,
    }


DDL = """
CREATE TABLE IF NOT EXISTS ds1_final_evaluation (
    run_id TEXT, run_ts TEXT,
    strategy_id TEXT, strategy TEXT,
    model TEXT, model_label TEXT,
    escalated INTEGER, mode TEXT, escalation_reason TEXT,
    invoice_idx INTEGER, pdf_name TEXT,
    quality TEXT, supplier TEXT, commodity TEXT,
    prompt_tokens INTEGER, completion_tokens INTEGER, latency_ms REAL,
    tp INTEGER, fp INTEGER, fn INTEGER,
    precision REAL, recall REAL, f1 REAL,
    null_rate REAL, gt_present INTEGER,
    cost_usd REAL, cost_czk REAL,
    n_fields INTEGER,
    mean_logprob REAL,
    format_errors TEXT, arith_errors TEXT,
    confidence_json TEXT,
    extracted_json TEXT, gt_json TEXT,
    per_field_json TEXT, api_error TEXT
)
"""


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(DDL)
    # Schema migration: add columns introduced after the initial table creation
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(ds1_final_evaluation)")}
    _new_cols = {
        "mode":             "TEXT",
        "escalation_reason":"TEXT",
        "n_fields":         "INTEGER",
        "mean_logprob":     "REAL",
        "format_errors":    "TEXT",
        "arith_errors":     "TEXT",
        "confidence_json":  "TEXT",
        "per_field_json":   "TEXT",
        # Latency breakdown (Fix / report: OCR vs LLM, primary vs fallback)
        "ocr_ms":                  "REAL",
        "text_llm_ms":             "REAL",
        "vision_llm_ms":           "REAL",
        "full_pipeline_ms":        "REAL",
        # Fix 3 before/after audit fields
        "amount_inc_vat_raw":      "REAL",
        "amount_inc_vat_computed": "REAL",
        "vat_rate_used":           "REAL",
        "vat_inc_corrected":       "INTEGER",
        "aiv_gt_present":          "INTEGER",
        "aiv_match_raw":           "INTEGER",
        "aiv_match_corrected":     "INTEGER",
    }
    for col, typ in _new_cols.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE ds1_final_evaluation ADD COLUMN {col} {typ}")
    conn.commit()
    return conn


# ─────────────────────────────────────────────────────────────────────────────
# HLAVNÍ SMYČKA
# ─────────────────────────────────────────────────────────────────────────────

def run(limit: int | None = None, dry_run: bool = False,
        commodity_filter: str | None = None) -> None:
    print(f"\n{'='*68}")
    print("  DS1 FINAL EVALUATION — Kaskáda GPT-4.1 text → Vision mini")
    print(f"{'='*68}\n")

    if not OPENAI_API_KEY and not dry_run:
        print("CHYBA: OPENAI_API_KEY není nastaven v .env")
        _sys.exit(1)

    global EXCLUDE_COMMODITIES
    if commodity_filter:
        valid = set(FIELD_REGISTRY.keys())
        cf = commodity_filter.lower()
        if cf not in valid:
            print(f"Neznámá komodita: {cf}. Dostupné: {sorted(valid)}")
            _sys.exit(1)
        EXCLUDE_COMMODITIES = [c for c in valid if c != cf]

    results_conn = init_db(DB_RESULTS_PATH)

    print("Načítání ground truth z DS1...")
    gt_df = load_ground_truth(DS1_ROOT)
    print(f"  GT: {len(gt_df)} řádků")

    print("\nVýběr testovací sady DS1...")
    sample = collect_sample(DS1_ROOT, gt_df, limit=limit)
    sample_with_gt = [s for s in sample if s["has_gt"]]
    print(f"\nCelkem: {len(sample)} faktur  (s GT: {len(sample_with_gt)})")

    if dry_run:
        n = len(sample_with_gt)
        esc_rate = 0.15
        # Primary cost estimate: avg ~3500 input tokens with all fields
        avg_pt_prim = 3500
        cost_prim   = n * (avg_pt_prim * PRIMARY_MODEL["price_in"] +
                          300 * PRIMARY_MODEL["price_out"]) / 1e6
        # Vision fallback: img tokens + text tokens
        avg_pt_vis  = 5000   # includes 3 page images ~765 tokens each
        cost_vis    = int(n * esc_rate) * (avg_pt_vis * VISION_MODEL["price_in"] +
                          400 * VISION_MODEL["price_out"]) / 1e6
        total = cost_prim + cost_vis
        print(f"\nDRY RUN — odhad nákladů na {n} faktur:")
        print(f"  Primární (GPT-4.1): ${cost_prim:.2f}")
        print(f"  Eskalace Vision:    ${cost_vis:.2f}  (odhad {esc_rate*100:.0f}%)")
        print(f"  CELKEM:             ${total:.2f}  ({total*USD_TO_CZK:.0f} Kč)")
        print("\nPole per komodita:")
        for comm, flds in FIELD_REGISTRY.items():
            if comm not in EXCLUDE_COMMODITIES:
                print(f"  {comm}: {len(flds)} polí")
        return

    RUN_ID = str(uuid.uuid4())[:8]
    RUN_TS = datetime.now(UTC).isoformat()
    results: list[dict] = []

    print(f"\nBenchmark start  RUN_ID={RUN_ID}")
    print(f"PRIMARY:  {PRIMARY_MODEL['label']} · text json_light · few-shot RAG n=2")
    print(f"FALLBACK: {VISION_MODEL['label']} · Vision standard · text RAG n=2")
    print(f"Faktury s GT: {len(sample_with_gt)}\n")

    _t_start = time.time()
    field_cache: dict[str, list[dict]] = {}

    for idx, inv in enumerate(sample_with_gt, 1):
        name = inv["pdf_name"]
        comm = inv["commodity"]
        qual = inv["quality"]
        sup  = inv["supplier"]

        elapsed = time.time() - _t_start
        avg_s   = elapsed / idx if idx > 1 else 15
        remain  = int(avg_s * (len(sample_with_gt) - idx) / 60)
        print(f"  [{idx:4d}/{len(sample_with_gt)}]  {name[:45]}  (~{remain}min zbývá)", flush=True)

        # OCR — všechny strany dokumentu
        try:
            t_ocr = time.time()
            html  = pdf_to_ocr_text(inv["pdf_path"])
            ocr_s = time.time() - t_ocr
        except Exception as e:
            print(f"    [SKIP OCR] {e}", flush=True)
            continue

        # Fields for this commodity
        if comm not in field_cache:
            field_cache[comm] = load_commodity_fields(comm)
        fields = field_cache[comm]

        # Ground truth
        gt     = build_gt_record(inv["gt_row"], fields)
        schema = build_json_schema(fields)

        # Cascade
        res, escalated, cost, mode_used, esc_reason = cascade_extract(
            html, inv["pdf_path"], comm, fields, schema
        )

        # ── Fix 3: deterministic amount_inc_vat correction ───────────────────
        aiv_raw_f     = _to_float_or_none(res["extracted"].get("amount_inc_vat"))
        aex_f         = _to_float_or_none(res["extracted"].get("amount_ex_vat"))
        vat_rate_used = None
        aiv_computed  = None
        vat_corrected = False
        if ENABLE_VAT_INC_CORRECTION:
            ref_date = _ref_date_for_vat(res["extracted"])
            corrected_val, vat_corrected, vat_rate_used, aiv_computed = correct_amount_inc_vat(
                aex_f, aiv_raw_f, comm, ref_date, VAT_INC_TOLERANCE_CZK
            )
            if vat_corrected:
                res["extracted"]["amount_inc_vat"] = corrected_val
                _VAT_AUDIT.append({
                    "invoice_id":      name,
                    "commodity":       comm,
                    "extracted_value": aiv_raw_f,
                    "computed_value":  aiv_computed,
                    "vat_rate_used":   vat_rate_used,
                    "ref_date":        str(ref_date) if ref_date else "",
                })

        # amount_inc_vat exact-match vs GT, before (raw) and after (corrected)
        gt_aiv_norm    = normalize_value("amount_inc_vat", gt.get("amount_inc_vat"), "float")
        aiv_gt_present = bool(gt_aiv_norm)
        aiv_match_raw  = int(aiv_gt_present and
                             normalize_value("amount_inc_vat", aiv_raw_f, "float") == gt_aiv_norm)
        aiv_match_corr = int(aiv_gt_present and
                             normalize_value("amount_inc_vat", res["extracted"].get("amount_inc_vat"),
                                             "float") == gt_aiv_norm)

        # metrics computed on the corrected extraction (Fix 3 = ON is the final config)
        metrics     = compute_metrics(res["extracted"], gt, fields)
        f1_val      = metrics["f1"]
        fmt_errs    = check_formats(res["extracted"])
        arith_errs  = check_arithmetic(res["extracted"], comm)
        mean_lp     = res.get("mean_logprob")
        confidence  = res.get("confidence") or {}
        lp_str      = f"lp={mean_lp:.3f}" if mean_lp is not None else "lp=n/a"
        # Summarise self-confidence: show min confidence of required fields (if any scored)
        scored_confs = [v for v in confidence.values() if v is not None]
        conf_str    = f"conf_min={min(scored_confs):.2f}" if scored_confs else "conf=n/a"
        esc_tag     = f"VIS({esc_reason})" if escalated else "TXT"
        warn_str    = ""
        if fmt_errs:   warn_str += f"  FMT:{len(fmt_errs)}"
        if arith_errs: warn_str += f"  ARITH:{len(arith_errs)}"
        print(
            f"    OCR={ocr_s:.1f}s  F1={f1_val:.3f}  null={metrics['null_rate']:.2f}"
            f"  {conf_str}  {lp_str}  {esc_tag}  ${cost:.4f}{warn_str}",
            flush=True,
        )

        model_id = (
            f"{PRIMARY_MODEL['model_id']}+{VISION_MODEL['model_id']}"
            if escalated else PRIMARY_MODEL["model_id"]
        )
        row = {
            "run_id": RUN_ID, "run_ts": RUN_TS,
            "strategy_id": "S3v2", "strategy": "cascade_vision",
            "model": model_id, "model_label": f"Cascade({'vision' if escalated else 'text'})",
            "escalated": int(escalated), "mode": mode_used,
            "escalation_reason": esc_reason,
            "invoice_idx": idx, "pdf_name": name,
            "quality": qual, "supplier": sup, "commodity": comm,
            "prompt_tokens":      res["prompt_tokens"],
            "completion_tokens":  res["completion_tokens"],
            "latency_ms":         res["latency_ms"],
            # Latency breakdown: OCR vs LLM (primary text + fallback vision)
            "ocr_ms":             ocr_s * 1000.0,
            "text_llm_ms":        res.get("text_llm_ms", res["latency_ms"]),
            "vision_llm_ms":      res.get("vision_llm_ms", 0.0),
            "full_pipeline_ms":   ocr_s * 1000.0
                                  + res.get("text_llm_ms", res["latency_ms"])
                                  + res.get("vision_llm_ms", 0.0),
            # Fix 3 before/after audit fields
            "amount_inc_vat_raw":      aiv_raw_f,
            "amount_inc_vat_computed": aiv_computed,
            "vat_rate_used":           vat_rate_used,
            "vat_inc_corrected":       int(vat_corrected),
            "aiv_gt_present":          int(aiv_gt_present),
            "aiv_match_raw":           aiv_match_raw,
            "aiv_match_corrected":     aiv_match_corr,
            "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
            "precision":  metrics["precision"],
            "recall":     metrics["recall"],
            "f1":         metrics["f1"],
            "null_rate":  metrics["null_rate"],
            "gt_present": metrics["gt_present"],
            "cost_usd":   cost,
            "cost_czk":   cost * USD_TO_CZK,
            "n_fields":      len(fields),
            "mean_logprob":  mean_lp,
            "format_errors":  json.dumps(fmt_errs,   ensure_ascii=False) if fmt_errs   else None,
            "arith_errors":   json.dumps(arith_errs, ensure_ascii=False) if arith_errs else None,
            "confidence_json": json.dumps(confidence, ensure_ascii=False) if confidence else None,
            "extracted_json": json.dumps(res["extracted"], ensure_ascii=False),
            "gt_json":        json.dumps(gt, ensure_ascii=False),
            "per_field_json": json.dumps(metrics["per_field"], ensure_ascii=False),
            "api_error":      res["api_error"],
        }
        results.append(row)

        try:
            df_row = pd.DataFrame([{k: v for k, v in row.items() if k != "per_field_json"}])
            df_row["per_field_json"] = row["per_field_json"]
            df_row.to_sql("ds1_final_evaluation", results_conn, if_exists="append", index=False)
            results_conn.commit()
        except Exception:
            pass

    # ── SOUHRN ───────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    if df.empty:
        print("Žádné výsledky.")
        return

    print(f"\n{'='*68}")
    print("  VÝSLEDKY — DS1 Kaskáda GPT-4.1 text → Vision mini")
    print(f"{'='*68}")
    print(f"  Faktur zpracováno: {len(df)}")
    print(f"  Eskalace (Vision): {df['escalated'].sum()} ({df['escalated'].mean()*100:.1f}%)")
    print()

    summ = df.groupby("commodity").agg(
        F1=("f1", "mean"), Prec=("precision", "mean"), Rec=("recall", "mean"),
        Null=("null_rate", "mean"), N=("f1", "count"),
        Fields=("n_fields", "first"),
    ).round(4)
    print(summ.to_string())
    print()

    overall_f1   = df["f1"].mean()
    overall_pre  = df["precision"].mean()
    overall_rec  = df["recall"].mean()
    total_cost   = df["cost_usd"].sum()
    print(f"  Makro-F1:  {overall_f1:.4f}")
    print(f"  Precision: {overall_pre:.4f}")
    print(f"  Recall:    {overall_rec:.4f}")
    print(f"  Náklady:   ${total_cost:.2f}  ({total_cost*USD_TO_CZK:.0f} Kč)")

    # Per-quality breakdown
    if "quality" in df.columns:
        q_summ = df.groupby("quality").agg(F1=("f1", "mean"), N=("f1", "count")).round(4)
        print("\n  Dle kvality dokumentu:")
        print(q_summ.to_string())

    # Acceptance criteria check
    print(f"\n{'─'*68}")
    print("  OVĚŘENÍ AKCEPTAČNÍCH KRITÉRIÍ")
    print(f"{'─'*68}")
    print(f"  Krit. 1  F1 ≥ 0.60:         {overall_f1:.4f}  {'✓' if overall_f1 >= 0.60 else '✗'}")
    q3_f1 = df[df["quality"].str.upper() == "Q3"]["f1"].mean() if "quality" in df.columns else float("nan")
    print(f"  Krit. 5  Q3 F1 ≥ 0.45:      {q3_f1:.4f}  {'✓' if q3_f1 >= 0.45 else '✗'}")
    for comm in sorted(FIELD_REGISTRY.keys()):
        if comm in EXCLUDE_COMMODITIES:
            continue
        comm_f1 = df[df["commodity"] == comm]["f1"].mean() if comm in df["commodity"].values else float("nan")
        ok = "✓" if comm_f1 >= 0.50 else "✗"
        print(f"  Krit. 4  {comm:<16} F1 ≥ 0.50:  {comm_f1:.4f}  {ok}")

    # ── FIELD-LEVEL ACCURACY + LATENCY + JSON VALIDITY ───────────────────────
    from collections import defaultdict
    field_tp: dict[str, int] = defaultdict(int)
    field_present: dict[str, int] = defaultdict(int)
    for r in results:
        for fname, info in json.loads(r["per_field_json"]).items():
            st = info.get("status")
            if st in ("tp", "fp", "fn"):
                field_present[fname] += 1
                if st == "tp":
                    field_tp[fname] += 1

    def facc(f: str) -> float | None:
        return round(field_tp[f] / field_present[f], 4) if field_present.get(f) else None

    aiv_present = sum(r["aiv_gt_present"] for r in results)
    aiv_before  = round(sum(r["aiv_match_raw"]       for r in results) / aiv_present, 4) if aiv_present else None
    aiv_after   = round(sum(r["aiv_match_corrected"] for r in results) / aiv_present, 4) if aiv_present else None
    n_corrected = int(sum(r["vat_inc_corrected"] for r in results))

    ocr_ms_mean   = float(df["ocr_ms"].mean())
    full_ms_mean  = float(df["full_pipeline_ms"].mean())
    text_ms_mean  = float(df["text_llm_ms"].mean())
    _vis_rows     = df[df["escalated"] == 1]
    vision_ms_mean = float(_vis_rows["vision_llm_ms"].mean()) if len(_vis_rows) else 0.0
    jvr = round(float(df["api_error"].isna().mean()), 4)
    cost_per_doc = total_cost / len(df)

    # Acceptance criteria K1–K9 (thresholds per notebooks/eval_report.ipynb)
    def _val(x): return round(x, 4) if x is not None else None
    _crit = [
        ("K1", "Macro F1 ≥ 0.60",              overall_f1,               0.60, (overall_f1 or 0) >= 0.60),
        ("K2", "invoice_number acc ≥ 0.85",    facc("invoice_number"),   0.85, (facc("invoice_number") or 0) >= 0.85),
        ("K3", "supplier_tax_id acc ≥ 0.85",   facc("supplier_tax_id"),  0.85, (facc("supplier_tax_id") or 0) >= 0.85),
        ("K4", "customer_tax_id acc ≥ 0.85",   facc("customer_tax_id"),  0.85, (facc("customer_tax_id") or 0) >= 0.85),
        ("K5", "amount_inc_vat acc ≥ 0.85",    aiv_after,                0.85, (aiv_after or 0) >= 0.85),
        ("K6", "due_date acc ≥ 0.85",          facc("due_date"),         0.85, (facc("due_date") or 0) >= 0.85),
        ("K7", "JSON Validity Rate ≥ 0.98",    jvr,                      0.98, (jvr or 0) >= 0.98),
        ("K8", "Avg latency/doc ≤ 60 s",       full_ms_mean / 1000.0,    60.0, (full_ms_mean / 1000.0) <= 60.0),
        ("K9", "F1 Q3 ≥ 0.45",                 q3_f1,                    0.45, (q3_f1 or 0) >= 0.45),
    ]

    by_commodity = {
        rec["commodity"]: {"N": int(rec["N"]), "F1": rec["F1"],
                           "Precision": rec["Prec"], "Recall": rec["Rec"], "Null": rec["Null"]}
        for rec in summ.reset_index().to_dict(orient="records")
    }
    by_quality = {
        q: {"N": int((df["quality"] == q).sum()),
            "F1": round(float(df[df["quality"] == q]["f1"].mean()), 4)}
        for q in sorted(df["quality"].unique())
    }

    print(f"\n  amount_inc_vat accuracy  before Fix3: {aiv_before}  →  after: {aiv_after}  ({n_corrected} corrected)")
    print(f"  due_date accuracy: {facc('due_date')}")
    print(f"  Latency/doc — OCR: {ocr_ms_mean/1000:.2f}s  full pipeline: {full_ms_mean/1000:.2f}s")
    print(f"  JSON Validity Rate: {jvr*100:.1f}%")
    print("\n  Akceptační kritéria K1–K9:")
    for k, name, val, thr, ok in _crit:
        print(f"    {k}  {name:<28} {(f'{val:.4f}') if val is not None else 'n/a':>8}  {'PASS' if ok else 'FAIL'}")

    # ── EXPORT (v3 — new rerun; v2 baseline preserved) ───────────────────────
    csv_path   = OUTPUT_DIR / f"ds1_benchmark_results_{OUTPUT_TAG}.csv"
    json_path  = OUTPUT_DIR / f"ds1_benchmark_summary_{OUTPUT_TAG}.json"
    jsonl_path = OUTPUT_DIR / f"ds1_benchmark_full_{OUTPUT_TAG}.jsonl"
    audit_path = OUTPUT_DIR / f"vat_inc_corrections_{OUTPUT_TAG}.csv"

    df.drop(columns=["per_field_json"], errors="ignore").to_csv(csv_path, index=False)

    # Full rows incl. per_field_json for downstream analysis / figures
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Fix 3 audit log: every amount_inc_vat replacement
    pd.DataFrame(_VAT_AUDIT, columns=["invoice_id", "commodity", "extracted_value",
                                      "computed_value", "vat_rate_used", "ref_date"]
                 ).to_csv(audit_path, index=False)

    summary = {
        "benchmark":       "DS1 Final Evaluation — Cascade Vision v3 (Fix 1+2+3)",
        "run_id":          RUN_ID,
        "run_ts":          RUN_TS,
        "primary_model":   PRIMARY_MODEL["model_id"],
        "fallback_model":  VISION_MODEL["model_id"],
        "lie_format":      "json_light",
        "fix3_vat_inc_correction": ENABLE_VAT_INC_CORRECTION,
        "n_invoices":      len(df),
        "n_escalated":     int(df["escalated"].sum()),
        "escalation_rate": round(float(df["escalated"].mean()), 4),
        "macro_f1":        round(overall_f1, 4),
        "precision":       round(overall_pre, 4),
        "recall":          round(overall_rec, 4),
        "total_cost_usd":  round(total_cost, 4),
        "cost_per_doc_usd": round(cost_per_doc, 6),
        "latency": {
            "ocr_only_ms_mean":            round(ocr_ms_mean, 1),
            "full_pipeline_ms_mean":       round(full_ms_mean, 1),
            "text_llm_ms_mean":            round(text_ms_mean, 1),
            "vision_llm_ms_mean_escalated": round(vision_ms_mean, 1),
        },
        "json_validity_rate":              jvr,
        "amount_inc_vat_accuracy_before":  aiv_before,
        "amount_inc_vat_accuracy_after":   aiv_after,
        "amount_inc_vat_corrections":      n_corrected,
        "due_date_accuracy":               facc("due_date"),
        "field_accuracy":  {f: facc(f) for f in sorted(field_present)},
        "by_commodity":    by_commodity,
        "by_quality":      by_quality,
        "acceptance_criteria": [
            {"id": k, "criterion": name, "threshold": thr,
             "value": _val(val), "pass": bool(ok)}
            for k, name, val, thr, ok in _crit
        ],
        "per_commodity":   summ.reset_index().to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  CSV:   {csv_path}")
    print(f"  JSON:  {json_path}")
    print(f"  JSONL: {jsonl_path}")
    print(f"  AUDIT: {audit_path}")
    print(f"  DB:    {DB_RESULTS_PATH}")
    print(f"\n{'='*68}")
    print("  Hotovo.")
    print(f"{'='*68}\n")
    results_conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="DS1 Final Evaluation — Cascade Vision v2")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max faktur na komoditu (default: všechny)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Odhad nákladů bez API volání")
    ap.add_argument("--commodity", type=str, default=None,
                    help="Testuj jen jednu komoditu (např. plyn_mo)")
    args = ap.parse_args()
    run(limit=args.limit, dry_run=args.dry_run, commodity_filter=args.commodity)
