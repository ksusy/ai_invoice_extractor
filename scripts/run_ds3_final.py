"""
Run the FINAL DS1 pipeline (scripts/run_ds1_final.py) on the DS3 generalization
set — same cascade (GPT-4.1 text -> gpt-4.1-mini vision, OCR json_light,
few-shot RAG), post-processing REMOVED (Fix3 off), evaluated on the common
field set present in the DS3 ground truth.

Thin wrapper: overrides the dataset-root / output-path module globals so the
DS1 artifacts are never touched, then calls run(). Results land in a separate
folder + DB.

    python scripts/run_ds3_final.py            # full run (162 docs, real API)
    python scripts/run_ds3_final.py --dry-run  # cost estimate only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import run_ds1_final as R  # noqa: E402  (same scripts/ dir)

ROOT = R.ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description="DS3 generalization — final pipeline")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--commodity", type=str, default=None)
    args = ap.parse_args()

    # --- redirect dataset + outputs to DS3 (DS1 artifacts untouched) ---
    R.DS1_ROOT   = ROOT / "data" / "ds3"
    R.OUTPUT_DIR = ROOT / "experiments" / "data" / "08_generalizace_ds3"
    R.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    R.DB_RESULTS_PATH = ROOT / "ds3_final_evaluation.db"
    R.OUTPUT_TAG = "ds3final"

    # --- post-processing removed: disable Fix3 VAT correction in-pipeline ---
    R.ENABLE_VAT_INC_CORRECTION = False

    print(f"[DS3] data root : {R.DS1_ROOT}")
    print(f"[DS3] output dir: {R.OUTPUT_DIR}")
    print(f"[DS3] results db: {R.DB_RESULTS_PATH}")
    print(f"[DS3] Fix3 VAT correction: {R.ENABLE_VAT_INC_CORRECTION}")

    R.run(limit=args.limit, dry_run=args.dry_run, commodity_filter=args.commodity)


if __name__ == "__main__":
    main()
