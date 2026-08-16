"""
One-time script: builds a FAISS retrieval index from DS1 + DS2 ground truth.

What it does:
  1. Loads all ground_truth_*.csv from data/ds1_clean/ and data/ds2/
  2. Runs Tesseract OCR on each corresponding PDF (cached to data/retrieval/ocr_cache.json)
  3. Embeds the first 1500 chars of each OCR text via text-embedding-3-small
  4. Saves a FAISS IndexFlatIP (cosine similarity) to data/retrieval/faiss.index
  5. Saves metadata (stem, commodity, example fields) to data/retrieval/metadata.json

Run from project root:
    python src/core/retrieval/build_index.py

Requirements:
    pip install faiss-cpu openai
    OPENAI_API_KEY must be set (or in .env)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Importy záměrně až zde — musí následovat po úpravě sys.path výše.
# ruff: noqa: E402

import numpy as np
import pandas as pd
import pytesseract
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(ROOT / ".env")

from src.core.ocr_engine.tesseract_setup import nastav_tesseract

nastav_tesseract()

# ── Paths ─────────────────────────────────────────────────────────────────────
DS1_DIR    = ROOT / "data" / "ds1"
DS2_DIR    = ROOT / "data" / "ds2"
INDEX_DIR  = ROOT / "data" / "retrieval"
INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH  = INDEX_DIR / "metadata.json"
OCR_CACHE  = INDEX_DIR / "ocr_cache.json"

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM   = 1536
OCR_CHARS   = 1500   # chars per invoice to embed
BATCH_SIZE  = 100    # texts per OpenAI embedding API call
MAX_PAGES   = 3      # max PDF pages to OCR per invoice

# Key GT fields to include in each few-shot example string
EXAMPLE_FIELDS = [
    "invoice_number",
    "period_from", "period_to",
    "amount_ex_vat", "amount_inc_vat",
    "supplier_tax_id", "customer_tax_id",
    "total_consumption",
    "consumption_point_code",
    "supplier",
]


# ── GT loading ────────────────────────────────────────────────────────────────

def load_all_gt() -> pd.DataFrame:
    """Merge ground truth CSVs from both datasets."""
    frames = []

    # DS1 — semicolon-delimited
    for csv_path in sorted(DS1_DIR.glob("ground_truth_*.csv")):
        commodity = csv_path.stem.replace("ground_truth_", "")
        df = pd.read_csv(csv_path, sep=";", dtype=str)
        df["_commodity"] = commodity
        df["_dataset"]   = "ds1"
        frames.append(df)

    # DS2 — comma-delimited
    for csv_path in sorted(DS2_DIR.glob("ground_truth_*.csv")):
        commodity = csv_path.stem.replace("ground_truth_", "")
        df = pd.read_csv(csv_path, dtype=str)
        df["_commodity"] = commodity
        df["_dataset"]   = "ds2"
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No ground_truth_*.csv files found in ds1_clean/ or ds2/")

    gt = pd.concat(frames, ignore_index=True)
    gt["_stem"] = gt["scan_filename"].fillna("").apply(
        lambda x: Path(x).stem.lower().strip()
    )
    return gt


def find_pdf(stem: str, dataset: str) -> Path | None:
    """Find the PDF file for a given stem in the correct dataset directory."""
    base = DS1_DIR if dataset == "ds1" else DS2_DIR
    matches = list(base.rglob(f"{stem}.pdf"))
    if not matches:
        # Case-insensitive fallback
        matches = [p for p in base.rglob("*.pdf") if p.stem.lower() == stem.lower()]
    return matches[0] if matches else None


def format_example(row: pd.Series) -> str:
    """Format one GT row into a compact few-shot string."""
    parts = []
    for field in EXAMPLE_FIELDS:
        val = row.get(field, "")
        if pd.notna(val) and str(val).strip() not in ("", "nan", "none"):
            parts.append(f"{field}={val}")
    commodity = row.get("_commodity", row.get("commodity", "unknown"))
    return f"[{commodity}] " + ", ".join(parts)


# ── OCR ───────────────────────────────────────────────────────────────────────

def ocr_pdf(pdf_path: Path) -> str:
    """OCR a PDF via PyMuPDF + pytesseract; return up to OCR_CHARS chars of plain text."""
    import cv2
    import fitz
    import numpy as np
    from PIL import Image

    doc = fitz.open(str(pdf_path))
    parts: list[str] = []
    for pg_num in range(min(doc.page_count, MAX_PAGES)):
        page = doc[pg_num]
        mat  = fitz.Matrix(200 / 72, 200 / 72)
        pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pil  = Image.fromarray(gray)
        text = pytesseract.image_to_string(pil, lang="ces+eng", config="--oem 3 --psm 6")
        parts.append(text.strip())
    doc.close()
    return "\n".join(parts)[:OCR_CHARS]


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_batch(client: OpenAI, texts: list[str]) -> np.ndarray:
    """Embed a list of texts, return (N, EMBED_DIM) float32 array."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [r.embedding for r in sorted(resp.data, key=lambda x: x.index)]
    return np.array(vecs, dtype="float32")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import faiss

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    # ── 1. Load GT ────────────────────────────────────────────────────────────
    print("Loading GT CSVs...")
    gt = load_all_gt()
    print(f"  {len(gt)} total GT rows")

    # ── 2. OCR phase ──────────────────────────────────────────────────────────
    ocr_cache: dict[str, str] = {}
    if OCR_CACHE.exists():
        ocr_cache = json.loads(OCR_CACHE.read_text(encoding="utf-8"))
        print(f"  OCR cache loaded: {len(ocr_cache)} entries")

    records = gt[gt["_stem"] != ""].to_dict("records")
    print(f"\nOCR phase: {len(records)} PDFs to process...")

    new_ocr = failed = 0
    for i, rec in enumerate(records):
        stem = rec["_stem"]
        if stem in ocr_cache:
            continue

        pdf = find_pdf(stem, rec["_dataset"])
        if pdf is None:
            print(f"  [SKIP] not found: {stem}")
            ocr_cache[stem] = ""
            failed += 1
            continue

        try:
            text = ocr_pdf(pdf)
            ocr_cache[stem] = text
            new_ocr += 1
        except Exception as exc:
            print(f"  [ERR] {stem}: {exc}")
            ocr_cache[stem] = ""
            failed += 1

        if (i + 1) % 25 == 0 or i == len(records) - 1:
            OCR_CACHE.write_text(
                json.dumps(ocr_cache, ensure_ascii=False), encoding="utf-8"
            )
            done_pct = (i + 1) / len(records) * 100
            print(f"  {i+1}/{len(records)} ({done_pct:.0f}%) — new OCR: {new_ocr}, failed: {failed}")

    OCR_CACHE.write_text(json.dumps(ocr_cache, ensure_ascii=False), encoding="utf-8")
    print(f"  OCR done. New: {new_ocr}, Failed: {failed}")

    # ── 3. Filter valid records ───────────────────────────────────────────────
    # Index DS2 ONLY: final evaluation runs on DS1, so DS1 examples in the index
    # would leak the test invoice's own ground truth into the prompt.
    valid_records = [
        r for r in records
        if r["_dataset"] == "ds2" and ocr_cache.get(r["_stem"], "").strip()
    ]
    print(f"\n  Valid DS2 records (DS1 excluded — anti-leakage): {len(valid_records)}/{len(records)}")

    if not valid_records:
        print("ERROR: no valid records to embed. Check OCR setup.")
        sys.exit(1)

    # ── 4. Build metadata + texts ─────────────────────────────────────────────
    texts    = [ocr_cache[r["_stem"]] for r in valid_records]
    metadata = [
        {
            "stem":      r["_stem"],
            "commodity": r.get("_commodity", ""),
            "supplier":  r.get("supplier", ""),
            "dataset":   r["_dataset"],
            "example":   format_example(pd.Series(r)),
        }
        for r in valid_records
    ]

    # ── 5. Embed in batches ───────────────────────────────────────────────────
    print(f"\nEmbedding phase: {len(texts)} texts in batches of {BATCH_SIZE}...")
    all_vecs: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        vecs  = embed_batch(client, batch)
        all_vecs.append(vecs)
        end = min(start + BATCH_SIZE, len(texts))
        print(f"  Embedded {end}/{len(texts)}")

    matrix = np.vstack(all_vecs)  # (N, EMBED_DIM)
    faiss.normalize_L2(matrix)    # in-place L2 norm → cosine similarity via dot product

    # ── 6. Build and save FAISS index ─────────────────────────────────────────
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(matrix)
    faiss.write_index(index, str(INDEX_PATH))
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nDone. Index: {index.ntotal} vectors")
    print(f"  {INDEX_PATH}")
    print(f"  {META_PATH}")
    print(f"  (OCR cache: {OCR_CACHE})")


if __name__ == "__main__":
    main()
