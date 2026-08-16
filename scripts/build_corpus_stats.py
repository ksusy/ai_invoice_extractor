#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Agregace referenčního vzorku dokumentů z KEM pro kapitolu 3 práce.

Skript projde lokální korpus faktur (``data/sorted/``) a uloží pouze agregované,
anonymizované statistiky — tedy žádné cesty k souborům, jména odběratelů ani
obsah faktur. Výstupy slouží jako vstup notebooku ``01_analyza_dat.ipynb``,
který je tak spustitelný i bez přístupu k samotným fakturám.

Výstupy (experiments/data/01_analyza_dat/):
    typy_dokumentu.csv     komodita × dodavatel × typ PDF -> počet dokumentů
    pocet_stran.csv        komodita -> počet stran (vzorek dle NB00, seed 2)
    laplace_variance.csv   komodita -> Laplaceova variance (ostrost skenu)

Spuštění (vyžaduje lokální korpus faktur):
    python scripts/build_corpus_stats.py
"""
from __future__ import annotations

import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SORTED_DIR = ROOT / "data" / "sorted"
# Pracovní cache typů PDF (obsahuje cesty k reálným fakturám, proto mimo repozitář).
PDF_TYPE_CACHE = ROOT / "_archive" / "exp_old" / "results" / "data" / "pdf_type_cache.csv"
OUT_DIR = ROOT / "experiments" / "data" / "01_analyza_dat"

SAMPLE_PAGES = 300   # PDF na komoditu (shodné s NB00)
SEED_PAGES = 2


def parts(path: str) -> list[str]:
    return re.split(r"[\\/]", str(path))


def build_document_types() -> pd.DataFrame:
    """komodita × dodavatel × typ PDF -> počet dokumentů (bez cest k souborům)."""
    cache = pd.read_csv(PDF_TYPE_CACHE, dtype=str)
    counts: Counter = Counter()
    for path, pdf_type in zip(cache["path"], cache["pdf_type"]):
        seg = parts(path)
        try:
            i = seg.index("sorted")
        except ValueError:
            continue
        # …/sorted/<komodita>/<davka>/<dodavatel>/<soubor>.pdf
        if len(seg) < i + 4:
            continue
        counts[(seg[i + 1], seg[i + 3], pdf_type)] += 1

    return (pd.DataFrame(
        [{"komodita": k, "dodavatel": s, "typ_pdf": t, "pocet": n}
         for (k, s, t), n in sorted(counts.items())])
        .sort_values(["komodita", "dodavatel", "typ_pdf"], ignore_index=True))


def build_page_counts() -> pd.DataFrame:
    """Počet stran u náhodného vzorku faktur na komoditu."""
    try:
        import pypdfium2 as pdfium
        def n_pages(p: Path) -> int | None:
            try:
                doc = pdfium.PdfDocument(str(p))
                n = len(doc)
                doc.close()
                return n
            except Exception:
                return None
    except ImportError:
        from pypdf import PdfReader
        def n_pages(p: Path) -> int | None:
            try:
                return len(PdfReader(str(p)).pages)
            except Exception:
                return None

    records = []
    for comm_dir in sorted(SORTED_DIR.iterdir()):
        if not comm_dir.is_dir() or comm_dir.name.startswith("_"):
            continue
        pdfs = sorted(comm_dir.rglob("*.pdf"))
        rng = random.Random(SEED_PAGES)
        if len(pdfs) > SAMPLE_PAGES:
            pdfs = rng.sample(pdfs, SAMPLE_PAGES)
        for pdf in pdfs:
            n = n_pages(pdf)
            if n:
                records.append({"komodita": comm_dir.name, "pocet_stran": n})
        print(f"  {comm_dir.name}: {len(pdfs)} PDF")
    return pd.DataFrame(records)


def main() -> None:
    if not SORTED_DIR.exists():
        sys.exit(f"Chybi lokalni korpus faktur: {SORTED_DIR}\n"
                 "Skript neni potreba spoustet — agregovane vystupy jsou v repozitari.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if PDF_TYPE_CACHE.exists():
        df = build_document_types()
        df.to_csv(OUT_DIR / "typy_dokumentu.csv", index=False, encoding="utf-8")
        print(f"typy_dokumentu.csv: {len(df)} radku, {df['pocet'].sum()} dokumentu")

    print("Pocty stran (vzorkovani dle NB00):")
    pages = build_page_counts()
    pages.to_csv(OUT_DIR / "pocet_stran.csv", index=False, encoding="utf-8")
    print(f"pocet_stran.csv: {len(pages)} dokumentu")


if __name__ == "__main__":
    main()
