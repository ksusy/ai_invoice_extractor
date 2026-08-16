"""
FewShotRetriever — loads a FAISS index and retrieves similar invoice examples
for few-shot prompting. Built by build_index.py (run once).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INDEX_PATH = _ROOT / "data" / "retrieval" / "faiss.index"
DEFAULT_META_PATH  = _ROOT / "data" / "retrieval" / "metadata.json"

EMBED_MODEL = "text-embedding-3-small"
MAX_CHARS   = 1500  # chars to embed per query (same as index build)


class FewShotRetriever:
    """
    Retrieves the n most similar GT invoice examples for a given OCR text.

    Usage:
        retriever = FewShotRetriever()
        few_shot_text = retriever.get_examples(ocr_text, n=2)
        # inject few_shot_text into the LLM prompt
    """

    def __init__(
        self,
        index_path: Path = DEFAULT_INDEX_PATH,
        meta_path: Path = DEFAULT_META_PATH,
    ) -> None:
        try:
            import faiss
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "FewShotRetriever requires 'faiss-cpu' and 'openai'. "
                f"Install with: pip install faiss-cpu openai\n{e}"
            )

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {index_path}\n"
                "Run: python src/core/retrieval/build_index.py"
            )
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")

        self._faiss  = faiss
        self._client = OpenAI()
        self._index  = faiss.read_index(str(index_path))
        self._meta   = json.loads(meta_path.read_text(encoding="utf-8"))
        logger.info("FewShotRetriever ready: %d vectors", self._index.ntotal)

    def get_examples(self, ocr_text: str, n: int = 2,
                     exclude_stem: str | None = None) -> str:
        """
        Return a formatted few-shot string with n similar invoice examples,
        ready to inject directly into an LLM prompt. Returns "" on failure.

        exclude_stem: skip examples whose stem matches (case-insensitive) —
        prevents leaking the query invoice's own ground truth when the
        query document is itself part of the index.
        """
        if not ocr_text or not ocr_text.strip():
            return ""
        try:
            query_vec = self._embed(ocr_text[:MAX_CHARS])
            # over-fetch so we still have n examples after self-exclusion
            k = n + 2 if exclude_stem else n
            scores, ids = self._index.search(query_vec, k)
            excl = exclude_stem.lower().strip() if exclude_stem else None
            lines = []
            for idx, score in zip(ids[0], scores[0]):
                if idx < 0 or idx >= len(self._meta):
                    continue
                m = self._meta[idx]
                if excl and m.get("stem", "").lower().strip() == excl:
                    continue
                lines.append(
                    f"Example {len(lines) + 1} (similarity={score:.2f}): {m['example']}"
                )
                if len(lines) >= n:
                    break
            if not lines:
                return ""
            return (
                "Reference examples from similar correctly-extracted invoices:\n"
                + "\n".join(lines)
                + "\n"
            )
        except Exception:
            logger.warning("FewShotRetriever.get_examples failed", exc_info=True)
            return ""

    def _embed(self, text: str) -> np.ndarray:
        """Embed text, return normalized (1, D) float32 array."""
        resp = self._client.embeddings.create(model=EMBED_MODEL, input=[text])
        vec  = np.array([resp.data[0].embedding], dtype="float32")
        self._faiss.normalize_L2(vec)
        return vec
