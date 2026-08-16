"""Extrakce faktury kaskádovou pipeline vyhodnocenou v práci.

Tento koncový bod volá přímo :func:`src.core.cascade.extract_invoice`, tedy
tentýž kód, kterým bylo změřeno F1 = 0,85 na sadě DS1. Volání je synchronní —
zpracování trvá jednotky až desítky sekund a výsledek se vrací rovnou, bez
ukládání do databáze.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from src.domain.constants import SUPPORTED_FILE_EXTENSIONS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/extract", tags=["extract"])

KOMODITY = ["elektrina_nn", "elektrina_vn", "plyn_MO", "plyn_VO", "teplo", "voda"]


class ExtractResponse(BaseModel):
    """Výsledek extrakce jedné faktury."""

    filename: str
    commodity: str = Field(description="Použitá komodita (zadaná, nebo odhadnutá z textu)")
    fields: dict[str, str] = Field(description="Extrahovaná pole po normalizaci")
    confidence: dict[str, float] = Field(description="Jistota modelu po jednotlivých polích")
    mean_confidence: float = Field(description="Průměrná jistota — podklad pro kontrolu člověkem")
    escalated: bool = Field(description="Použila se záložní cesta přes Vision model?")
    mode: str = Field(description="text = hlavní cesta, vision = záložní")
    escalation_reason: str
    cost_usd: float
    ocr_ms: float
    llm_ms: float
    format_errors: list[str] = Field(description="Pole v nečekaném formátu (IČO, EAN, datum…)")
    arith_errors: list[str] = Field(description="Nesouhlasící součty a přepočty")
    warnings: list[str] = Field(default_factory=list)


@router.post("/", response_model=ExtractResponse)
async def extract_invoice_endpoint(
    file: Annotated[UploadFile, File(description="Faktura ve formátu PDF")],
    commodity: Annotated[str | None, Form(description=f"Jedna z: {', '.join(KOMODITY)}")] = None,
) -> ExtractResponse:
    """Zpracuje fakturu kaskádovou pipeline a vrátí strukturovaná data.

    Komoditu je vhodné zadat — v systému KEM ji uživatel při nahrání zná.
    Není-li uvedena, odhadne se z textu faktury, což u velkoodběru a vysokého
    napětí není spolehlivé.
    """
    nazev = file.filename or "unknown.pdf"
    if Path(nazev).suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Nepodporovaný formát. Podporováno: {sorted(SUPPORTED_FILE_EXTENSIONS)}",
        )
    if commodity is not None and commodity not in KOMODITY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Neznámá komodita {commodity!r}. Použijte jednu z: {', '.join(KOMODITY)}",
        )

    # Kaskáda pracuje se souborem na disku (OCR i Vision renderují stránky),
    # nahraný obsah se proto odloží do dočasného souboru.
    obsah = await file.read()
    with tempfile.TemporaryDirectory() as tmp:
        cesta = Path(tmp) / Path(nazev).name
        cesta.write_bytes(obsah)

        from src.core.cascade import extract_invoice

        try:
            vysledek = extract_invoice(cesta, commodity=commodity)
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e)) from e
        except Exception as e:
            logger.exception("Extrakce selhala pro %s", nazev)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Extrakce selhala: {e}",
            ) from e

    varovani: list[str] = []
    if commodity is None:
        varovani.append(
            "Komodita nebyla zadána a byla odhadnuta z textu faktury — ověřte ji ve výsledku."
        )
    if vysledek.api_error:
        varovani.append(f"Chyba volání modelu: {vysledek.api_error}")

    return ExtractResponse(
        filename=nazev,
        commodity=vysledek.commodity,
        fields=vysledek.fields,
        confidence=vysledek.confidence,
        mean_confidence=vysledek.mean_confidence,
        escalated=vysledek.escalated,
        mode=vysledek.mode,
        escalation_reason=vysledek.escalation_reason,
        cost_usd=round(vysledek.cost_usd, 6),
        ocr_ms=vysledek.ocr_ms,
        llm_ms=vysledek.llm_ms,
        format_errors=vysledek.format_errors,
        arith_errors=vysledek.arith_errors,
        warnings=varovani,
    )
