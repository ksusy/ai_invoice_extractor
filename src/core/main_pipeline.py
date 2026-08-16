"""Main processing pipeline orchestrator with benchmarking support.

Coordinates the invoice processing workflow:
    Ingest -> Classify -> OCR (if scanned) -> Extract -> Analyze -> Update DB -> API Response

Supports multi-strategy benchmarking for extraction comparison.

Hlavní orchestrátor zpracování faktur s podporou benchmarkingu.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.analysis import (
    AnalysisReport,
    create_correction_analyser,
    create_transitional_analyser,
)
from src.core.classification import DocumentKind, create_pdf_classifier
from src.core.extraction import (
    BaseExtractionStrategy,
    ExtractionContext,
    create_langchain_strategy,
    create_regex_strategy,
)
from src.core.ingestion import create_ingestion_service
from src.core.ocr_engine import OCRResult, create_tesseract_engine
from src.domain.entities import (
    CommodityType,
    ExtractionResult,
    InvoiceData,
    InvoiceType,
    clean_consumption_point_code,
    clean_tax_id,
)
from src.infrastructure.db.database import get_session_context
from src.infrastructure.db.models import (
    DBExtractionResult,
    ElectricityNNDetail,
    ElectricityVNDetail,
    GasMODetail,
    GasVODetail,
    HeatDetail,
    Invoice,
    Transaction,
    WaterDetail,
)
from src.infrastructure.db.models import (
    OCRResult as DBOCRResult,
)

logger = logging.getLogger(__name__)

ACCEPTANCE_THRESHOLD = 0.6


class ProcessingStatus(StrEnum):
    """Pipeline processing status values."""

    PENDING = "pending"
    CLASSIFYING = "classifying"
    CLASSIFIED = "classified"
    OCR_PROCESSING = "ocr_processing"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


class ProcessingResult:
    """Container for pipeline processing results."""

    def __init__(
        self,
        transaction_id: uuid.UUID,
        status: ProcessingStatus,
        document_kind: DocumentKind | None = None,
        ocr_result: OCRResult | None = None,
        extraction_result: ExtractionResult | None = None,
        invoice_id: uuid.UUID | None = None,
        error_message: str | None = None,
    ) -> None:
        self.transaction_id = transaction_id
        self.status = status
        self.document_kind = document_kind
        self.ocr_result = ocr_result
        self.extraction_result = extraction_result
        self.invoice_id = invoice_id
        self.error_message = error_message

    def __repr__(self) -> str:
        return (
            f"ProcessingResult(transaction_id={self.transaction_id}, "
            f"status={self.status.value}, document_kind={self.document_kind})"
        )

    @property
    def is_successful(self) -> bool:
        """Check if processing completed successfully."""
        return self.status == ProcessingStatus.COMPLETED


# ════════════════════════════════════════════════════════════════════════════
# BENCHMARKING DATA STRUCTURES
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class StrategyBenchmark:
    """Benchmark result for a single extraction strategy.

    Attributes:
        strategy_name: Name of the extraction strategy.
        extraction_result: Result from the strategy.
        latency_ms: Time taken for extraction in milliseconds.
        success: Whether extraction succeeded without errors.
        confidence: Confidence score (0.0 to 1.0).
        field_count: Number of fields successfully extracted.
    """

    strategy_name: str
    extraction_result: ExtractionResult | None = None
    latency_ms: int = 0
    success: bool = False
    confidence: float = 0.0
    field_count: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass
class BenchmarkComparison:
    """Comparison of multiple extraction strategies.

    Attributes:
        source_file: Source file being benchmarked.
        benchmarks: Results from each strategy.
        best_strategy: Name of best performing strategy.
        best_confidence: Highest confidence achieved.
        winner_by_fields: Strategy that extracted most fields.
        total_latency_ms: Sum of all strategy latencies.
        analysis_report: Optional analysis report (correction/transitional).
    """

    source_file: str
    benchmarks: list[StrategyBenchmark] = field(default_factory=list)
    best_strategy: str | None = None
    best_confidence: float = 0.0
    winner_by_fields: str | None = None
    total_latency_ms: int = 0
    analysis_report: AnalysisReport | None = None


# ════════════════════════════════════════════════════════════════════════════
# BENCHMARKING ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════


class BenchmarkOrchestrator:
    """Orchestrator for running multiple extraction strategies in parallel.

    Compares Regex vs LangChain extraction and provides detailed
    benchmarking metrics for each strategy.

    Usage:
        orchestrator = BenchmarkOrchestrator()
        comparison = await orchestrator.benchmark_text(ocr_text, "invoice.pdf")
    """

    def __init__(
        self,
        strategies: list[BaseExtractionStrategy] | None = None,
        run_analysis: bool = True,
    ) -> None:
        """Initialize benchmarking orchestrator.

        Args:
            strategies: List of strategies to benchmark.
                        Defaults to Regex + LangChain OpenAI.
            run_analysis: If True, run transitional/correction analysis.
        """
        self._strategies = strategies or self._get_default_strategies()
        self._run_analysis = run_analysis
        self._transitional_analyser = create_transitional_analyser() if run_analysis else None
        self._correction_analyser = create_correction_analyser() if run_analysis else None

    def _get_default_strategies(self) -> list[BaseExtractionStrategy]:
        """Get default set of strategies for benchmarking."""
        strategies = [create_regex_strategy()]

        # Try to add LangChain strategies (may fail if API keys not configured)
        try:
            strategies.append(create_langchain_strategy(provider="openai"))
        except Exception as e:
            logger.warning(f"Could not create OpenAI strategy: {e}")

        return strategies

    async def benchmark_text(
        self,
        text: str,
        source_filename: str,
        parallel: bool = True,
    ) -> BenchmarkComparison:
        """Benchmark all strategies on the same text.

        Args:
            text: OCR'd text to extract from.
            source_filename: Original source filename.
            parallel: If True, run strategies in parallel.

        Returns:
            BenchmarkComparison with results from all strategies.
        """
        context = ExtractionContext(
            raw_text=text,
            source_filename=source_filename,
        )

        benchmarks: list[StrategyBenchmark] = []

        if parallel:
            # Run all strategies concurrently
            tasks = [
                self._run_strategy_benchmark(strategy, context)
                for strategy in self._strategies
            ]
            benchmarks = await asyncio.gather(*tasks)
        else:
            # Run sequentially (for debugging)
            for strategy in self._strategies:
                benchmark = await self._run_strategy_benchmark(strategy, context)
                benchmarks.append(benchmark)

        # Determine winners
        best_strategy: str | None = None
        best_confidence: float = 0.0
        winner_by_fields: str | None = None
        max_fields: int = 0

        for bm in benchmarks:
            if bm.success:
                if bm.confidence > best_confidence:
                    best_confidence = bm.confidence
                    best_strategy = bm.strategy_name
                if bm.field_count > max_fields:
                    max_fields = bm.field_count
                    winner_by_fields = bm.strategy_name

        total_latency = sum(bm.latency_ms for bm in benchmarks)

        # Run analysis on best result
        analysis_report: AnalysisReport | None = None
        if self._run_analysis and best_strategy:
            best_result = next(
                (bm for bm in benchmarks if bm.strategy_name == best_strategy),
                None,
            )
            if best_result and best_result.extraction_result and best_result.extraction_result.invoice_data:
                analysis_report = await self._run_invoice_analysis(
                    best_result.extraction_result.invoice_data
                )

        return BenchmarkComparison(
            source_file=source_filename,
            benchmarks=benchmarks,
            best_strategy=best_strategy,
            best_confidence=best_confidence,
            winner_by_fields=winner_by_fields,
            total_latency_ms=total_latency,
            analysis_report=analysis_report,
        )

    async def _run_strategy_benchmark(
        self,
        strategy: BaseExtractionStrategy,
        context: ExtractionContext,
    ) -> StrategyBenchmark:
        """Run a single strategy and collect benchmark metrics.

        Args:
            strategy: Extraction strategy to run.
            context: Extraction context with text.

        Returns:
            StrategyBenchmark with results and timings.
        """
        start_time = time.time()

        try:
            result = await strategy.extract(context)
            latency_ms = int((time.time() - start_time) * 1000)

            field_count = 0
            if result.invoice_data:
                field_count = self._count_extracted_fields(result.invoice_data)

            return StrategyBenchmark(
                strategy_name=strategy.name,
                extraction_result=result,
                latency_ms=latency_ms,
                success=result.invoice_data is not None and not result.errors,
                confidence=result.confidence,
                field_count=field_count,
                errors=result.errors,
            )

        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error(f"Strategy {strategy.name} failed: {e}", exc_info=True)

            return StrategyBenchmark(
                strategy_name=strategy.name,
                latency_ms=latency_ms,
                success=False,
                errors=[str(e)],
            )

    def _count_extracted_fields(self, invoice: InvoiceData) -> int:
        """Count number of non-null fields in extracted invoice.

        Args:
            invoice: Extracted invoice data.

        Returns:
            Count of extracted fields.
        """
        count = 0

        # Core fields
        if invoice.invoice_number and invoice.invoice_number != "UNKNOWN":
            count += 1
        if invoice.variable_symbol:
            count += 1
        if invoice.supply_point.ean_code:
            count += 1
        if invoice.supply_point.eic_code:
            count += 1
        if invoice.period.period_from:
            count += 1
        if invoice.period.period_to:
            count += 1
        if invoice.issue_date:
            count += 1
        if invoice.due_date:
            count += 1
        if invoice.customer_tax_id:
            count += 1
        if invoice.supplier_tax_id:
            count += 1
        if invoice.total_amount_inc_vat:
            count += 1
        if invoice.total_amount_ex_vat:
            count += 1

        # Add commodity detail fields
        for detail in invoice.electricity_nn_details:
            if detail.consumption_low_tariff:
                count += 1
            if detail.consumption_high_tariff:
                count += 1
            if detail.distribution_tariff:
                count += 1

        for detail in invoice.electricity_vn_details:
            if detail.supply_consumption:
                count += 1
            if detail.annual_reserved_capacity:
                count += 1
            if detail.peak_consumption:
                count += 1

        for detail in invoice.gas_mo_details:
            if detail.consumption_m3:
                count += 1
            if detail.consumption_mwh:
                count += 1

        for detail in invoice.gas_vo_details:
            if detail.consumption_m3:
                count += 1
            if detail.consumption_mwh:
                count += 1

        for detail in invoice.water_details:
            if detail.consumption_m3:
                count += 1

        for detail in invoice.heat_details:
            if detail.consumption_gj:
                count += 1
            if detail.heat_consumption:
                count += 1

        return count

    async def _run_invoice_analysis(
        self,
        invoice: InvoiceData,
    ) -> AnalysisReport:
        """Run transitional and correction analysis on invoice.

        Args:
            invoice: Extracted invoice data.

        Returns:
            Combined AnalysisReport.
        """
        warnings: list[str] = []

        # Run transitional analysis
        if self._transitional_analyser:
            transitional_report = await self._transitional_analyser.analyse(invoice)
            warnings.extend(transitional_report.warnings)
            is_transitional = transitional_report.is_transitional
            cross_year = transitional_report.cross_year
        else:
            is_transitional = invoice.is_transitional
            cross_year = invoice.is_cross_year()

        # Run correction analysis
        if self._correction_analyser:
            correction_report = await self._correction_analyser.analyse(invoice)
            warnings.extend(correction_report.warnings)
            is_correction = correction_report.is_correction
            linked_invoice = correction_report.linked_invoice
        else:
            is_correction = invoice.is_correction
            linked_invoice = invoice.correction_info.original_invoice_number if invoice.correction_info else None

        return AnalysisReport(
            is_correction=is_correction,
            is_transitional=is_transitional,
            linked_invoice=linked_invoice,
            cross_year=cross_year,
            warnings=warnings,
        )

    def format_benchmark_report(self, comparison: BenchmarkComparison) -> str:
        """Format benchmark comparison as human-readable report.

        Args:
            comparison: Benchmark comparison result.

        Returns:
            Formatted string report.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("BENCHMARK REPORT: EXTRACTION STRATEGY COMPARISON")
        lines.append("=" * 70)
        lines.append(f"Source file: {comparison.source_file}")
        lines.append(f"Total latency: {comparison.total_latency_ms} ms")
        lines.append("")

        lines.append("STRATEGY RESULTS:")
        lines.append("-" * 50)

        for bm in comparison.benchmarks:
            status = "✓" if bm.success else "✗"
            lines.append(f"\n{status} {bm.strategy_name}")
            lines.append(f"   Latency: {bm.latency_ms} ms")
            lines.append(f"   Confidence: {bm.confidence:.2%}")
            lines.append(f"   Fields extracted: {bm.field_count}")
            if bm.errors:
                lines.append(f"   Errors: {', '.join(bm.errors[:2])}")

        lines.append("")
        lines.append("WINNER SUMMARY:")
        lines.append("-" * 50)
        lines.append(f"Best by confidence: {comparison.best_strategy} ({comparison.best_confidence:.2%})")
        lines.append(f"Best by field count: {comparison.winner_by_fields}")

        if comparison.analysis_report:
            lines.append("")
            lines.append("INVOICE ANALYSIS:")
            lines.append("-" * 50)
            ar = comparison.analysis_report
            if ar.is_correction:
                lines.append("Type: OPRAVNÁ FAKTURA (correction)")
                if ar.linked_invoice:
                    lines.append(f"Corrects: {ar.linked_invoice}")
            elif ar.is_transitional:
                lines.append("Type: PŘECHODOVÁ FAKTURA (transitional)")
            else:
                lines.append("Type: Běžná faktura (regular)")

            if ar.cross_year:
                lines.append("Cross-year: YES (spans multiple calendar years)")

            if ar.warnings:
                lines.append("Warnings:")
                for w in ar.warnings:
                    lines.append(f"  - {w}")

        lines.append("=" * 70)
        return "\n".join(lines)


class ProcessingOrchestrator:
    """Main pipeline orchestrator that coordinates all processing stages.

    This class implements the facade pattern to provide a simple interface
    for the complete invoice processing workflow.

    Workflow stages:
        1. Ingest: Accept file, save to data/raw/, create Transaction record
        2. Classify: Determine if PDF is native or scanned
        3. OCR: Run OCR if scanned, or extract text layer if native
        4. Extract: Extract structured data from text using strategies
        5. Analyse: Run transitional / correction analysis
        6. Save: Persist extracted data to commodity-specific tables
        7. Complete: Update final status in database
    """

    def __init__(
        self,
        session: AsyncSession | None = None,
        extraction_strategy: BaseExtractionStrategy | None = None,
        vision_strategy: BaseExtractionStrategy | None = None,
    ) -> None:
        """Initialize the orchestrator with optional session injection.

        Args:
            session: Async database session. If None, sessions are created
                     per operation using context managers.
            extraction_strategy: Primary text-based extraction strategy.
                                 Defaults to regex strategy.
            vision_strategy: Optional vision LLM fallback used when text
                             extraction confidence is below ACCEPTANCE_THRESHOLD.
        """
        self._session = session
        self._ingestion_service = create_ingestion_service(session)
        self._classifier = create_pdf_classifier(session)
        self._ocr_engine = create_tesseract_engine()
        self._extraction_strategy = extraction_strategy or create_regex_strategy()
        self._vision_strategy = vision_strategy
        self._transitional_analyser = create_transitional_analyser()
        self._correction_analyser = create_correction_analyser()

    # ── session helper ───────────────────────────────────────────

    @asynccontextmanager
    async def _get_session(self):
        """Yield the injected session or create a short-lived one."""
        if self._session:
            yield self._session
        else:
            async with get_session_context() as session:
                yield session

    async def _pdf_to_images(self, file_bytes: bytes) -> bytes | None:
        """Render PDF pages as a single stacked PNG for vision LLM fallback.

        Applies the same grayscale-normalize + NLM denoise preprocessing used on
        the primary OCR path (validated NB07 vision config: gpt-4.1-mini +
        grayscale/denoise) before handing the image to the Vision API.
        """
        try:
            import fitz
            from PIL import Image as PILImage

            from src.core.ocr_engine.tesseract_engine import TesseractEngine

            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page_images: list[PILImage.Image] = []
            for i in range(min(len(doc), 4)):
                pix = doc[i].get_pixmap(matrix=fitz.Matrix(1.5, 1.5), colorspace=fitz.csRGB)
                img = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
                # Same preprocessing as the primary OCR path (grayscale + denoise)
                page_images.append(TesseractEngine._preprocess_for_ocr(img))
            doc.close()
            if not page_images:
                return None
            try:
                import cv2
                import numpy as np
                arrs = [np.array(img) for img in page_images]  # grayscale (H, W)
                max_w = max(a.shape[1] for a in arrs)
                padded = [
                    cv2.copyMakeBorder(a, 0, 0, 0, max_w - a.shape[1], cv2.BORDER_CONSTANT, value=255)
                    for a in arrs
                ]
                combined = np.vstack(padded)
                _, buf = cv2.imencode(".png", combined)
                return buf.tobytes()
            except ImportError:
                import io as _io
                buf = _io.BytesIO()
                page_images[0].save(buf, format="PNG")
                return buf.getvalue()
        except Exception as e:
            logger.warning("PDF render for vision fallback failed: %s", e)
            return None

    async def _run_iterative_extraction(
        self,
        text: str,
        source_filename: str,
        file_bytes: bytes | None = None,
    ) -> ExtractionResult:
        """Iteration 1: text LLM. Iteration 2: vision LLM if confidence < ACCEPTANCE_THRESHOLD."""
        context = ExtractionContext(raw_text=text, source_filename=source_filename)
        result = await self._extraction_strategy.extract(context)

        if result.confidence >= ACCEPTANCE_THRESHOLD or self._vision_strategy is None:
            return result

        logger.info(
            "Confidence %.2f < %.2f — retrying with vision LLM for %s",
            result.confidence, ACCEPTANCE_THRESHOLD, source_filename,
        )
        image_bytes: bytes | None = None
        if file_bytes:
            image_bytes = await self._pdf_to_images(file_bytes)

        vision_context = ExtractionContext(
            raw_text=text,
            source_filename=source_filename,
            image_bytes=image_bytes,
        )
        vision_result = await self._vision_strategy.extract(vision_context)

        if vision_result.invoice_data and vision_result.confidence > result.confidence:
            logger.info(
                "Vision fallback improved confidence %.2f → %.2f for %s",
                result.confidence, vision_result.confidence, source_filename,
            )
            return vision_result
        return result

    async def process_file(
        self,
        source: str | bytes | Path,
        original_filename: str | None = None,
    ) -> ProcessingResult:
        """Process a single file through the complete pipeline.

        Args:
            source: File path or raw bytes.
            original_filename: Original filename (required for bytes input).

        Returns:
            ProcessingResult with final status and metadata.

        When no session was injected at construction, a **single** session
        is created for the entire pipeline run to guarantee atomicity.
        """
        if self._session:
            # API flow — session already provided by FastAPI dependency
            return await self._process_file_inner(source, original_filename)

        # CLI flow — create one session for the whole pipeline so that
        # all components share it and avoid cross-session deadlocks.
        async with get_session_context() as session:
            old_sessions = (
                self._session,
                self._ingestion_service._session,
                self._classifier._session,
            )
            self._session = session
            self._ingestion_service._session = session
            self._classifier._session = session
            try:
                return await self._process_file_inner(source, original_filename)
            finally:
                (
                    self._session,
                    self._ingestion_service._session,
                    self._classifier._session,
                ) = old_sessions

    async def _process_file_inner(
        self,
        source: str | bytes | Path,
        original_filename: str | None = None,
    ) -> ProcessingResult:
        """Core pipeline logic — expects self._session to be set."""
        transaction: Transaction | None = None

        try:
            # Stage 1: Ingest
            transaction = await self._ingestion_service.ingest_and_save(
                source=source,
                original_filename=original_filename,
            )

            # Stage 2: Classify
            await self._update_status(transaction.id, ProcessingStatus.CLASSIFYING)
            doc_kind = await self._classifier.classify_transaction(transaction)

            # Stage 3: OCR / Text Extraction
            await self._update_status(transaction.id, ProcessingStatus.OCR_PROCESSING)
            ocr_result = await self._run_ocr_stage(transaction, doc_kind)

            if not ocr_result.is_successful:
                await self._update_status(
                    transaction.id,
                    ProcessingStatus.FAILED,
                    error_message=f"OCR failed: {ocr_result.error_message}",
                )
                return ProcessingResult(
                    transaction_id=transaction.id,
                    status=ProcessingStatus.FAILED,
                    document_kind=doc_kind,
                    ocr_result=ocr_result,
                    error_message=ocr_result.error_message,
                )

            # Save OCR result to database
            ocr_db_id = await self._save_ocr_result(transaction.id, ocr_result)

            # Stage 4: Extraction (text LLM, with vision fallback if confidence < threshold)
            await self._update_status(transaction.id, ProcessingStatus.EXTRACTING)
            extraction_start = time.time()
            _file_bytes: bytes | None = None
            if transaction.file_path:
                with suppress(Exception):
                    _file_bytes = Path(transaction.file_path).read_bytes()
            try:
                extraction_result = await asyncio.wait_for(
                    self._run_iterative_extraction(
                        ocr_result.full_text,
                        transaction.filename,
                        file_bytes=_file_bytes,
                    ),
                    timeout=180.0,
                )
            except TimeoutError:
                extraction_result = ExtractionResult(
                    source_file=transaction.filename,
                    strategy_name=self._extraction_strategy.name,
                    errors=["Extraction timed out after 180s"],
                )
            extraction_duration_ms = (time.time() - extraction_start) * 1000

            # Save extraction result to database
            await self._save_extraction_result(
                transaction.id,
                extraction_result,
                ocr_result_id=ocr_db_id,
                duration_ms=extraction_duration_ms,
            )

            # Check for extraction errors
            if extraction_result.errors and not extraction_result.invoice_data:
                await self._update_status(
                    transaction.id,
                    ProcessingStatus.FAILED,
                    error_message=f"Extraction failed: {'; '.join(extraction_result.errors)}",
                )
                return ProcessingResult(
                    transaction_id=transaction.id,
                    status=ProcessingStatus.FAILED,
                    document_kind=doc_kind,
                    ocr_result=ocr_result,
                    extraction_result=extraction_result,
                    error_message="; ".join(extraction_result.errors),
                )

            # Stage 5: Analysis (transitional / correction)
            if extraction_result.invoice_data:
                await self._update_status(transaction.id, ProcessingStatus.ANALYZING)
                await self._run_analysis_stage(extraction_result.invoice_data)

            # Stage 6: Save to commodity tables
            invoice_id = None
            if extraction_result.invoice_data:
                invoice_id = await self._save_invoice_data(
                    transaction.id,
                    extraction_result.invoice_data,
                    extraction_confidence=extraction_result.confidence,
                )

                # Update commodity in transaction
                async with self._get_session() as session:
                    stmt = (
                        update(Transaction)
                        .where(Transaction.id == transaction.id)
                        .values(commodity=extraction_result.invoice_data.commodity.value)
                    )
                    await session.execute(stmt)

            # Stage 7: Mark as completed
            await self._update_status(transaction.id, ProcessingStatus.COMPLETED)

            return ProcessingResult(
                transaction_id=transaction.id,
                status=ProcessingStatus.COMPLETED,
                document_kind=doc_kind,
                ocr_result=ocr_result,
                extraction_result=extraction_result,
                invoice_id=invoice_id,
            )

        except Exception as e:
            error_msg = str(e)
            if transaction:
                # Save a failed extraction result so the error is recorded in extraction_results
                try:
                    failed_result = ExtractionResult(
                        source_file=original_filename or "",
                        strategy_name=self._extraction_strategy.name,
                        errors=[error_msg],
                    )
                    await self._save_extraction_result(
                        transaction.id, failed_result,
                    )
                except Exception:
                    logger.exception("Failed to save error extraction result for %s", transaction.id)

                await self._update_status(
                    transaction.id,
                    ProcessingStatus.ERROR,
                    error_message=error_msg,
                )
                return ProcessingResult(
                    transaction_id=transaction.id,
                    status=ProcessingStatus.ERROR,
                    error_message=error_msg,
                )
            raise

    async def _run_ocr_stage(
        self,
        transaction: Transaction,
        doc_kind: DocumentKind,
    ) -> OCRResult:
        """Run OCR or text extraction based on document type.

        Args:
            transaction: Transaction record with file path.
            doc_kind: Classification result (native/scanned).

        Returns:
            OCRResult with extracted text.
        """
        if not transaction.file_path:
            return OCRResult(
                full_text="",
                engine_name="error",
                error_message="No file path in transaction",
            )

        file_path = Path(transaction.file_path)
        if not file_path.exists():
            return OCRResult(
                full_text="",
                engine_name="error",
                error_message=f"File not found: {transaction.file_path}",
            )

        file_bytes = file_path.read_bytes()

        if doc_kind == DocumentKind.NATIVE_PDF:
            # Extract text directly from PDF layer
            return await self._ocr_engine.extract_native_text(file_bytes)
        else:
            # Run OCR on scanned document
            if file_path.suffix.lower() == ".pdf":
                return await self._ocr_engine.recognize_pdf(file_bytes)
            else:
                return await self._ocr_engine.recognize(file_bytes)

    async def _run_extraction_stage(
        self,
        text: str,
        source_filename: str,
    ) -> ExtractionResult:
        """Run extraction strategy on OCR text.

        Args:
            text: Raw OCR text.
            source_filename: Original source file name.

        Returns:
            ExtractionResult with parsed data.
        """
        context = ExtractionContext(
            raw_text=text,
            source_filename=source_filename,
        )
        return await self._extraction_strategy.extract(context)

    async def _run_analysis_stage(self, invoice: InvoiceData) -> AnalysisReport:
        """Run transitional and correction analysis, mutating invoice flags.

        Args:
            invoice: Extracted invoice data (will be mutated in-place).

        Returns:
            Combined AnalysisReport.
        """
        warnings: list[str] = []

        # Transitional analysis
        transitional_report = await self._transitional_analyser.analyse(invoice)
        warnings.extend(transitional_report.warnings)
        if transitional_report.is_transitional:
            invoice.is_transitional = True
            invoice.invoice_type = InvoiceType.TRANSITIONAL

            # Split commodity details proportionally by year
            if transitional_report.cross_year and invoice.period:
                split_result = self._transitional_analyser.calculate_split(invoice)
                if split_result.is_cross_year and split_result.splits:
                    year_invoices = self._transitional_analyser.split_commodity_details(
                        invoice, split_result
                    )
                    # Replace original single-period details with per-year splits
                    self._replace_details_with_splits(invoice, year_invoices)

        # Correction analysis
        correction_report = await self._correction_analyser.analyse(invoice)
        warnings.extend(correction_report.warnings)
        if correction_report.is_correction:
            invoice.is_correction = True
            invoice.invoice_type = InvoiceType.CORRECTION

        return AnalysisReport(
            is_correction=correction_report.is_correction,
            is_transitional=transitional_report.is_transitional,
            linked_invoice=correction_report.linked_invoice,
            cross_year=transitional_report.cross_year,
            warnings=warnings,
        )

    @staticmethod
    def _replace_details_with_splits(
        invoice: InvoiceData,
        year_invoices: dict[int, InvoiceData],
    ) -> None:
        """Replace original single-period details with per-year split details.

        Merges details from all year splits into the main invoice so that
        each year gets its own detail record with proportional consumption.
        """
        invoice.electricity_nn_details = []
        invoice.electricity_vn_details = []
        invoice.gas_mo_details = []
        invoice.gas_vo_details = []
        invoice.water_details = []
        invoice.heat_details = []

        for _year, yi in sorted(year_invoices.items()):
            invoice.electricity_nn_details.extend(yi.electricity_nn_details)
            invoice.electricity_vn_details.extend(yi.electricity_vn_details)
            invoice.gas_mo_details.extend(yi.gas_mo_details)
            invoice.gas_vo_details.extend(yi.gas_vo_details)
            invoice.water_details.extend(yi.water_details)
            invoice.heat_details.extend(yi.heat_details)

    async def _save_ocr_result(
        self,
        transaction_id: uuid.UUID,
        ocr_result: OCRResult,
    ) -> uuid.UUID:
        """Save OCR result to database and return its ID."""
        db_ocr = DBOCRResult(
            transaction_id=transaction_id,
            engine_name=ocr_result.engine_name,
            raw_text=ocr_result.full_text,
            confidence=ocr_result.confidence,
            duration_ms=ocr_result.latency_ms,
            error_message=ocr_result.error_message,
        )

        async with self._get_session() as session:
            session.add(db_ocr)
            await session.flush()
            return db_ocr.id

    async def _save_extraction_result(
        self,
        transaction_id: uuid.UUID,
        extraction_result: ExtractionResult,
        ocr_result_id: uuid.UUID | None = None,
        duration_ms: float = 0.0,
    ) -> None:
        """Save extraction result to database."""
        extracted_data = None
        if extraction_result.invoice_data:
            extracted_data = extraction_result.invoice_data.model_dump(mode="json")

        error_message = None
        if extraction_result.errors:
            error_message = "; ".join(extraction_result.errors)

        # Determine model name from strategy
        model_name = getattr(self._extraction_strategy, 'model_name', None)
        if not model_name:
            # Fallback: parse from strategy name
            strategy = extraction_result.strategy_name
            if "openai" in strategy or "gpt" in strategy:
                parts = strategy.split("_")
                model_name = parts[-1] if len(parts) > 2 else "gpt-4o-mini"
            elif "anthropic" in strategy or "claude" in strategy:
                parts = strategy.split("_")
                model_name = parts[-1] if len(parts) > 2 else "claude-3-haiku"

        db_extraction = DBExtractionResult(
            transaction_id=transaction_id,
            ocr_result_id=ocr_result_id,
            strategy_name=extraction_result.strategy_name,
            model_name=model_name,
            extracted_json=extracted_data or {},
            confidence=extraction_result.confidence,
            duration_ms=duration_ms,
            cost_usd=extraction_result.cost_usd,
            token_count=extraction_result.token_count,
            error_message=error_message,
        )

        async with self._get_session() as session:
            session.add(db_extraction)
            await session.flush()

    async def _save_invoice_data(
        self,
        transaction_id: uuid.UUID,
        invoice_data: InvoiceData,
        extraction_confidence: float | None = None,
    ) -> uuid.UUID:
        """Save extracted invoice data to the invoices and commodity tables.

        Args:
            transaction_id: Parent transaction UUID.
            invoice_data: Parsed invoice data.

        Returns:
            UUID of the created Invoice record.
        """
        # ── Universal normalization (all commodities) ────────────────
        # 1. Clean consumption point code
        try:
            cp_raw = invoice_data.supply_point.consumption_point_code or None
            cp_clean = clean_consumption_point_code(cp_raw)
            invoice_data.supply_point.consumption_point_code = cp_clean or ""

            # Commodity-specific format validation for confidence penalty
            if cp_clean and extraction_confidence is not None:
                format_ok = True
                commodity = invoice_data.commodity

                if commodity in (CommodityType.ELEKTRINA_NN, CommodityType.ELEKTRINA_VN):
                    # Czech EAN: 18 digits starting with 859
                    if not cp_clean.startswith("859") or len(cp_clean) != 18:
                        format_ok = False
                # Czech gas EIC: 16 chars starting with 27ZG
                elif commodity in (CommodityType.PLYN_MO, CommodityType.PLYN_VO) and (
                    not cp_clean.startswith("27") or len(cp_clean) < 16
                ):
                    format_ok = False

                if not format_ok:
                    penalty = 0.15
                    extraction_confidence = max(0.0, extraction_confidence - penalty)
                    logger.warning(
                        "Transaction %s: kod_odberne_misto '%s' -> '%s' does not match expected "
                        "format for %s (confidence -%.2f -> %.2f)",
                        transaction_id, cp_raw, cp_clean,
                        commodity.value, penalty, extraction_confidence,
                    )
        except Exception:
            logger.exception("Failed to normalise consumption_point_code for transaction %s", transaction_id)

        # 2. Clean tax identifiers
        try:
            invoice_data.customer_tax_id = clean_tax_id(invoice_data.customer_tax_id)
            invoice_data.supplier_tax_id = clean_tax_id(invoice_data.supplier_tax_id)
        except Exception:
            logger.exception("Failed to normalise tax IDs for transaction %s", transaction_id)

        # Determine best supply point code (prefer consumption_point_code, then ean, then eic)
        supply_point_code = (
            invoice_data.supply_point.consumption_point_code
            or invoice_data.supply_point.ean_code
            or invoice_data.supply_point.eic_code
            or None
        )

        # Create main invoice record
        invoice = Invoice(
            transaction_id=transaction_id,
            source_filename=invoice_data.source_filename,
            invoice_number=invoice_data.invoice_number,
            supply_point_code=supply_point_code,
            commodity=invoice_data.commodity.value,
            period_from=invoice_data.period.period_from if invoice_data.period else None,
            period_to=invoice_data.period.period_to if invoice_data.period else None,
            issue_date=invoice_data.issue_date,
            due_date=invoice_data.due_date,
            tax_point_date=invoice_data.vat_date,
            customer_cin=invoice_data.customer_tax_id,
            supplier_cin=invoice_data.supplier_tax_id,
            total_amount_ex_vat=invoice_data.total_amount_ex_vat,
            total_amount_inc_vat=invoice_data.total_amount_inc_vat,
            vat_rate=invoice_data.vat_rate,
            extraction_strategy=self._extraction_strategy.name,
            extraction_confidence=extraction_confidence,
            raw_extracted_json=invoice_data.model_dump(mode="json"),
        )

        async def save_invoice(session: AsyncSession) -> uuid.UUID:
            # Check for duplicate invoice (same invoice_number + commodity + supplier)
            dup_query = select(Invoice.id).where(
                Invoice.invoice_number == invoice.invoice_number,
                Invoice.commodity == invoice.commodity,
                Invoice.supplier_cin == invoice.supplier_cin,
            ).limit(1)
            existing = (await session.execute(dup_query)).scalar_one_or_none()
            if existing:
                logger.warning(
                    "Duplicate invoice detected: number=%s, commodity=%s, supplier_cin=%s "
                    "(existing id=%s). Skipping save, returning existing.",
                    invoice.invoice_number, invoice.commodity,
                    invoice.supplier_cin, existing,
                )
                return existing

            session.add(invoice)
            await session.flush()

            # Save commodity-specific details
            for detail in invoice_data.electricity_nn_details:
                db_detail = ElectricityNNDetail(
                    invoice_id=invoice.id,
                    period_from=detail.period.period_from if detail.period else None,
                    period_to=detail.period.period_to if detail.period else None,
                    consumption_low_tariff=detail.consumption_low_tariff,
                    consumption_high_tariff=detail.consumption_high_tariff,
                    total_consumption=detail.total_consumption,
                    meter_reading_start=detail.meter_reading_start,
                    meter_reading_end=detail.meter_reading_end,
                    distribution_tariff=detail.distribution_tariff,
                    circuit_breaker_value=detail.circuit_breaker_value,
                    amount_ex_vat=detail.amount_ex_vat,
                    amount_inc_vat=detail.amount_inc_vat,
                    supply_charge=detail.supply_charge,
                    distribution_charge=detail.distribution_charge,
                    system_services=detail.system_services,
                    renewable_energy_fee=detail.renewable_energy_fee,
                )
                session.add(db_detail)

            for detail in invoice_data.electricity_vn_details:
                db_detail = ElectricityVNDetail(
                    invoice_id=invoice.id,
                    period_from=detail.period.period_from if detail.period else None,
                    period_to=detail.period.period_to if detail.period else None,
                    supply_consumption=detail.supply_consumption,
                    peak_consumption=detail.peak_consumption,
                    off_peak_consumption=detail.off_peak_consumption,
                    supply_charge=detail.supply_charge,
                    supply_tax_charge=detail.supply_tax_charge,
                    quarter_hour_max=detail.quarter_hour_max,
                    eru_rate=detail.eru_rate,
                    annual_reserved_capacity=detail.annual_reserved_capacity,
                    annual_reserved_capacity_charge=detail.annual_reserved_capacity_charge,
                    monthly_reserved_capacity=detail.monthly_reserved_capacity,
                    monthly_reserved_capacity_charge=detail.monthly_reserved_capacity_charge,
                    grid_usage_rate=detail.grid_usage_rate,
                    grid_usage_charge=detail.grid_usage_charge,
                    reserved_capacity_excess=detail.reserved_capacity_excess,
                    reserved_capacity_excess_rate=detail.reserved_capacity_excess_rate,
                    reserved_capacity_excess_charge=detail.reserved_capacity_excess_charge,
                    power_factor=detail.power_factor,
                    reactive_power_quantity=detail.reactive_power_quantity,
                    reactive_power_rate=detail.reactive_power_rate,
                    reactive_power_charge=detail.reactive_power_charge,
                    service_price=detail.service_price,
                    operating_price=detail.operating_price,
                    renewable_energy_fee=detail.renewable_energy_fee,
                    amount_ex_vat=detail.amount_ex_vat,
                    amount_inc_vat=detail.amount_inc_vat,
                )
                session.add(db_detail)

            for detail in invoice_data.gas_mo_details:
                db_detail = GasMODetail(
                    invoice_id=invoice.id,
                    period_from=detail.period.period_from if detail.period else None,
                    period_to=detail.period.period_to if detail.period else None,
                    consumption_m3=detail.consumption_m3,
                    consumption_mwh=detail.consumption_mwh,
                    conversion_factor=detail.conversion_factor,
                    combustion_heat=detail.combustion_heat,
                    meter_reading_start=detail.meter_reading_start,
                    meter_reading_end=detail.meter_reading_end,
                    commodity_charge=detail.commodity_charge,
                    distribution_charge=detail.distribution_charge,
                    fixed_monthly_fee=detail.fixed_monthly_fee,
                    period_months=detail.period_months,
                    commodity_unit_price=detail.commodity_unit_price,
                    commodity_total_price=detail.commodity_total_price,
                    fixed_monthly_fee_unit_price=detail.fixed_monthly_fee_unit_price,
                    distribution_unit_price=detail.distribution_unit_price,
                    distribution_fixed_price=detail.distribution_fixed_price,
                    reserved_capacity_unit_price=detail.reserved_capacity_unit_price,
                    reserved_capacity_price=detail.reserved_capacity_price,
                    market_operator_price=detail.market_operator_price,
                    natural_gas_tax_total=detail.natural_gas_tax_total,
                    amount_ex_vat=detail.amount_ex_vat,
                    amount_inc_vat=detail.amount_inc_vat,
                )
                session.add(db_detail)

            for detail in invoice_data.gas_vo_details:
                db_detail = GasVODetail(
                    invoice_id=invoice.id,
                    period_from=detail.period.period_from if detail.period else None,
                    period_to=detail.period.period_to if detail.period else None,
                    consumption_m3=detail.consumption_m3,
                    consumption_mwh=detail.consumption_mwh,
                    conversion_factor=detail.conversion_factor,
                    combustion_heat=detail.combustion_heat,
                    daily_reserved_capacity=detail.daily_reserved_capacity,
                    other_supply_services_price=detail.other_supply_services_price,
                    trade_reserved_capacity_unit_price=detail.trade_reserved_capacity_unit_price,
                    trade_reserved_capacity_price=detail.trade_reserved_capacity_price,
                    distribution_service_price=detail.distribution_service_price,
                    distribution_system_unit_price=detail.distribution_system_unit_price,
                    distribution_reserved_capacity_unit_price=detail.distribution_reserved_capacity_unit_price,
                    distribution_reserved_capacity_price=detail.distribution_reserved_capacity_price,
                    market_operator_price=detail.market_operator_price,
                    natural_gas_tax_total=detail.natural_gas_tax_total,
                    amount_ex_vat=detail.amount_ex_vat,
                    amount_inc_vat=detail.amount_inc_vat,
                )
                session.add(db_detail)

            for detail in invoice_data.water_details:
                db_detail = WaterDetail(
                    invoice_id=invoice.id,
                    period_from=detail.period.period_from if detail.period else None,
                    period_to=detail.period.period_to if detail.period else None,
                    consumption_m3=detail.consumption_m3,
                    meter_reading_start=detail.meter_reading_start,
                    meter_reading_end=detail.meter_reading_end,
                    water_rate=detail.water_rate,
                    sewage_rate=detail.sewage_rate,
                    precipitation_water=detail.precipitation_water,
                    wastewater_charge=detail.wastewater_charge,
                    amount_ex_vat=detail.amount_ex_vat,
                    amount_inc_vat=detail.amount_inc_vat,
                )
                session.add(db_detail)

            for detail in invoice_data.heat_details:
                db_detail = HeatDetail(
                    invoice_id=invoice.id,
                    period_from=detail.period.period_from if detail.period else None,
                    period_to=detail.period.period_to if detail.period else None,
                    consumption_gj=detail.consumption_gj,
                    heat_consumption=detail.heat_consumption,
                    hot_water_heating=detail.hot_water_heating,
                    cold_water=detail.cold_water,
                    total_heat_consumption=detail.total_heat_consumption,
                    reserved_capacity=detail.reserved_capacity,
                    supplementary_water=detail.supplementary_water,
                    heated_area=detail.heated_area,
                    fixed_monthly_fee=detail.fixed_monthly_fee,
                    variable_charge=detail.variable_charge,
                    amount_ex_vat=detail.amount_ex_vat,
                    amount_inc_vat=detail.amount_inc_vat,
                )
                session.add(db_detail)

            return invoice.id

        async with self._get_session() as session:
            return await save_invoice(session)

    async def process_upload(
        self,
        content: BinaryIO,
        filename: str,
        content_type: str | None = None,
    ) -> ProcessingResult:
        """Process a file uploaded via FastAPI.

        Args:
            content: File-like object with read() method.
            filename: Original filename from upload.
            content_type: MIME type from upload.

        Returns:
            ProcessingResult with status and metadata.
        """
        file_bytes = content.read() if hasattr(content, "read") else content
        if isinstance(file_bytes, str):
            file_bytes = file_bytes.encode()

        return await self.process_file(
            source=file_bytes,
            original_filename=filename,
        )

    async def process_directory(
        self,
        directory: str | Path,
        recursive: bool = False,
        max_concurrency: int = 4,
    ) -> list[ProcessingResult]:
        """Process all supported files in a directory (concurrently).

        Args:
            directory: Path to directory containing invoice files.
            recursive: If True, also process subdirectories.
            max_concurrency: Maximum number of concurrent file processing tasks.

        Returns:
            List of ProcessingResult for each processed file.
        """
        # First ingest all files
        transactions = await self._ingestion_service.ingest_directory(
            directory=directory,
            recursive=recursive,
        )

        if not transactions:
            return []

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _process_one(tx: Transaction) -> ProcessingResult:
            async with semaphore:
                return await self._process_transaction(tx)

        results = await asyncio.gather(
            *[_process_one(tx) for tx in transactions],
            return_exceptions=False,
        )
        return list(results)

    async def _process_transaction(self, tx: Transaction) -> ProcessingResult:
        """Process a single pre-ingested transaction through the pipeline.

        Handles classify → OCR → extract → analyse → save stages.
        Returns ProcessingResult in all cases (never raises).
        """
        try:
            # Classify
            await self._update_status(tx.id, ProcessingStatus.CLASSIFYING)
            doc_kind = await self._classifier.classify_transaction(tx)

            # OCR
            await self._update_status(tx.id, ProcessingStatus.OCR_PROCESSING)
            ocr_result = await self._run_ocr_stage(tx, doc_kind)

            if not ocr_result.is_successful:
                await self._update_status(
                    tx.id,
                    ProcessingStatus.FAILED,
                    error_message=ocr_result.error_message,
                )
                return ProcessingResult(
                    transaction_id=tx.id,
                    status=ProcessingStatus.FAILED,
                    document_kind=doc_kind,
                    error_message=ocr_result.error_message,
                )

            ocr_db_id = await self._save_ocr_result(tx.id, ocr_result)

            # Extract (text LLM, with vision fallback if confidence < threshold)
            await self._update_status(tx.id, ProcessingStatus.EXTRACTING)
            ext_start = time.time()
            _tx_file_bytes: bytes | None = None
            if tx.file_path:
                with suppress(Exception):
                    _tx_file_bytes = Path(tx.file_path).read_bytes()
            try:
                extraction_result = await asyncio.wait_for(
                    self._run_iterative_extraction(
                        ocr_result.full_text,
                        tx.filename,
                        file_bytes=_tx_file_bytes,
                    ),
                    timeout=180.0,
                )
            except TimeoutError:
                extraction_result = ExtractionResult(
                    source_file=tx.filename,
                    strategy_name=self._extraction_strategy.name,
                    errors=["Extraction timed out after 180s"],
                )
            ext_duration = (time.time() - ext_start) * 1000
            await self._save_extraction_result(
                tx.id, extraction_result, ocr_result_id=ocr_db_id, duration_ms=ext_duration,
            )

            if extraction_result.errors and not extraction_result.invoice_data:
                await self._update_status(
                    tx.id,
                    ProcessingStatus.FAILED,
                    error_message="; ".join(extraction_result.errors),
                )
                return ProcessingResult(
                    transaction_id=tx.id,
                    status=ProcessingStatus.FAILED,
                    document_kind=doc_kind,
                    extraction_result=extraction_result,
                    error_message="; ".join(extraction_result.errors),
                )

            # Save invoice data
            invoice_id = None
            if extraction_result.invoice_data:
                # Update commodity on transaction as early as possible
                async with self._get_session() as session:
                    stmt = (
                        update(Transaction)
                        .where(Transaction.id == tx.id)
                        .values(commodity=extraction_result.invoice_data.commodity.value)
                    )
                    await session.execute(stmt)

                # Run analysis before saving
                await self._run_analysis_stage(extraction_result.invoice_data)
                invoice_id = await self._save_invoice_data(
                    tx.id,
                    extraction_result.invoice_data,
                    extraction_confidence=extraction_result.confidence,
                )

            await self._update_status(tx.id, ProcessingStatus.COMPLETED)

            return ProcessingResult(
                transaction_id=tx.id,
                status=ProcessingStatus.COMPLETED,
                document_kind=doc_kind,
                extraction_result=extraction_result,
                invoice_id=invoice_id,
            )

        except Exception as e:
            error_msg = str(e)
            # Save a failed extraction result so the error is recorded in extraction_results
            try:
                failed_result = ExtractionResult(
                    source_file=tx.filename,
                    strategy_name=self._extraction_strategy.name,
                    errors=[error_msg],
                )
                await self._save_extraction_result(tx.id, failed_result)
            except Exception:
                logger.exception("Failed to save error extraction result for %s", tx.id)

            await self._update_status(
                tx.id,
                ProcessingStatus.ERROR,
                error_message=error_msg,
            )
            return ProcessingResult(
                transaction_id=tx.id,
                status=ProcessingStatus.ERROR,
                error_message=error_msg,
            )

    async def process_with_benchmark(
        self,
        source: str | bytes | Path,
        original_filename: str | None = None,
        strategies: list[BaseExtractionStrategy] | None = None,
    ) -> tuple[ProcessingResult, BenchmarkComparison]:
        """Process a file with multi-strategy benchmark comparison.

        Runs full pipeline and additionally benchmarks multiple extraction
        strategies to compare their performance.

        Args:
            source: File path or raw bytes.
            original_filename: Original filename (required for bytes input).
            strategies: List of strategies to benchmark.
                        Defaults to Regex + LangChain.

        Returns:
            Tuple of (ProcessingResult, BenchmarkComparison).
        """
        transaction: Transaction | None = None
        benchmark_orchestrator = BenchmarkOrchestrator(strategies=strategies)

        try:
            # Stage 1: Ingest
            transaction = await self._ingestion_service.ingest_and_save(
                source=source,
                original_filename=original_filename,
            )

            # Stage 2: Classify
            await self._update_status(transaction.id, ProcessingStatus.CLASSIFYING)
            doc_kind = await self._classifier.classify_transaction(transaction)

            # Stage 3: OCR
            await self._update_status(transaction.id, ProcessingStatus.OCR_PROCESSING)
            ocr_result = await self._run_ocr_stage(transaction, doc_kind)

            if not ocr_result.is_successful:
                await self._update_status(
                    transaction.id,
                    ProcessingStatus.FAILED,
                    error_message=ocr_result.error_message,
                )
                return (
                    ProcessingResult(
                        transaction_id=transaction.id,
                        status=ProcessingStatus.FAILED,
                        document_kind=doc_kind,
                        ocr_result=ocr_result,
                        error_message=ocr_result.error_message,
                    ),
                    BenchmarkComparison(source_file=original_filename or "unknown"),
                )

            ocr_db_id = await self._save_ocr_result(transaction.id, ocr_result)

            # Stage 4: Benchmark Extraction (parallel)
            await self._update_status(transaction.id, ProcessingStatus.EXTRACTING)
            benchmark = await benchmark_orchestrator.benchmark_text(
                text=ocr_result.full_text,
                source_filename=transaction.filename,
                parallel=True,
            )

            # Use the best result for saving
            best_result: ExtractionResult | None = None
            if benchmark.best_strategy:
                best_bm = next(
                    (bm for bm in benchmark.benchmarks if bm.strategy_name == benchmark.best_strategy),
                    None,
                )
                if best_bm:
                    best_result = best_bm.extraction_result

            # Save all extraction results for comparison
            for bm in benchmark.benchmarks:
                if bm.extraction_result:
                    await self._save_extraction_result(
                        transaction.id, bm.extraction_result,
                        ocr_result_id=ocr_db_id,
                        duration_ms=bm.latency_ms,
                    )

            # Check for extraction errors
            if not best_result or (best_result.errors and not best_result.invoice_data):
                error_msg = "All extraction strategies failed"
                if best_result and best_result.errors:
                    error_msg = "; ".join(best_result.errors)

                await self._update_status(
                    transaction.id,
                    ProcessingStatus.FAILED,
                    error_message=error_msg,
                )
                return (
                    ProcessingResult(
                        transaction_id=transaction.id,
                        status=ProcessingStatus.FAILED,
                        document_kind=doc_kind,
                        ocr_result=ocr_result,
                        extraction_result=best_result,
                        error_message=error_msg,
                    ),
                    benchmark,
                )

            # Stage 5: Save to commodity tables
            invoice_id = None
            if best_result.invoice_data:
                # Run analysis before saving
                await self._run_analysis_stage(best_result.invoice_data)
                invoice_id = await self._save_invoice_data(
                    transaction.id,
                    best_result.invoice_data,
                    extraction_confidence=best_result.confidence,
                )

            # Stage 6: Mark as completed
            await self._update_status(transaction.id, ProcessingStatus.COMPLETED)

            return (
                ProcessingResult(
                    transaction_id=transaction.id,
                    status=ProcessingStatus.COMPLETED,
                    document_kind=doc_kind,
                    ocr_result=ocr_result,
                    extraction_result=best_result,
                    invoice_id=invoice_id,
                ),
                benchmark,
            )

        except Exception as e:
            error_msg = str(e)
            benchmark = BenchmarkComparison(source_file=original_filename or "unknown")

            if transaction:
                await self._update_status(
                    transaction.id,
                    ProcessingStatus.ERROR,
                    error_message=error_msg,
                )
                return (
                    ProcessingResult(
                        transaction_id=transaction.id,
                        status=ProcessingStatus.ERROR,
                        error_message=error_msg,
                    ),
                    benchmark,
                )
            raise

    async def get_transaction_status(
        self,
        transaction_id: uuid.UUID,
    ) -> ProcessingStatus | None:
        """Get the current processing status of a transaction.

        Args:
            transaction_id: UUID of the transaction.

        Returns:
            Current ProcessingStatus or None if not found.
        """
        tx = await self._ingestion_service.get_transaction(transaction_id)
        if tx:
            try:
                return ProcessingStatus(tx.status)
            except ValueError:
                return None
        return None

    async def _update_status(
        self,
        transaction_id: uuid.UUID,
        status: ProcessingStatus,
        error_message: str | None = None,
    ) -> None:
        """Update the processing status of a transaction.

        Args:
            transaction_id: UUID of the transaction.
            status: New processing status.
            error_message: Optional error message if status is ERROR/FAILED.
        """
        values: dict = {"status": status.value}

        if error_message:
            values["error_message"] = error_message

        if status == ProcessingStatus.COMPLETED:
            values["completed_at"] = datetime.now(UTC)

        stmt = (
            update(Transaction)
            .where(Transaction.id == transaction_id)
            .values(**values)
        )

        async with self._get_session() as session:
            await session.execute(stmt)


def create_orchestrator(
    session: AsyncSession | None = None,
    extraction_strategy: BaseExtractionStrategy | None = None,
    vision_strategy: BaseExtractionStrategy | None = None,
) -> ProcessingOrchestrator:
    """Factory function to create a ProcessingOrchestrator instance.

    Args:
        session: Optional async session.
        extraction_strategy: Primary text-based extraction strategy.
        vision_strategy: Optional vision LLM fallback for low-confidence results.
    """
    return ProcessingOrchestrator(
        session=session,
        extraction_strategy=extraction_strategy,
        vision_strategy=vision_strategy,
    )


def create_benchmark_orchestrator(
    strategies: list[BaseExtractionStrategy] | None = None,
    run_analysis: bool = True,
) -> BenchmarkOrchestrator:
    """Factory function to create a BenchmarkOrchestrator instance.

    Args:
        strategies: List of strategies to benchmark.
        run_analysis: If True, run transitional/correction analysis.

    Returns:
        Configured BenchmarkOrchestrator.
    """
    return BenchmarkOrchestrator(strategies=strategies, run_analysis=run_analysis)
