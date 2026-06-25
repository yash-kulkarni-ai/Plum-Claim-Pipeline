"""
Document Classification Agent for the Plum Health Insurance Claims Processing System.

This module implements the first active intelligence node in the LangGraph
pipeline.  Its sole responsibility is to classify every uploaded document
into a standardised ``DocumentType`` category before downstream agents
(verification, OCR, policy validation) begin processing.

Architecture position
---------------------
::

    ClaimSubmission (documents attached)
        ↓
    classify_documents_agent      ← this module
        ↓  document_results populated
    DocumentVerificationAgent
        ↓
    OCRExtractionAgent
        ↓  ...

Design principles
-----------------
- The public interface is a single LangGraph node function
  ``classify_documents_agent(state) -> dict``.  It never raises; all
  failures are captured and returned as ``ProcessingError`` objects so
  the graph can continue gracefully.
- Gemini credentials are read from the ``GEMINI_API_KEY`` environment
  variable at call time.  No client singleton is cached at import time,
  making the agent safe for LangGraph retries and parallel execution.
- Structured output is requested via ``response_schema=BulkClassificationOutput``
  inside ``GenerateContentConfig``.  The response is parsed with Pydantic's
  ``model_validate_json`` — no manual JSON wrangling.
- Confidence is clamped to ``[0.0, 1.0]`` defensively, even though the
  prompt instructs Gemini to stay within that range.
- The current implementation sends only document metadata (filename, MIME
  type) to Gemini.  The public interface is designed so that a future
  version can send actual file bytes by updating the prompt builder without
  changing the node signature.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models.claim import ProcessingError
from app.models.document import DocumentType
from app.observability.events import AgentActions, EventMessages
from app.observability.tracer import AgentTracer
from app.graph.state import OverallState

__all__ = ["classify_documents_agent"]

# ---------------------------------------------------------------------------
# Internal schema models
# ---------------------------------------------------------------------------

_AGENT_NAME = "DocumentClassificationAgent"
_GEMINI_MODEL = "gemini-2.5-flash"


class SingleClassificationResult(BaseModel):
    """
    Classification outcome for a single uploaded document.

    Returned by Gemini as one element of ``BulkClassificationOutput.classifications``.
    The ``document_id`` field must echo back the ID supplied in the prompt
    payload so that the agent can map results back to the original documents
    in constant time.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    document_id: str = Field(
        ...,
        description=(
            "Echo of the document_id supplied in the classification request.  "
            "Must match exactly; the agent uses this to correlate the result "
            "with the original UploadedDocument."
        ),
    )

    document_type: DocumentType = Field(
        ...,
        description=(
            "Classified document category.  Must be one of the allowed "
            "DocumentType values.  Use UNKNOWN when the document cannot be "
            "confidently assigned to any category."
        ),
    )

    confidence: float = Field(
        ...,
        description=(
            "Classification confidence in [0.0, 1.0].  Values <= 0.50 "
            "paired with document_type=UNKNOWN indicate that classification "
            "was not possible."
        ),
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, value: object) -> float:
        """Clamp confidence to [0.0, 1.0] regardless of what Gemini returns."""
        try:
            score = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, score))


class BulkClassificationOutput(BaseModel):
    """
    Top-level structured output returned by Gemini for a batch of documents.

    The ``classifications`` list must have exactly one entry per document
    supplied in the prompt, preserving the original ordering so that
    ``zip``-based correlation is deterministic.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    classifications: list[SingleClassificationResult] = Field(
        default_factory=list,
        description=(
            "One classification result per document.  Ordering must match "
            "the ordering of documents in the request payload."
        ),
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_classification_prompt(
    documents: list[dict[str, str]],
) -> str:
    """
    Build the Gemini classification prompt for a batch of documents.

    Parameters
    ----------
    documents:
        List of lightweight document descriptors, each containing
        ``document_id``, ``filename``, ``mime_type``, and ``file_path``.

    Returns
    -------
    str
        A structured prompt string ready to send to Gemini.

    Notes
    -----
    The prompt is designed to be extended in the future: when the agent
    gains access to actual file bytes, the document bytes can be injected
    alongside the metadata descriptors without changing the prompt structure.
    """
    allowed_types = (
        "PRESCRIPTION, HOSPITAL_BILL, PHARMACY_BILL, LAB_REPORT, "
        "DENTAL_REPORT, DIAGNOSTIC_REPORT, DISCHARGE_SUMMARY, UNKNOWN"
    )

    doc_lines = "\n".join(
        f"  - document_id: {d['document_id']!r}, "
        f"filename: {d['filename']!r}, "
        f"mime_type: {d['mime_type']!r}"
        for d in documents
    )

    return f"""You are a medical document classification expert for an Indian health insurance claims system.

Classify each of the following uploaded documents into exactly one category from the allowed list.

ALLOWED CATEGORIES:
{allowed_types}

CLASSIFICATION RULES:
1. PRESCRIPTION       — Doctor's Rx slip or letterhead with diagnosis and medicines.
2. HOSPITAL_BILL      — Itemised invoice from a hospital, clinic, or day-care centre.
3. PHARMACY_BILL      — Receipt from a licensed pharmacy listing dispensed medicines.
4. LAB_REPORT         — Diagnostic laboratory result sheet (blood tests, cultures, etc.).
5. DENTAL_REPORT      — Dental treatment summary or procedure note.
6. DIAGNOSTIC_REPORT  — Imaging or special investigation report (MRI, CT, PET, ultrasound).
7. DISCHARGE_SUMMARY  — Hospital discharge document summarising an inpatient admission.
8. UNKNOWN            — Cannot be confidently assigned to any category above.

CONFIDENCE RULES:
- Return confidence as a float between 0.0 and 1.0.
- If you cannot determine the type, set document_type to UNKNOWN and confidence <= 0.50.
- Do NOT invent or modify document_id values — echo them back exactly as provided.
- Classify using the filename and MIME type as the primary signals.

DOCUMENTS TO CLASSIFY:
{doc_lines}

Return a BulkClassificationOutput with one SingleClassificationResult per document,
preserving the original document ordering."""


# ---------------------------------------------------------------------------
# Gemini client factory
# ---------------------------------------------------------------------------


def _build_client() -> genai.Client:
    """
    Construct a ``genai.Client`` from the ``GEMINI_API_KEY`` environment
    variable.

    Raises
    ------
    EnvironmentError
        When ``GEMINI_API_KEY`` is not set or is empty.

    Notes
    -----
    A new client is constructed on every agent invocation.  This is
    intentional: it avoids stale state between LangGraph retries and
    eliminates hidden shared mutable state at the module level.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set or is empty."
        )
    return genai.Client(api_key=api_key)


# ---------------------------------------------------------------------------
# Core classification logic
# ---------------------------------------------------------------------------


def _invoke_gemini(
    client: genai.Client,
    documents: list[dict[str, str]],
) -> BulkClassificationOutput:
    """
    Call Gemini with a structured output configuration and parse the response.

    Parameters
    ----------
    client:
        An authenticated ``genai.Client`` instance.
    documents:
        Lightweight document descriptors built from the uploaded documents.

    Returns
    -------
    BulkClassificationOutput
        Validated structured output parsed from the Gemini response.

    Raises
    ------
    Exception
        Any exception from the Gemini SDK or Pydantic validation propagates
        to the caller; ``classify_documents_agent`` handles it.
    """
    prompt = _build_classification_prompt(documents)

    response = client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BulkClassificationOutput,
        ),
    )

    raw_text: str = response.text or ""
    return BulkClassificationOutput.model_validate_json(raw_text)


def _build_document_payload(
    documents: list[object],
) -> list[dict[str, str]]:
    """
    Extract the minimal metadata fields needed for classification from a
    list of ``UploadedDocument`` instances.

    Using a plain ``dict`` at the boundary keeps ``_invoke_gemini`` decoupled
    from the Pydantic model and makes future extension straightforward.

    Parameters
    ----------
    documents:
        List of ``UploadedDocument`` instances from the claim submission.

    Returns
    -------
    list[dict[str, str]]
        One dict per document containing ``document_id``, ``filename``,
        ``mime_type``, and ``file_path``.
    """
    payloads: list[dict[str, str]] = []
    for doc in documents:
        payloads.append(
            {
                "document_id": str(getattr(doc, "document_id", "")),
                "filename": str(getattr(doc, "filename", "")),
                "mime_type": str(getattr(doc, "mime_type", "")),
                "file_path": str(getattr(doc, "file_path", "")),
            }
        )
    return payloads


def _build_document_results(
    documents: list[object],
    classifications: list[SingleClassificationResult],
) -> list[dict[str, object]]:
    """
    Merge the original document metadata with classification results into
    the ``document_results`` list that will be stored on ``OverallState``.

    Parameters
    ----------
    documents:
        Original ``UploadedDocument`` instances from the claim.
    classifications:
        Ordered classification results from Gemini, one per document.

    Returns
    -------
    list[dict[str, object]]
        One dict per document containing ``document_id``, ``document_type``,
        ``confidence``, ``original_filename``, and ``mime_type``.

    Notes
    -----
    Results are correlated by index rather than by ``document_id`` as a
    defensive measure: if Gemini echoes back a mismatched ID the index
    correlation ensures we still produce a 1:1 output list.  The
    ``document_id`` in the result dict always comes from the original
    ``UploadedDocument``, never from the Gemini response.
    """
    results: list[dict[str, object]] = []
    for doc, classification in zip(documents, classifications):
        results.append(
            {
                "document_id": str(getattr(doc, "document_id", "")),
                "document_type": classification.document_type,
                "confidence": classification.confidence,
                "original_filename": str(getattr(doc, "filename", "")),
                "mime_type": str(getattr(doc, "mime_type", "")),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Public LangGraph node
# ---------------------------------------------------------------------------


def classify_documents_agent(state: OverallState) -> dict[str, object]:
    """
    LangGraph node: classify every uploaded document in the claim submission.

    Reads ``state["claim"].documents``, sends a batch classification request
    to Gemini 2.5 Flash, and returns the results as ``document_results``
    together with the agent's trace events.

    The node never raises.  All failures are captured as ``ProcessingError``
    and returned in the ``errors`` field so that the LangGraph graph can
    continue to the Decision Agent via the early-termination route.

    Parameters
    ----------
    state:
        The current ``OverallState``.  Only ``state["claim"]`` is read.

    Returns
    -------
    dict[str, object]
        A partial state update dict.  On success::

            {
                "document_results": [...],
                "trace_events": [start_event, success_event],
            }

        On empty document list::

            {
                "trace_events": [warning_event],
            }

        On any exception::

            {
                "errors": [processing_error],
                "trace_events": [start_event, critical_event],
            }
    """
    claim = state["claim"]
    documents = claim.documents

    # ------------------------------------------------------------------
    # Guard: nothing to classify
    # ------------------------------------------------------------------
    if not documents:
        warning_event = AgentTracer.warning_event(
            agent_name=_AGENT_NAME,
            action=AgentActions.DOCUMENT_CLASSIFICATION,
            message=EventMessages.ocr_warning(
                "no documents found in claim submission; classification skipped"
            ),
            metadata={"claim_id": claim.claim_id},
        )
        return {"trace_events": [warning_event]}

    # ------------------------------------------------------------------
    # Emit start event
    # ------------------------------------------------------------------
    start_event = AgentTracer.start_event(
        agent_name=_AGENT_NAME,
        action=AgentActions.DOCUMENT_CLASSIFICATION,
        message=EventMessages.agent_started(_AGENT_NAME),
        metadata={
            "claim_id": claim.claim_id,
            "document_count": len(documents),
        },
    )

    try:
        # ------------------------------------------------------------------
        # Build payload and call Gemini
        # ------------------------------------------------------------------
        doc_payloads = _build_document_payload(documents)
        client = _build_client()
        bulk_output = _invoke_gemini(client, doc_payloads)

        # ------------------------------------------------------------------
        # Merge results: fall back to UNKNOWN for any missing entries
        # ------------------------------------------------------------------
        classifications = bulk_output.classifications

        # If Gemini returned fewer results than documents (defensive), pad
        # the list with UNKNOWN entries so the 1:1 contract holds.
        while len(classifications) < len(documents):
            missing_index = len(classifications)
            missing_id = str(getattr(documents[missing_index], "document_id", "unknown"))
            classifications.append(
                SingleClassificationResult(
                    document_id=missing_id,
                    document_type=DocumentType.UNKNOWN,
                    confidence=0.0,
                )
            )

        document_results = _build_document_results(documents, classifications)

        # ------------------------------------------------------------------
        # Emit success event
        # ------------------------------------------------------------------
        classified_types = [
            str(r["document_type"]) for r in document_results
        ]
        success_event = AgentTracer.success_event(
            agent_name=_AGENT_NAME,
            action=AgentActions.DOCUMENT_CLASSIFICATION,
            message=EventMessages.agent_completed(_AGENT_NAME),
            metadata={
                "claim_id": claim.claim_id,
                "document_count": len(documents),
                "classified_types": classified_types,
            },
        )

        return {
            "document_results": document_results,
            "trace_events": [start_event, success_event],
        }

    except Exception as exc:  # noqa: BLE001
        # ------------------------------------------------------------------
        # Capture failure — never let the graph crash
        # ------------------------------------------------------------------
        processing_error = ProcessingError(
            component=_AGENT_NAME,
            message=EventMessages.processing_error(
                f"document classification failed: {exc}"
            ),
            recoverable=False,
            timestamp=datetime.now(timezone.utc),
        )
        critical_event = AgentTracer.critical_event(
            agent_name=_AGENT_NAME,
            action=AgentActions.DOCUMENT_CLASSIFICATION,
            message=EventMessages.processing_error(
                f"document classification failed: {exc}"
            ),
            metadata={
                "claim_id": claim.claim_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        return {
            "errors": [processing_error],
            "trace_events": [start_event, critical_event],
        }