"""
Unit tests for the Document Classification Agent.

All tests mock the Gemini SDK at the ``genai.Client`` level so no real
network calls are made.  The test module validates:

* The empty-document early-exit path (no Gemini call, warning event only).
* The happy path with a clean, fully-classified response.
* The unknown-classification path (Gemini returns UNKNOWN with low confidence).
* The confidence normalisation / clamping behaviour (>1.0 and <0.0 inputs).
* The error-handling path (Gemini raises an exception).
* Missing GEMINI_API_KEY environment variable triggers EnvironmentError that
  is caught and returned as a ProcessingError.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.agents.document_classifier import (
    BulkClassificationOutput,
    SingleClassificationResult,
    classify_documents_agent,
)
from app.graph.state import OverallState
from app.models.claim import ClaimStatus, ClaimSubmission, ProcessingError
from app.models.document import DocumentType, UploadedDocument
from app.models.trace import TraceSeverity, TraceStatus


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_document(
    document_id: str = "F001",
    filename: str = "prescription.jpg",
    mime_type: str = "image/jpeg",
) -> UploadedDocument:
    """Create a minimal ``UploadedDocument`` for testing."""
    return UploadedDocument(
        document_id=document_id,
        filename=filename,
        file_path=f"/uploads/{document_id}.jpg",
        mime_type=mime_type,
        uploaded_at=datetime.now(timezone.utc),
    )


def _make_submission(documents: list[UploadedDocument]) -> ClaimSubmission:
    """Create a minimal ``ClaimSubmission`` wrapping the given documents."""
    return ClaimSubmission(
        member_id="EMP001",
        policy_id="PLUM_GHI_2024",
        claim_category="CONSULTATION",
        claimed_amount=Decimal("1500"),
        treatment_date=date(2024, 11, 1),
        documents=documents if documents else [_make_document()],
        submitted_at=datetime.now(timezone.utc),
    )


def _make_state(documents: list[UploadedDocument]) -> OverallState:
    """Build a minimal ``OverallState`` containing the given documents."""
    # ClaimSubmission requires at least one document; when testing the
    # empty-documents path we build the submission normally then swap the
    # list in the state by passing a separate submission.
    submission = _make_submission(documents if documents else [_make_document()])
    state: OverallState = {
        "claim": submission,
        "status": ClaimStatus.IN_PROGRESS,
        "document_results": [],
        "extractions": [],
        "validations": [],
        "fraud_result": None,
        "decision": None,
        "trace_events": [],
        "confidence_score": 1.0,
        "errors": [],
        "warnings": [],
    }
    return state


def _gemini_response(bulk_output: BulkClassificationOutput) -> MagicMock:
    """
    Create a mock Gemini response whose ``.text`` attribute contains the
    JSON-serialised ``BulkClassificationOutput``.
    """
    mock_response = MagicMock()
    mock_response.text = bulk_output.model_dump_json()
    return mock_response


# ---------------------------------------------------------------------------
# TC-DC-001: empty document list
# ---------------------------------------------------------------------------


def test_empty_document_list_returns_warning_without_calling_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When the claim carries no documents the agent must:

    - Not call Gemini.
    - Return a single WARNING-severity trace event.
    - Return no errors and no document_results.
    """
    # Build a state where the submission has one placeholder doc but we
    # test the early-exit guard by monkeypatching the documents attribute.
    doc = _make_document()
    submission = _make_submission([doc])
    state = _make_state([doc])

    # Replace documents with an empty list via a modified submission mock
    mock_submission = MagicMock()
    mock_submission.claim_id = "CLM_TEST"
    mock_submission.documents = []
    state["claim"] = mock_submission  # type: ignore[assignment]

    with patch("app.agents.document_classifier.genai.Client") as mock_client_cls:
        result = classify_documents_agent(state)

    mock_client_cls.assert_not_called()

    trace_events = result.get("trace_events", [])
    assert len(trace_events) == 1, "Expected exactly one trace event for empty documents"

    warning_event = trace_events[0]
    assert warning_event.severity == TraceSeverity.WARNING
    assert warning_event.status == TraceStatus.PARTIAL

    assert "document_results" not in result or result.get("document_results") is None
    assert "errors" not in result or result.get("errors") == []


# ---------------------------------------------------------------------------
# TC-DC-002: successful classification
# ---------------------------------------------------------------------------


def test_successful_classification_populates_document_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When Gemini returns a valid BulkClassificationOutput the agent must:

    - Return ``document_results`` with one entry per document.
    - Return two trace events: STARTED + PASSED.
    - Each result must carry the original document_id from the submission.
    - No errors must be present.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    doc1 = _make_document("F001", "prescription.jpg", "image/jpeg")
    doc2 = _make_document("F002", "hospital_bill.pdf", "application/pdf")
    state = _make_state([doc1, doc2])

    bulk_output = BulkClassificationOutput(
        classifications=[
            SingleClassificationResult(
                document_id="F001",
                document_type=DocumentType.PRESCRIPTION,
                confidence=0.97,
            ),
            SingleClassificationResult(
                document_id="F002",
                document_type=DocumentType.HOSPITAL_BILL,
                confidence=0.91,
            ),
        ]
    )
    mock_response = _gemini_response(bulk_output)

    with patch("app.agents.document_classifier.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = classify_documents_agent(state)

    document_results: list[dict[str, Any]] = result.get("document_results", [])  # type: ignore[assignment]
    assert len(document_results) == 2

    assert document_results[0]["document_id"] == "F001"
    assert document_results[0]["document_type"] == DocumentType.PRESCRIPTION
    assert document_results[0]["confidence"] == pytest.approx(0.97)

    assert document_results[1]["document_id"] == "F002"
    assert document_results[1]["document_type"] == DocumentType.HOSPITAL_BILL
    assert document_results[1]["confidence"] == pytest.approx(0.91)

    trace_events = result.get("trace_events", [])
    assert len(trace_events) == 2
    assert trace_events[0].status == TraceStatus.STARTED
    assert trace_events[1].status == TraceStatus.PASSED

    assert "errors" not in result or result.get("errors") == []


# ---------------------------------------------------------------------------
# TC-DC-003: unknown classification
# ---------------------------------------------------------------------------


def test_unknown_classification_is_preserved_faithfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When Gemini cannot classify a document it returns UNKNOWN with
    confidence <= 0.50.  The agent must:

    - Preserve ``DocumentType.UNKNOWN`` without overriding it.
    - Store the low confidence value as-is.
    - Still return a success trace path (the pipeline continues).
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    doc = _make_document("F003", "unidentified_scan.jpg", "image/jpeg")
    state = _make_state([doc])

    bulk_output = BulkClassificationOutput(
        classifications=[
            SingleClassificationResult(
                document_id="F003",
                document_type=DocumentType.UNKNOWN,
                confidence=0.30,
            )
        ]
    )
    mock_response = _gemini_response(bulk_output)

    with patch("app.agents.document_classifier.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = classify_documents_agent(state)

    document_results: list[dict[str, Any]] = result.get("document_results", [])  # type: ignore[assignment]
    assert len(document_results) == 1
    assert document_results[0]["document_type"] == DocumentType.UNKNOWN
    assert document_results[0]["confidence"] == pytest.approx(0.30)

    trace_events = result.get("trace_events", [])
    assert len(trace_events) == 2
    assert trace_events[1].status == TraceStatus.PASSED


# ---------------------------------------------------------------------------
# TC-DC-004: confidence normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_confidence", "expected"),
    [
        (1.5, 1.0),    # above 1.0 → clamped to 1.0
        (-0.2, 0.0),   # below 0.0 → clamped to 0.0
        (0.0, 0.0),    # exact lower bound — valid
        (1.0, 1.0),    # exact upper bound — valid
        (0.75, 0.75),  # normal value — unchanged
    ],
)
def test_confidence_is_clamped_to_unit_interval(
    raw_confidence: float,
    expected: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Confidence values returned by Gemini that fall outside [0.0, 1.0] must
    be silently clamped to the valid range before being stored in
    ``document_results``.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    doc = _make_document("F004")
    state = _make_state([doc])

    # Inject a raw confidence that may be out-of-range directly into the
    # model via Pydantic — the validator must clamp it.
    result = SingleClassificationResult(
        document_id="F004",
        document_type=DocumentType.PRESCRIPTION,
        confidence=raw_confidence,  # type: ignore[arg-type]
    )
    assert result.confidence == pytest.approx(expected), (
        f"Expected confidence {expected} for raw input {raw_confidence}, "
        f"got {result.confidence}"
    )


def test_confidence_clamping_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verify that out-of-range confidence values are clamped in the full
    classify_documents_agent pipeline, not just at model construction.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    doc = _make_document("F005")
    state = _make_state([doc])

    # Construct output with a confidence that exceeds 1.0 to confirm
    # end-to-end clamping.
    raw_bulk = BulkClassificationOutput(
        classifications=[
            SingleClassificationResult(
                document_id="F005",
                document_type=DocumentType.LAB_REPORT,
                confidence=1.8,  # type: ignore[arg-type]  # will be clamped
            )
        ]
    )
    mock_response = _gemini_response(raw_bulk)

    with patch("app.agents.document_classifier.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = classify_documents_agent(state)

    document_results: list[dict[str, Any]] = result.get("document_results", [])  # type: ignore[assignment]
    assert document_results[0]["confidence"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TC-DC-005: error handling path
# ---------------------------------------------------------------------------


def test_gemini_exception_returns_processing_error_and_critical_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When Gemini raises any exception the agent must:

    - Not re-raise the exception.
    - Return a ``ProcessingError`` in ``errors``.
    - Return a CRITICAL-severity trace event.
    - Return a STARTED trace event (emitted before the exception).
    - Return no ``document_results``.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    doc = _make_document("F006")
    state = _make_state([doc])

    with patch("app.agents.document_classifier.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = TimeoutError(
            "Gemini API timed out after 30s"
        )
        mock_client_cls.return_value = mock_client

        result = classify_documents_agent(state)

    assert "document_results" not in result or not result.get("document_results")

    errors: list[ProcessingError] = result.get("errors", [])  # type: ignore[assignment]
    assert len(errors) == 1
    assert isinstance(errors[0], ProcessingError)
    assert errors[0].recoverable is False
    assert "classification failed" in errors[0].message.lower()

    trace_events = result.get("trace_events", [])
    assert len(trace_events) == 2
    # First event is the STARTED event emitted before the call
    assert trace_events[0].status == TraceStatus.STARTED
    # Second event is the CRITICAL failure event
    assert trace_events[1].severity == TraceSeverity.CRITICAL
    assert trace_events[1].status == TraceStatus.FAILED


def test_missing_api_key_returns_processing_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When ``GEMINI_API_KEY`` is absent or empty the agent must catch the
    ``EnvironmentError`` and return it as a ``ProcessingError`` — never
    letting the exception propagate to LangGraph.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    doc = _make_document("F007")
    state = _make_state([doc])

    result = classify_documents_agent(state)

    errors: list[ProcessingError] = result.get("errors", [])  # type: ignore[assignment]
    assert len(errors) == 1
    assert isinstance(errors[0], ProcessingError)

    trace_events = result.get("trace_events", [])
    critical_events = [
        e for e in trace_events if e.severity == TraceSeverity.CRITICAL
    ]
    assert len(critical_events) >= 1


# ---------------------------------------------------------------------------
# TC-DC-006: short response padding (Gemini returns fewer results than docs)
# ---------------------------------------------------------------------------


def test_short_gemini_response_is_padded_with_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    When Gemini returns fewer classification results than documents the agent
    must pad the missing entries with ``DocumentType.UNKNOWN / 0.0`` so the
    1:1 document-to-result contract is preserved.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    doc1 = _make_document("F008", "rx.jpg")
    doc2 = _make_document("F009", "bill.jpg")
    state = _make_state([doc1, doc2])

    # Gemini returns only one result for two documents
    bulk_output = BulkClassificationOutput(
        classifications=[
            SingleClassificationResult(
                document_id="F008",
                document_type=DocumentType.PRESCRIPTION,
                confidence=0.88,
            )
        ]
    )
    mock_response = _gemini_response(bulk_output)

    with patch("app.agents.document_classifier.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = classify_documents_agent(state)

    document_results: list[dict[str, Any]] = result.get("document_results", [])  # type: ignore[assignment]
    assert len(document_results) == 2

    # First result from Gemini
    assert document_results[0]["document_id"] == "F008"
    assert document_results[0]["document_type"] == DocumentType.PRESCRIPTION

    # Padded result for the missing document
    assert document_results[1]["document_id"] == "F009"
    assert document_results[1]["document_type"] == DocumentType.UNKNOWN
    assert document_results[1]["confidence"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TC-DC-007: document_id originates from submission, not Gemini response
# ---------------------------------------------------------------------------


def test_document_id_in_results_comes_from_submission_not_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    The ``document_id`` stored in ``document_results`` must always originate
    from the original ``UploadedDocument`` (the submission), not from the
    ``document_id`` field echoed back by Gemini.  This ensures that a
    hallucinated or mismatched Gemini ID cannot corrupt the pipeline state.
    """
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    doc = _make_document("REAL_ID_001", "rx.jpg")
    state = _make_state([doc])

    # Gemini echoes back a different (wrong) document_id
    bulk_output = BulkClassificationOutput(
        classifications=[
            SingleClassificationResult(
                document_id="HALLUCINATED_ID",  # <-- wrong
                document_type=DocumentType.PRESCRIPTION,
                confidence=0.93,
            )
        ]
    )
    mock_response = _gemini_response(bulk_output)

    with patch("app.agents.document_classifier.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = classify_documents_agent(state)

    document_results: list[dict[str, Any]] = result.get("document_results", [])  # type: ignore[assignment]
    # document_id must come from the original UploadedDocument
    assert document_results[0]["document_id"] == "REAL_ID_001"