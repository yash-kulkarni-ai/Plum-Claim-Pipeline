"""
LangGraph shared state definition for the Plum Health Insurance Claims
Processing System.

This module defines ``OverallState``, the ``TypedDict`` that is passed as
the mutable memory object between every node in the claim-processing graph.
It mirrors ``ClaimState`` (``app.models.claim``) field-for-field so that the
Pydantic business model and the LangGraph runtime state remain structurally
identical, simplifying serialisation at graph boundaries.

Reducer contract
----------------
LangGraph merges node return values into the shared state using per-field
reducer functions.  List-valued fields that accumulate history across nodes
(extractions, validations, trace events, errors, warnings) are annotated with
explicit ``append_*`` reducers.  These reducers:

- Never overwrite existing history — they always produce ``existing + incoming``.
- Safely handle ``None`` on both sides (initial state before any node runs,
  or a node that returns ``{}`` for a list field).
- Return a new list object on every call so that no two state snapshots share
  a reference.

Scalar fields (``claim``, ``status``, ``fraud_result``, ``decision``,
``confidence_score``, ``document_results``) use LangGraph's default
last-write-wins behaviour, which is the correct semantic for those fields.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from app.models.claim import (
    ClaimStatus,
    ClaimSubmission,
    ProcessingError,
    ProcessingWarning,
)
from app.models.decision import (
    ClaimDecision,
    FraudResult,
    ValidationResult,
)
from app.models.extraction import ExtractionResult
from app.models.trace import TraceEvent


# ---------------------------------------------------------------------------
# Reducer functions
# ---------------------------------------------------------------------------


def append_extractions(
    existing: list[ExtractionResult] | None,
    incoming: list[ExtractionResult] | None,
) -> list[ExtractionResult]:
    """
    Merge two ``ExtractionResult`` lists without losing history.

    LangGraph calls this reducer each time a node returns a value for the
    ``extractions`` field.  Both arguments may be ``None`` (LangGraph passes
    ``None`` for the existing value before the first write, and a node may
    return ``None`` or omit the field entirely when it produces no new
    extractions).

    Parameters
    ----------
    existing:
        The current accumulated list in the graph state.  ``None`` before
        any node has written to this field.
    incoming:
        The list returned by the most recent node.  ``None`` when the node
        did not update this field.

    Returns
    -------
    list[ExtractionResult]
        A new list containing all items from *existing* followed by all
        items from *incoming*, preserving insertion order.
    """
    return (existing or []) + (incoming or [])


def append_validations(
    existing: list[ValidationResult] | None,
    incoming: list[ValidationResult] | None,
) -> list[ValidationResult]:
    """
    Merge two ``ValidationResult`` lists without losing history.

    Each call to the Policy Validation Agent may produce one or more
    ``ValidationResult`` objects.  This reducer ensures that results from
    successive validation passes accumulate rather than overwrite each other,
    preserving the complete policy-check trace for explainability.

    Parameters
    ----------
    existing:
        Accumulated validation results so far.  ``None`` before the first
        policy validation node run.
    incoming:
        Validation results from the most recent node.  ``None`` when the
        node did not update this field.

    Returns
    -------
    list[ValidationResult]
        Merged list in chronological append order.
    """
    return (existing or []) + (incoming or [])


def append_trace_events(
    existing: list[TraceEvent] | None,
    incoming: list[TraceEvent] | None,
) -> list[TraceEvent]:
    """
    Merge two ``TraceEvent`` lists without losing history.

    The trace event list is the append-only audit trail for the entire claim
    pipeline.  Every agent appends its events; no agent may truncate or
    replace the list.  This reducer enforces that invariant at the LangGraph
    state-merge level.

    Parameters
    ----------
    existing:
        All trace events accumulated so far.  ``None`` at graph start.
    incoming:
        Events emitted by the most recent node.  ``None`` when the node
        emitted no events (e.g. it returned ``{}``).

    Returns
    -------
    list[TraceEvent]
        Merged list in chronological append order.
    """
    return (existing or []) + (incoming or [])


def append_errors(
    existing: list[ProcessingError] | None,
    incoming: list[ProcessingError] | None,
) -> list[ProcessingError]:
    """
    Merge two ``ProcessingError`` lists without losing history.

    Recoverable failures recorded by any agent must be preserved so that the
    Decision Agent and Explainability Agent can factor them into the final
    decision and the audit report.  This reducer guarantees that no error
    record is silently dropped by a later node's state update.

    Parameters
    ----------
    existing:
        Processing errors accumulated so far.  ``None`` at graph start.
    incoming:
        Errors recorded by the most recent node.  ``None`` when the node
        encountered no errors.

    Returns
    -------
    list[ProcessingError]
        Merged list in chronological append order.
    """
    return (existing or []) + (incoming or [])


def append_warnings(
    existing: list[ProcessingWarning] | None,
    incoming: list[ProcessingWarning] | None,
) -> list[ProcessingWarning]:
    """
    Merge two ``ProcessingWarning`` lists without losing history.

    Non-fatal warnings (low OCR confidence, missing optional fields, document
    anomalies) are accumulated across all nodes so that the complete
    degradation history is available when the Explainability Agent builds its
    summary.

    Parameters
    ----------
    existing:
        Processing warnings accumulated so far.  ``None`` at graph start.
    incoming:
        Warnings recorded by the most recent node.  ``None`` when the node
        recorded no warnings.

    Returns
    -------
    list[ProcessingWarning]
        Merged list in chronological append order.
    """
    return (existing or []) + (incoming or [])


# ---------------------------------------------------------------------------
# OverallState
# ---------------------------------------------------------------------------


class OverallState(TypedDict):
    """
    Shared memory object passed between every node in the LangGraph pipeline.

    ``OverallState`` is structurally identical to ``ClaimState``
    (``app.models.claim``) and maps one-to-one to its fields.  The separation
    exists because LangGraph requires a ``TypedDict`` for its state container
    while the rest of the system uses Pydantic models for validation and
    serialisation.  The API layer converts between the two representations at
    graph entry and exit points.

    Field semantics
    ---------------
    Fields annotated with an ``append_*`` reducer accumulate history across
    nodes (last-write semantics would silently drop earlier entries).  All
    other fields use LangGraph's default last-write-wins merge, which is
    correct for scalar values and for fields that are set exactly once
    (``claim``, ``decision``) or replaced wholesale (``status``,
    ``fraud_result``, ``confidence_score``).

    Mutation contract
    -----------------
    No node should mutate the state object it receives.  Every node must
    return a ``dict`` containing only the fields it wishes to update; LangGraph
    applies the appropriate reducer or last-write merge for each field.
    Returning ``{}`` is valid and means the node made no state changes.
    """

    # ------------------------------------------------------------------
    # Immutable submission record — set once at graph entry; never updated.
    # ------------------------------------------------------------------

    claim: ClaimSubmission
    """Original claim submission as received from the frontend."""

    # ------------------------------------------------------------------
    # Scalar fields — last-write-wins semantics (LangGraph default).
    # ------------------------------------------------------------------

    status: ClaimStatus
    """
    Current workflow lifecycle status.  Updated by each agent at the
    start and end of its processing step.
    """

    document_results: list[dict]  # type: ignore[type-arg]
    """
    Classification and verification results for each uploaded document.
    Populated by the Document Classification and Document Verification
    agents.  Typed as ``list[dict]`` pending a dedicated schema model.
    """

    fraud_result: FraudResult | None
    """
    Output of the Fraud Detection Agent.  ``None`` until that agent runs.
    """

    decision: ClaimDecision | None
    """
    Final claim decision.  ``None`` until the Decision Agent runs.
    """

    confidence_score: float
    """
    Global processing confidence in [0.0, 1.0].  Starts at 1.0; decreased
    by agents that encounter degraded processing conditions.
    """

    # ------------------------------------------------------------------
    # Append-accumulating fields — explicit reducer functions preserve
    # history across nodes.
    # ------------------------------------------------------------------

    extractions: Annotated[list[ExtractionResult], append_extractions]
    """
    OCR extraction results, one per processed document.  New results are
    appended by the OCR Extraction Agent without overwriting prior entries.
    """

    validations: Annotated[list[ValidationResult], append_validations]
    """
    Policy rule check results.  New results are appended by the Policy
    Validation Agent; the full list forms the explainability trace.
    """

    trace_events: Annotated[list[TraceEvent], append_trace_events]
    """
    Append-only audit trail.  Every agent appends its events here; no
    agent may truncate or replace the list.
    """

    errors: Annotated[list[ProcessingError], append_errors]
    """
    Recoverable pipeline failures.  Accumulated across all agents.
    """

    warnings: Annotated[list[ProcessingWarning], append_warnings]
    """
    Non-fatal degradation notices.  Accumulated across all agents.
    """