"""
Observability and audit trail contracts for the Plum Health Insurance Claims
Processing System.

This module defines the schema layer through which every agent in the
multi-agent pipeline emits structured telemetry.  It is the single source of
truth for the event format that flows between agents, the explainability
system, the operations UI timeline, and the evaluation framework.

Architecture position
---------------------
::

    Agent (any)
        ↓  emits TraceEvent
    Claim State (accumulated per claim)
        ↓
    TraceTimeline  ──→  UI Timeline
        ↓
    TraceSummary   ──→  Operations Dashboard
        ↓
    AgentExecutionSummary  ──→  Performance Monitoring

Design principles
-----------------
- ``TraceEvent`` is the atomic unit of observability.  Every significant
  action — started, passed, failed, skipped — produces exactly one event.
  No agent should emit raw dicts.
- ``event_id`` is auto-generated as a UUID4 string when not supplied, so
  callers never need to manage identity themselves.
- All string fields are whitespace-stripped and validated non-empty so that
  the audit trail is always readable without defensive handling downstream.
- ``metadata`` is intentionally untyped (``dict[str, Any]``) to decouple
  this schema layer from the varied payloads each agent needs to attach
  (confidence scores, document IDs, error messages, processing times, etc.).
- Computed properties (``is_failure``, ``is_warning``, ``is_critical``,
  ``total_events``) are pure and side-effect-free so they can be used safely
  in list comprehensions and filter expressions across the codebase.
- No business logic, policy rules, claim decisions, or persistence code lives
  in this module.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------

_SHARED_CONFIG = ConfigDict(
    extra="forbid",
    validate_assignment=True,
    populate_by_name=True,
)


# ---------------------------------------------------------------------------
# Shared validator helpers
# ---------------------------------------------------------------------------


def _require_nonempty_str(field_name: str, value: Any) -> str:
    """
    Strip whitespace from a required string field and reject empty values.

    Raises ``ValueError`` with a message that names the offending field so
    that Pydantic surfaces a useful, actionable error to the caller.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string, got {type(value).__name__!r}."
        )
    stripped = value.strip()
    if not stripped:
        raise ValueError(
            f"{field_name} must not be empty or contain only whitespace."
        )
    return stripped


def _require_nonnegative_int(field_name: str, value: Any) -> int:
    """
    Coerce ``value`` to ``int`` and assert it is >= 0.

    Raises ``ValueError`` with the field name and the offending value so the
    caller can surface a precise error message.
    """
    try:
        int_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a non-negative integer, got {value!r}."
        ) from exc
    if int_value < 0:
        raise ValueError(f"{field_name} must be >= 0, got {int_value}.")
    return int_value


# ---------------------------------------------------------------------------
# TraceStatus
# ---------------------------------------------------------------------------


class TraceStatus(StrEnum):
    """
    Execution outcome of a single traceable operation within an agent.

    Using ``StrEnum`` ensures every value serialises to its plain string form
    in JSON without a custom encoder and that comparisons against raw strings
    work transparently across the codebase.

    Values
    ------
    STARTED
        The operation has begun but has not yet completed.  Events with this
        status are emitted at the entry point of long-running steps so that
        the UI can render a live "in progress" indicator and so that timeouts
        can be detected by comparing the ``STARTED`` timestamp against wall
        clock time.
    PASSED
        The operation completed successfully and all expected outputs were
        produced.  This is the happy-path terminal status.
    FAILED
        The operation encountered an unrecoverable error.  Downstream agents
        that depend on its output must treat that output as unavailable and
        either skip their own work (emitting ``SKIPPED``) or degrade
        gracefully (emitting ``PARTIAL``).
    SKIPPED
        The operation was intentionally bypassed because a prerequisite was
        not met or the operation was not applicable to this claim.  Examples:
        OCR is skipped when document verification determines the file is
        unreadable; the fraud check is skipped when the claimed amount is
        below the low-risk threshold.
    PARTIAL
        The operation completed but with degraded quality or incomplete
        results.  Examples: a bill was extracted but one line item's amount
        was illegible; OCR succeeded but the doctor registration number was
        obscured by a rubber stamp.  Downstream agents should treat outputs
        from partial operations with reduced trust and should lower their
        confidence scores accordingly.
    """

    STARTED = "STARTED"
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    PARTIAL = "PARTIAL"


# ---------------------------------------------------------------------------
# TraceSeverity
# ---------------------------------------------------------------------------


class TraceSeverity(StrEnum):
    """
    Importance level of a single ``TraceEvent``.

    Severity is orthogonal to ``TraceStatus``: a ``PASSED`` event can carry
    ``WARNING`` severity if the operation succeeded but produced signals worth
    surfacing (e.g. low confidence, a detected rubber stamp over a
    registration number).  Conversely, a ``FAILED`` event that is expected
    and handled (e.g. a gracefully degraded component) may carry only
    ``WARNING`` rather than ``ERROR``.

    Values
    ------
    INFO
        Routine operational event.  Indicates normal progress through the
        pipeline.  The majority of events in a clean claim will carry this
        severity.
    WARNING
        Non-fatal anomaly.  The pipeline continues, but the signal is worth
        surfacing to a human reviewer or the operations dashboard.  Examples:
        low OCR confidence, missing optional fields, a suspicious (but
        sub-threshold) fraud signal.
    ERROR
        A component failed in a way that degrades the claim output.  The
        pipeline continues with whatever partial output is available, but the
        confidence score must be reduced and the failure must be visible in
        the audit trail.  Examples: LLM timeout, JSON parsing failure,
        unexpected extraction schema mismatch.
    CRITICAL
        A failure severe enough that the claim cannot be processed reliably
        and must be routed to ``MANUAL_REVIEW`` immediately.  Examples: member
        not found in the policy roster, no readable documents available for a
        required document type, fraud score exceeds the mandatory review
        threshold.
    """

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# TraceEvent
# ---------------------------------------------------------------------------


class TraceEvent(BaseModel):
    """
    A single structured telemetry event emitted by an agent.

    ``TraceEvent`` is the atomic unit of the observability system.  Every
    significant action — the start and end of each agent, each validation
    check, each OCR attempt, each fraud signal evaluation, and the final
    decision generation — produces exactly one ``TraceEvent`` instance.

    An ordered sequence of ``TraceEvent`` objects for a claim forms a complete
    audit trail from which any engineer or operations reviewer can reconstruct
    the full claim journey without inspecting agent internals.

    Naming conventions
    ------------------
    ``agent_name`` should be the class name of the emitting agent, e.g.:

    - ``DocumentClassificationAgent``
    - ``OCRAgent``
    - ``PolicyValidationAgent``
    - ``FraudDetectionAgent``
    - ``DecisionAgent``

    ``action`` should be a snake_case verb phrase describing what the agent
    was doing, e.g.:

    - ``classify_document``
    - ``extract_prescription``
    - ``validate_waiting_period``
    - ``evaluate_fraud_signals``
    - ``generate_decision``

    Metadata conventions
    --------------------
    The ``metadata`` dict is intentionally untyped.  Recommended keys by
    agent type:

    - All agents: ``claim_id``, ``document_id``, ``processing_time_ms``
    - Classification / OCR: ``confidence``, ``document_type``, ``mime_type``
    - Policy validation: ``rule_name``, ``validation_status``, ``reason``
    - Fraud detection: ``fraud_score``, ``signal_count``, ``signal_types``
    - Decision: ``decision``, ``approved_amount``, ``confidence_score``
    - Error events: ``error_type``, ``error_message``, ``stack_trace``
    """

    model_config = _SHARED_CONFIG

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "Globally unique identifier for this event, generated as a UUID4 "
            "string.  Auto-generated at construction time when not supplied.  "
            "Used by the UI timeline and audit log to deduplicate and order "
            "events."
        ),
        examples=["a1b2c3d4-e5f6-7890-abcd-ef1234567890"],
    )

    agent_name: str = Field(
        ...,
        description=(
            "Name of the agent that emitted this event.  Should be the class "
            "name of the emitting agent (e.g. 'DocumentClassificationAgent', "
            "'PolicyValidationAgent').  Must be non-empty after whitespace "
            "trimming."
        ),
        examples=[
            "DocumentClassificationAgent",
            "OCRAgent",
            "PolicyValidationAgent",
            "FraudDetectionAgent",
            "DecisionAgent",
        ],
    )

    action: str = Field(
        ...,
        description=(
            "Snake_case verb phrase describing the operation the agent was "
            "performing when this event was emitted (e.g. "
            "'classify_document', 'validate_waiting_period', "
            "'generate_decision').  Must be non-empty after whitespace "
            "trimming."
        ),
        examples=[
            "classify_document",
            "extract_prescription",
            "validate_waiting_period",
            "evaluate_fraud_signals",
            "generate_decision",
        ],
    )

    status: TraceStatus = Field(
        ...,
        description=(
            "Execution outcome of the operation at the moment this event was "
            "emitted.  ``STARTED`` events are paired with a terminal event "
            "(``PASSED``, ``FAILED``, ``PARTIAL``, or ``SKIPPED``) once the "
            "operation completes."
        ),
    )

    severity: TraceSeverity = Field(
        ...,
        description=(
            "Importance level of this event.  ``INFO`` for routine progress; "
            "``WARNING`` for non-fatal anomalies; ``ERROR`` for component "
            "failures that degrade output quality; ``CRITICAL`` for failures "
            "that force manual review."
        ),
    )

    message: str = Field(
        ...,
        description=(
            "Human-readable explanation of what happened at this step.  Must "
            "be specific enough to stand alone in an audit log or operations "
            "dashboard without additional context.  Must be non-empty after "
            "whitespace trimming."
        ),
        examples=[
            "Document F001 classified as PRESCRIPTION with confidence 0.97.",
            "Waiting period check failed: diabetes treatment on 2024-10-15 "
            "falls within the 90-day waiting period (expires 2024-11-30).",
            "OCR extraction partially succeeded: doctor registration number "
            "obscured by rubber stamp; confidence reduced to 0.61.",
            "LLM timeout after 30s; OCR step skipped for document F004.",
        ],
    )

    timestamp: datetime = Field(
        ...,
        description=(
            "UTC timestamp at which this event was emitted by the agent.  "
            "Events within a ``TraceTimeline`` should be ordered by this "
            "field to reconstruct the chronological claim journey."
        ),
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary key-value context attached by the emitting agent to "
            "support explainability, debugging, and performance analysis.  "
            "Common keys: ``claim_id``, ``document_id``, ``confidence``, "
            "``processing_time_ms``, ``error_message``, ``rule_name``, "
            "``decision``, ``approved_amount``.  Empty dict when no "
            "supplementary data is needed."
        ),
        examples=[
            {
                "claim_id": "CLM_2024_001",
                "document_id": "F007",
                "confidence": 0.97,
                "processing_time_ms": 842,
            },
            {
                "rule_name": "WAITING_PERIOD_CHECK",
                "condition": "diabetes",
                "waiting_period_days": 90,
                "treatment_date": "2024-10-15",
                "eligible_from": "2024-11-30",
            },
        ],
    )

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @field_validator("event_id", mode="before")
    @classmethod
    def _validate_event_id(cls, value: Any) -> str:
        """
        Accept a caller-supplied event ID or auto-generate a UUID4.

        When the caller supplies a value it is stripped and validated
        non-empty.  When ``None`` is supplied explicitly the field default
        factory kicks in and generates a fresh UUID4.
        """
        if value is None:
            return str(uuid.uuid4())
        return _require_nonempty_str("event_id", value)

    @field_validator("agent_name", mode="before")
    @classmethod
    def _validate_agent_name(cls, value: Any) -> str:
        """Strip whitespace and reject empty agent names."""
        return _require_nonempty_str("agent_name", value)

    @field_validator("action", mode="before")
    @classmethod
    def _validate_action(cls, value: Any) -> str:
        """Strip whitespace and reject empty action strings."""
        return _require_nonempty_str("action", value)

    @field_validator("message", mode="before")
    @classmethod
    def _validate_message(cls, value: Any) -> str:
        """Strip whitespace and reject empty message strings."""
        return _require_nonempty_str("message", value)

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def is_failure(self) -> bool:
        """
        Return ``True`` when this event records an operation failure.

        Specifically returns ``True`` only for ``TraceStatus.FAILED``.
        ``PARTIAL`` events are not failures; they indicate degraded-but-usable
        output and are handled separately.
        """
        return self.status is TraceStatus.FAILED

    @property
    def is_warning(self) -> bool:
        """
        Return ``True`` when this event carries ``TraceSeverity.WARNING``.

        Warning events indicate non-fatal anomalies that should be surfaced
        in the operations dashboard without blocking automated processing.
        """
        return self.severity is TraceSeverity.WARNING

    @property
    def is_critical(self) -> bool:
        """
        Return ``True`` when this event carries ``TraceSeverity.CRITICAL``.

        Critical events indicate failures that require immediate manual
        intervention.  Consumers that scan a ``TraceTimeline`` for critical
        events can use this property for fast filtering without switching on
        enum values.
        """
        return self.severity is TraceSeverity.CRITICAL


# ---------------------------------------------------------------------------
# AgentExecutionSummary
# ---------------------------------------------------------------------------


class AgentExecutionSummary(BaseModel):
    """
    Compact summary of a single agent's execution within a claim pipeline run.

    One ``AgentExecutionSummary`` is produced per agent per claim.  It
    aggregates the agent's overall outcome and timing so that the operations
    dashboard can render a high-level pipeline health view without loading the
    full ``TraceTimeline``.

    The ``started_at`` and ``completed_at`` fields may be ``None`` when a
    component failed before it could record its timestamps, ensuring the model
    can still be constructed in degraded states.  ``duration_ms`` is similarly
    optional to avoid forcing callers to handle timestamp arithmetic when one
    boundary is missing.
    """

    model_config = _SHARED_CONFIG

    agent_name: str = Field(
        ...,
        description=(
            "Name of the agent this summary describes.  Matches the "
            "``agent_name`` field on the ``TraceEvent`` objects emitted by "
            "this agent.  Must be non-empty after whitespace trimming."
        ),
        examples=["DocumentClassificationAgent", "PolicyValidationAgent"],
    )

    status: TraceStatus = Field(
        ...,
        description=(
            "Overall execution status of the agent for this claim run.  "
            "Typically derived from the terminal ``TraceEvent`` emitted by "
            "the agent."
        ),
    )

    started_at: datetime | None = Field(
        default=None,
        description=(
            "UTC timestamp at which the agent began processing.  ``None`` "
            "when the agent failed before it could record a start time (e.g. "
            "it was never invoked due to an upstream failure)."
        ),
    )

    completed_at: datetime | None = Field(
        default=None,
        description=(
            "UTC timestamp at which the agent finished processing.  ``None`` "
            "when the agent is still running or when it crashed before "
            "recording a completion time."
        ),
    )

    duration_ms: int | None = Field(
        default=None,
        description=(
            "Wall-clock time the agent spent processing, in milliseconds.  "
            "Must be >= 0 when present.  ``None`` when timing information is "
            "unavailable (e.g. the agent was skipped or crashed)."
        ),
        examples=[842, 1203, 0, None],
    )

    events_count: int = Field(
        ...,
        description=(
            "Number of ``TraceEvent`` objects emitted by this agent during "
            "this claim run.  Must be >= 0.  A value of 0 indicates the agent "
            "was registered in the pipeline but emitted no events (e.g. it "
            "was bypassed before it could start)."
        ),
        examples=[3, 7, 0],
    )

    @field_validator("agent_name", mode="before")
    @classmethod
    def _validate_agent_name(cls, value: Any) -> str:
        """Strip whitespace and reject empty agent names."""
        return _require_nonempty_str("agent_name", value)

    @field_validator("duration_ms", mode="before")
    @classmethod
    def _validate_duration_ms(cls, value: Any) -> int | None:
        """Accept ``None`` or a non-negative integer for duration."""
        if value is None:
            return None
        return _require_nonnegative_int("duration_ms", value)

    @field_validator("events_count", mode="before")
    @classmethod
    def _validate_events_count(cls, value: Any) -> int:
        """Reject negative event counts."""
        return _require_nonnegative_int("events_count", value)


# ---------------------------------------------------------------------------
# TraceTimeline
# ---------------------------------------------------------------------------


class TraceTimeline(BaseModel):
    """
    Complete ordered execution history for a single claim pipeline run.

    A ``TraceTimeline`` is the primary artifact consumed by the UI timeline
    renderer, the explainability system, and the evaluation framework.  It
    holds every ``TraceEvent`` emitted by every agent during the processing
    of one claim, in the order they were appended.

    The timeline is intentionally append-only in practice: agents push events
    as they execute, and the final serialised timeline is stored alongside the
    ``ClaimDecision`` so that the full claim journey is always retrievable from
    a single object.

    Ordering guarantee
    ------------------
    Events should be appended in chronological order.  Consumers that need
    strict ordering should sort by ``TraceEvent.timestamp`` before rendering.
    ``generated_at`` records when the timeline object itself was created, not
    when the last event was emitted.
    """

    model_config = _SHARED_CONFIG

    events: list[TraceEvent] = Field(
        default_factory=list,
        description=(
            "Ordered list of all ``TraceEvent`` objects emitted during the "
            "claim pipeline run.  Append-only in normal operation.  An empty "
            "list indicates no events have been recorded yet (e.g. the "
            "timeline was constructed before any agent ran)."
        ),
    )

    generated_at: datetime = Field(
        ...,
        description=(
            "UTC timestamp at which this ``TraceTimeline`` object was "
            "instantiated.  Used for audit log ordering and to detect "
            "timelines that were created but never populated."
        ),
    )

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def total_events(self) -> int:
        """
        Return the number of events recorded in this timeline.

        Equivalent to ``len(self.events)``.  Provided as a computed property
        so that callers do not need to know the internal structure of the
        model to answer the question "how many events did this claim produce?"
        """
        return len(self.events)


# ---------------------------------------------------------------------------
# TraceSummary
# ---------------------------------------------------------------------------


class TraceSummary(BaseModel):
    """
    High-level observability snapshot derived from a ``TraceTimeline``.

    ``TraceSummary`` is the lightweight model consumed by the operations
    dashboard for at-a-glance pipeline health.  It aggregates event counts
    by severity so that reviewers can quickly identify claims with errors or
    warnings without loading the full ``TraceTimeline``.

    Construction
    ------------
    This model is populated by the Explainability Agent after the pipeline
    completes by iterating over the ``TraceTimeline`` events.  It is not
    computed lazily here to keep the model a pure data contract.

    All counters must be non-negative.  A ``TraceSummary`` where
    ``failed_events + critical_events == 0`` indicates a clean pipeline run
    with no unhandled errors.
    """

    model_config = _SHARED_CONFIG

    total_events: int = Field(
        ...,
        description=(
            "Total number of ``TraceEvent`` objects in the corresponding "
            "``TraceTimeline``.  Must be >= 0."
        ),
        examples=[12, 0],
    )

    passed_events: int = Field(
        ...,
        description=(
            "Number of events whose ``status`` is ``TraceStatus.PASSED``.  "
            "Must be >= 0."
        ),
        examples=[8, 0],
    )

    failed_events: int = Field(
        ...,
        description=(
            "Number of events whose ``status`` is ``TraceStatus.FAILED``.  "
            "Must be >= 0.  A non-zero value indicates at least one component "
            "failure occurred during processing."
        ),
        examples=[1, 0],
    )

    warning_events: int = Field(
        ...,
        description=(
            "Number of events whose ``severity`` is ``TraceSeverity.WARNING``.  "
            "Must be >= 0.  Warning events do not block processing but should "
            "be surfaced in the operations dashboard."
        ),
        examples=[2, 0],
    )

    critical_events: int = Field(
        ...,
        description=(
            "Number of events whose ``severity`` is ``TraceSeverity.CRITICAL``.  "
            "Must be >= 0.  A non-zero value indicates at least one condition "
            "that required immediate manual intervention."
        ),
        examples=[0, 1],
    )

    @field_validator(
        "total_events",
        "passed_events",
        "failed_events",
        "warning_events",
        "critical_events",
        mode="before",
    )
    @classmethod
    def _validate_counts(cls, value: Any) -> int:
        """
        Coerce each counter to a non-negative integer.

        All five counter fields share this validator.  Pydantic will call it
        independently for each field, passing the field value as ``value``.
        """
        try:
            int_value = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Event count must be a non-negative integer, got {value!r}."
            ) from exc
        if int_value < 0:
            raise ValueError(
                f"Event count must be >= 0, got {int_value}."
            )
        return int_value