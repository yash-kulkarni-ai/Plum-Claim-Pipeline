"""
Core claim contracts and shared workflow state for the Plum Health Insurance
Claims Processing System.

This module defines the canonical business objects that travel through the
entire multi-agent pipeline.  It is the single authoritative schema that
every agent — Document Classification, Verification, OCR, Policy Validation,
Fraud Detection, Decision, and Explainability — reads from and writes to.

Architecture position
---------------------
::

    Frontend Submission
        ↓  ClaimSubmission
    LangGraph Graph Entry
        ↓
    ClaimState  ← this module; mutated at every node
        │
        ├── DocumentClassificationAgent  (populates document_results)
        ├── DocumentVerificationAgent    (populates document_results)
        ├── OCRAgent                     (populates extractions)
        ├── PolicyValidationAgent        (populates validations)
        ├── FraudDetectionAgent          (populates fraud_result)
        ├── DecisionAgent                (populates decision)
        └── ExplainabilityAgent          (reads trace_events; builds summary)
        ↓
    Final HTTP Response

Design principles
-----------------
- ``ClaimState`` is the single source of truth.  A future engineer inspecting
  a serialised ``ClaimState`` snapshot should be able to reconstruct the full
  claim journey — what was submitted, what was extracted, what validations
  ran, what errors occurred, and why the final decision was made — without
  consulting any other object.
- Every Pydantic model uses ``extra="forbid"`` to catch schema drift between
  agents at the earliest possible boundary.
- ``Decimal`` is the exclusive type for all monetary fields.
- Auto-generated UUIDs are used wherever unique identity is needed but the
  caller should not have to supply it.
- No business logic, policy rules, fraud algorithms, or LangGraph wiring
  lives in this file.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.models.decision import ClaimDecision, FraudResult, ValidationResult
from backend.app.models.document import UploadedDocument
from backend.app.models.extraction import ExtractionResult
from backend.app.models.trace import TraceEvent


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

    Raises ``ValueError`` naming the offending field so that Pydantic
    surfaces a precise, actionable error to the caller.
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


# ---------------------------------------------------------------------------
# ClaimCategory
# ---------------------------------------------------------------------------


class ClaimCategory(StrEnum):
    """
    Supported health insurance claim categories.

    These values mirror the ``opd_categories`` keys defined in the policy
    configuration (``policy_terms.json``).  The Policy Validation Agent maps
    the ``ClaimCategory`` on a ``ClaimSubmission`` directly to the
    corresponding policy sub-limits, co-pay rules, document requirements, and
    coverage flags stored in that configuration.

    Using ``StrEnum`` ensures every value serialises to its plain string form
    in JSON without a custom encoder and that comparisons against raw strings
    work transparently across the codebase.

    Values
    ------
    CONSULTATION
        General practitioner or specialist outpatient consultation.  Requires
        a prescription and a hospital bill.  Subject to a 10% co-pay and a
        per-visit sub-limit.
    DIAGNOSTIC
        Laboratory or imaging investigations ordered by a doctor.  Requires a
        prescription, a lab report, and a hospital bill.  High-value imaging
        tests (MRI, CT, PET) above the pre-auth threshold additionally require
        prior authorisation.
    PHARMACY
        Prescription medicine purchase from a licensed pharmacy.  Requires a
        valid prescription and a pharmacy bill.  Generic substitution is
        mandatory; branded-drug co-pay applies.
    DENTAL
        Dental procedures performed by a registered dental practitioner.
        Covered procedures (e.g. root canal, extractions) are reimbursed;
        cosmetic procedures (e.g. whitening, orthodontics) are excluded.
    VISION
        Spectacles, contact lenses, eye examinations, and cataract surgery.
        LASIK and refractive surgery are excluded.
    ALTERNATIVE_MEDICINE
        Treatments under recognised alternative medicine systems: Ayurveda,
        Homeopathy, Unani, Siddha, and Naturopathy.  Requires a prescription
        from a registered practitioner.  Subject to a session limit per year.
    """

    CONSULTATION = "CONSULTATION"
    DIAGNOSTIC = "DIAGNOSTIC"
    PHARMACY = "PHARMACY"
    DENTAL = "DENTAL"
    VISION = "VISION"
    ALTERNATIVE_MEDICINE = "ALTERNATIVE_MEDICINE"


# ---------------------------------------------------------------------------
# ClaimStatus
# ---------------------------------------------------------------------------


class ClaimStatus(StrEnum):
    """
    Processing lifecycle status of a claim as it travels through the pipeline.

    This enum tracks *workflow* state, not *decision* state.  It answers the
    question "where is this claim in the pipeline right now?" rather than
    "was the claim approved?"  The final decision outcome is carried
    separately by ``ClaimDecision.decision`` (a ``DecisionType`` enum).

    Values
    ------
    SUBMITTED
        The claim has been received from the frontend and a ``ClaimState``
        object has been created, but no agent has begun processing.
    IN_PROGRESS
        At least one agent is actively processing the claim.  This status
        persists until the pipeline either completes normally or fails
        unrecoverably.
    COMPLETED
        The pipeline ran to completion and the Decision Agent produced a
        ``ClaimDecision``.  The claim's ``decision`` field is populated.
        Note: a ``COMPLETED`` claim may still have been decided
        ``MANUAL_REVIEW`` — "completed" means the system finished, not that
        the claim was approved.
    FAILED
        The pipeline encountered an unrecoverable error that prevented a
        decision from being produced.  One or more entries in
        ``ClaimState.errors`` will describe what went wrong.
    MANUAL_REVIEW
        The claim has been flagged for human review and is waiting in the
        operations queue.  This status is set when the Decision Agent produces
        ``DecisionType.MANUAL_REVIEW``, or when a ``CRITICAL`` trace event
        forces early escalation before the pipeline completes.
    """

    SUBMITTED = "SUBMITTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


# ---------------------------------------------------------------------------
# ClaimSubmission
# ---------------------------------------------------------------------------


class ClaimSubmission(BaseModel):
    """
    Raw payload received from the frontend when a member files a claim.

    ``ClaimSubmission`` is the first business object that enters the system.
    It is constructed by the API ingestion layer from the multipart form data
    submitted by the member and is embedded unchanged in ``ClaimState`` for
    the lifetime of the pipeline run.  Agents must never mutate this object;
    it is an immutable record of what the member originally submitted.

    Document list invariant
    -----------------------
    At least one ``UploadedDocument`` must be present.  The Document
    Verification Agent will subsequently check whether the *correct* document
    types are present for the ``claim_category``; that check happens in the
    agent, not here.  This model only enforces that the member submitted
    something.

    Amount invariant
    ----------------
    ``claimed_amount`` must be strictly greater than zero.  The policy
    minimum-claim-amount check (₹500 per ``submission_rules``) is enforced
    by the Policy Validation Agent, not here; this model only rejects
    physically nonsensical zero or negative amounts.
    """

    model_config = _SHARED_CONFIG

    claim_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description=(
            "Globally unique identifier for this claim, auto-generated as a "
            "UUID4 string when not supplied by the caller.  Propagated "
            "unchanged through every agent and stored in all ``TraceEvent`` "
            "metadata payloads so that events from different agents can be "
            "correlated back to the same claim."
        ),
        examples=["clm_7f3a1b2c-4d5e-6f78-90ab-cdef01234567"],
    )

    member_id: str = Field(
        ...,
        description=(
            "Identifier of the insured member filing the claim.  Must match "
            "an entry in the policy roster (``policy_terms.json`` members "
            "list).  Membership validation is performed by the Policy "
            "Validation Agent; this model only rejects empty values."
        ),
        examples=["EMP001", "EMP007", "DEP002"],
    )

    policy_id: str = Field(
        ...,
        description=(
            "Identifier of the group health insurance policy under which the "
            "claim is being filed.  Must match the ``policy_id`` field in the "
            "loaded policy configuration.  Policy existence is verified by the "
            "Policy Validation Agent."
        ),
        examples=["PLUM_GHI_2024"],
    )

    claim_category: ClaimCategory = Field(
        ...,
        description=(
            "The OPD category this claim falls under.  Used by the Document "
            "Verification Agent to determine which document types are required "
            "and by the Policy Validation Agent to apply the appropriate "
            "sub-limit, co-pay, and coverage rules."
        ),
    )

    claimed_amount: Decimal = Field(
        ...,
        description=(
            "Total amount the member is claiming for reimbursement, in INR.  "
            "Must be strictly greater than zero.  The policy minimum-claim "
            "threshold check (₹500) is applied separately by the Policy "
            "Validation Agent."
        ),
        examples=[Decimal("1500.00"), Decimal("4500.00"), Decimal("12000.00")],
    )

    treatment_date: date = Field(
        ...,
        description=(
            "Date on which the treatment, procedure, or purchase took place, "
            "as reported by the member.  Used by the Policy Validation Agent "
            "to evaluate waiting periods, submission deadlines (30-day rule), "
            "and policy coverage dates.  Must be a valid calendar date."
        ),
        examples=["2024-11-01", "2024-10-15"],
    )

    documents: list[UploadedDocument] = Field(
        ...,
        description=(
            "One or more medical documents uploaded by the member to support "
            "the claim.  Must contain at least one document.  The Document "
            "Verification Agent will validate that the correct document types "
            "are present for the ``claim_category``."
        ),
        min_length=1,
    )

    submitted_at: datetime = Field(
        ...,
        description=(
            "UTC timestamp at which the claim was received by the API "
            "ingestion layer.  Used for audit trail ordering and for "
            "evaluating the 30-day submission deadline against "
            "``treatment_date``."
        ),
    )

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @field_validator("claim_id", mode="before")
    @classmethod
    def _validate_claim_id(cls, value: Any) -> str:
        """Accept a caller-supplied ID (stripped) or auto-generate UUID4."""
        if value is None:
            return str(uuid.uuid4())
        return _require_nonempty_str("claim_id", value)

    @field_validator("member_id", mode="before")
    @classmethod
    def _validate_member_id(cls, value: Any) -> str:
        """Strip whitespace and reject empty member identifiers."""
        return _require_nonempty_str("member_id", value)

    @field_validator("policy_id", mode="before")
    @classmethod
    def _validate_policy_id(cls, value: Any) -> str:
        """Strip whitespace and reject empty policy identifiers."""
        return _require_nonempty_str("policy_id", value)

    @field_validator("claimed_amount", mode="before")
    @classmethod
    def _validate_claimed_amount(cls, value: Any) -> Decimal:
        """Coerce to Decimal and reject zero or negative amounts."""
        if isinstance(value, Decimal):
            decimal_value = value
        else:
            try:
                decimal_value = Decimal(str(value))
            except Exception as exc:
                raise ValueError(
                    f"claimed_amount must be a positive number, got {value!r}."
                ) from exc
        if decimal_value <= Decimal("0"):
            raise ValueError(
                f"claimed_amount must be > 0, got {decimal_value}."
            )
        return decimal_value


# ---------------------------------------------------------------------------
# ProcessingError
# ---------------------------------------------------------------------------


class ProcessingError(BaseModel):
    """
    Records a recoverable failure that occurred during pipeline execution.

    ``ProcessingError`` is used to capture component-level failures that did
    not crash the overall pipeline but that degraded its output quality.
    Examples include LLM timeouts, Gemini Vision API errors, JSON parsing
    failures, and transient network errors.

    Recoverability flag
    -------------------
    ``recoverable=True`` indicates that the pipeline continued with whatever
    partial output was available and adjusted its confidence score
    accordingly.  ``recoverable=False`` indicates that the failure was
    terminal for that component, forcing the claim to either skip downstream
    steps that depended on that component's output or escalate to
    ``MANUAL_REVIEW``.

    This model does not carry stack traces by design.  Agent code should log
    the full exception through the standard logging infrastructure; the
    ``message`` field here carries only the human-readable summary that
    surfaces in the claim audit trail and the operations dashboard.
    """

    model_config = _SHARED_CONFIG

    component: str = Field(
        ...,
        description=(
            "Name of the pipeline component or agent that encountered the "
            "failure (e.g. 'OCRAgent', 'PolicyValidationAgent', "
            "'GeminiVisionClient').  Must be non-empty after whitespace "
            "trimming."
        ),
        examples=["OCRAgent", "GeminiVisionClient", "PolicyValidationAgent"],
    )

    message: str = Field(
        ...,
        description=(
            "Human-readable description of what went wrong.  Must be specific "
            "enough to appear in the claim audit log without additional "
            "context.  Must be non-empty after whitespace trimming."
        ),
        examples=[
            "Gemini Vision API timed out after 30s on document F004.",
            "Failed to parse extraction JSON: missing required field "
            "'patient.name'.",
            "Policy configuration not found for policy_id 'PLUM_GHI_2024'.",
        ],
    )

    recoverable: bool = Field(
        ...,
        description=(
            "Whether the pipeline was able to continue processing after this "
            "failure.  ``True`` means the affected step was skipped or "
            "substituted and the pipeline proceeded with reduced confidence.  "
            "``False`` means this failure was terminal for the affected "
            "component and downstream steps that depended on its output could "
            "not run."
        ),
    )

    timestamp: datetime = Field(
        ...,
        description=(
            "UTC timestamp at which the failure was recorded by the agent.  "
            "Used to correlate errors with ``TraceEvent`` timestamps in the "
            "audit trail."
        ),
    )

    @field_validator("component", mode="before")
    @classmethod
    def _validate_component(cls, value: Any) -> str:
        """Strip whitespace and reject empty component names."""
        return _require_nonempty_str("component", value)

    @field_validator("message", mode="before")
    @classmethod
    def _validate_message(cls, value: Any) -> str:
        """Strip whitespace and reject empty error messages."""
        return _require_nonempty_str("message", value)


# ---------------------------------------------------------------------------
# ProcessingWarning
# ---------------------------------------------------------------------------


class ProcessingWarning(BaseModel):
    """
    Records a degraded-but-non-fatal condition encountered during pipeline
    execution.

    ``ProcessingWarning`` differs from ``ProcessingError`` in that it does
    not indicate component failure — the step completed, but the output
    carries reduced reliability.  Warnings are surfaced in the operations
    dashboard and included in the claim audit trail so that human reviewers
    have full visibility into extraction quality and processing anomalies.

    Common warning scenarios
    ------------------------
    - OCR confidence below the acceptable floor on a required field.
    - An optional document was missing (e.g. no dental report for a dental
      claim, which is not required but would have improved confidence).
    - A document-alteration signal was detected (crossed-out amounts,
      multiple ORIGINAL stamps) but did not meet the fraud threshold.
    - A required field was extracted but its value appears implausible
      (e.g. a bill total that does not match the sum of its line items).
    """

    model_config = _SHARED_CONFIG

    source: str = Field(
        ...,
        description=(
            "Name of the agent or component that raised this warning (e.g. "
            "'OCRAgent', 'DocumentVerificationAgent').  Must be non-empty "
            "after whitespace trimming."
        ),
        examples=["OCRAgent", "DocumentVerificationAgent", "FraudDetectionAgent"],
    )

    message: str = Field(
        ...,
        description=(
            "Human-readable description of the degraded condition.  Must be "
            "specific enough to stand alone in the operations dashboard.  "
            "Must be non-empty after whitespace trimming."
        ),
        examples=[
            "Doctor registration number partially obscured by rubber stamp on "
            "document F007; extracted value may be incomplete.",
            "Bill total ₹1,500 does not match sum of line items ₹1,450; "
            "using line item sum.",
            "Pharmacy bill F004 is blurry; confidence reduced to 0.38.",
        ],
    )

    timestamp: datetime = Field(
        ...,
        description=(
            "UTC timestamp at which the warning was recorded by the source "
            "agent.  Used to correlate warnings with ``TraceEvent`` timestamps."
        ),
    )

    @field_validator("source", mode="before")
    @classmethod
    def _validate_source(cls, value: Any) -> str:
        """Strip whitespace and reject empty source names."""
        return _require_nonempty_str("source", value)

    @field_validator("message", mode="before")
    @classmethod
    def _validate_message(cls, value: Any) -> str:
        """Strip whitespace and reject empty warning messages."""
        return _require_nonempty_str("message", value)


# ---------------------------------------------------------------------------
# ClaimState
# ---------------------------------------------------------------------------


class ClaimState(BaseModel):
    """
    Complete workflow state shared across every node in the LangGraph pipeline.

    ``ClaimState`` is the most important model in the entire system.  It is
    the single source of truth for a claim's processing journey: every agent
    receives the current state, performs its work, and returns an updated
    state.  The final serialised ``ClaimState`` contains everything needed to
    understand what the system did and why.

    Lifecycle
    ---------
    1. **Creation** – The API ingestion layer constructs ``ClaimState`` from
       the incoming ``ClaimSubmission``.  Status is ``SUBMITTED``.
       ``confidence_score`` starts at ``1.0``.  All list fields are empty.

    2. **Document Classification / Verification** – The Document agents
       populate ``document_results`` and append ``TraceEvent`` objects.
       If required documents are missing or unreadable, the pipeline may
       terminate early and set ``status = FAILED`` or escalate to
       ``MANUAL_REVIEW``.

    3. **OCR Extraction** – The OCR Agent populates ``extractions`` (one
       ``ExtractionResult`` per processed document).  Any failures are
       appended to ``errors``; low-confidence extractions produce ``warnings``.
       ``confidence_score`` is decreased proportionally.

    4. **Policy Validation** – The Policy Agent populates ``validations``
       (one ``ValidationResult`` per rule evaluated).  Failed rules are
       reflected in the final ``decision``.

    5. **Fraud Detection** – The Fraud Agent populates ``fraud_result``.  A
       high fraud score or a ``requires_manual_review`` flag forces
       ``status = MANUAL_REVIEW``.

    6. **Decision** – The Decision Agent reads all accumulated state and
       populates ``decision``.  ``status`` transitions to ``COMPLETED``,
       ``FAILED``, or ``MANUAL_REVIEW``.

    7. **Explainability** – The Explainability Agent reads ``trace_events``
       and produces a human-readable audit summary for the operations team.

    Mutation contract
    -----------------
    Agents return a new ``ClaimState`` or a dict of field updates; they do
    not mutate the shared object in-place except through LangGraph's
    state-update mechanism.  This contract is enforced by LangGraph's node
    semantics, not by this model.

    Confidence score semantics
    --------------------------
    ``confidence_score`` starts at ``1.0`` and is reduced by agents when
    they encounter degraded conditions:

    - OCR confidence below threshold → subtract weighted penalty.
    - Required field not extracted → subtract penalty.
    - Policy validation rule skipped due to component failure → subtract
      penalty.
    - Fraud score above soft threshold but below hard threshold → subtract
      penalty.

    The Decision Agent uses the final ``confidence_score`` when populating
    ``ClaimDecision.confidence_score``.  A score below ``0.5`` typically
    triggers ``MANUAL_REVIEW``.
    """

    model_config = _SHARED_CONFIG

    claim: ClaimSubmission = Field(
        ...,
        description=(
            "The original claim submission as received from the frontend.  "
            "This field is set once at state creation and must not be "
            "modified by any downstream agent.  It is the immutable record "
            "of what the member submitted."
        ),
    )

    status: ClaimStatus = Field(
        default=ClaimStatus.SUBMITTED,
        description=(
            "Current workflow lifecycle status of the claim.  Starts at "
            "``SUBMITTED`` and transitions through ``IN_PROGRESS`` → "
            "``COMPLETED`` (or ``FAILED`` / ``MANUAL_REVIEW``) as the "
            "pipeline executes.  Updated by each agent at the start and end "
            "of its work."
        ),
    )

    document_results: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Structured outputs from the Document Classification Agent and "
            "the Document Verification Agent.  Each entry is a dict "
            "representing one document's classification and verification "
            "result.  The schema of each entry is defined by the agents "
            "themselves; a future refactor may introduce a dedicated typed "
            "model.  One entry per processed document."
        ),
    )

    extractions: list[ExtractionResult] = Field(
        default_factory=list,
        description=(
            "OCR extraction results produced by the OCR Agent, one per "
            "successfully processed document.  Each ``ExtractionResult`` "
            "carries the structured data extracted from a single document "
            "along with the extraction confidence and any non-fatal "
            "extraction warnings."
        ),
    )

    validations: list[ValidationResult] = Field(
        default_factory=list,
        description=(
            "Policy rule check results produced by the Policy Validation "
            "Agent, one per rule evaluated.  The ordered list of "
            "``ValidationResult`` objects forms the policy explainability "
            "trace: any reviewer can see exactly which rules ran, what they "
            "concluded, and why."
        ),
    )

    fraud_result: FraudResult | None = Field(
        default=None,
        description=(
            "Complete output of the Fraud Detection Agent.  ``None`` until "
            "the Fraud Agent has run.  When present, "
            "``fraud_result.requires_manual_review`` is the authoritative "
            "flag for fraud-based escalation."
        ),
    )

    decision: ClaimDecision | None = Field(
        default=None,
        description=(
            "Final claim decision produced by the Decision Agent.  ``None`` "
            "until the Decision Agent has run.  When present, this field "
            "contains the complete decision record including approved amount, "
            "confidence score, reason, financial breakdown, validation trace, "
            "and fraud result."
        ),
    )

    trace_events: list[TraceEvent] = Field(
        default_factory=list,
        description=(
            "Accumulated audit trail for this claim run.  Every agent "
            "appends one or more ``TraceEvent`` objects here as it executes.  "
            "The ordered list of events provides a complete, chronological "
            "record of the claim journey from submission to decision."
        ),
    )

    confidence_score: float = Field(
        default=1.0,
        description=(
            "Global claim processing confidence in the range [0.0, 1.0].  "
            "Initialised at 1.0 (full confidence) and decreased by agents "
            "when they encounter degraded conditions: low OCR confidence, "
            "unreadable documents, skipped validation steps, or component "
            "failures.  The Decision Agent reads this value when producing "
            "``ClaimDecision.confidence_score``.  A value below 0.5 "
            "typically triggers ``MANUAL_REVIEW``."
        ),
        examples=[1.0, 0.87, 0.52, 0.31],
    )

    errors: list[ProcessingError] = Field(
        default_factory=list,
        description=(
            "Recoverable failures recorded by agents during pipeline "
            "execution.  A non-empty list does not necessarily mean the "
            "claim cannot be decided; it means the pipeline continued with "
            "degraded output and reduced confidence.  Each entry describes "
            "which component failed, what went wrong, and whether processing "
            "continued."
        ),
    )

    warnings: list[ProcessingWarning] = Field(
        default_factory=list,
        description=(
            "Non-fatal anomalies recorded by agents during pipeline "
            "execution.  Examples: low OCR confidence, missing optional "
            "documents, bill total mismatch, rubber stamp over fields.  "
            "Warnings are surfaced in the operations dashboard and included "
            "in the claim audit trail."
        ),
    )

    created_at: datetime = Field(
        ...,
        description=(
            "UTC timestamp at which this ``ClaimState`` object was created "
            "by the API ingestion layer.  Immutable after construction."
        ),
    )

    updated_at: datetime = Field(
        ...,
        description=(
            "UTC timestamp of the most recent state update.  Set by each "
            "agent when it returns an updated state.  Used for SLA monitoring "
            "and stale-pipeline detection."
        ),
    )

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _validate_confidence_score(cls, value: Any) -> float:
        """Reject confidence scores outside [0.0, 1.0]."""
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"confidence_score must be a float in [0.0, 1.0], got {value!r}."
            ) from exc
        if not (0.0 <= score <= 1.0):
            raise ValueError(
                f"confidence_score must be in the range [0.0, 1.0], got {score}."
            )
        return score

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def has_errors(self) -> bool:
        """
        Return ``True`` when at least one ``ProcessingError`` has been
        recorded during the pipeline run.

        A ``True`` value does not necessarily mean the claim cannot be
        decided; it indicates that at least one component failed
        recoverably.  Reviewers should consult ``errors`` for details.
        """
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """
        Return ``True`` when at least one ``ProcessingWarning`` has been
        recorded during the pipeline run.

        Warnings indicate non-fatal degradation of processing quality.
        Callers can use this property as a fast gate before iterating over
        ``warnings`` for display.
        """
        return len(self.warnings) > 0

    @property
    def is_complete(self) -> bool:
        """
        Return ``True`` when the Decision Agent has produced a final
        ``ClaimDecision``.

        A ``True`` value means the pipeline ran to completion and
        ``decision`` is populated.  It does not imply the claim was
        approved; it only means an automated outcome (including
        ``MANUAL_REVIEW``) was produced.
        """
        return self.decision is not None

    @property
    def trace_count(self) -> int:
        """
        Return the total number of ``TraceEvent`` objects accumulated in
        the audit trail so far.

        Equivalent to ``len(self.trace_events)``.  Provided for conciseness
        in conditional expressions and log statements.
        """
        return len(self.trace_events)

    @property
    def validation_count(self) -> int:
        """
        Return the number of policy rule checks that have been evaluated.

        Equivalent to ``len(self.validations)``.  A value of zero means
        the Policy Validation Agent has not yet run.
        """
        return len(self.validations)

    @property
    def extraction_count(self) -> int:
        """
        Return the number of documents for which OCR extraction has
        completed.

        Equivalent to ``len(self.extractions)``.  Comparing this value
        against ``len(self.claim.documents)`` gives a quick read on how
        much of the document set has been processed.
        """
        return len(self.extractions)