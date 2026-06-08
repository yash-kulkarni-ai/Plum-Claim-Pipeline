"""
Decision models for the Plum Health Insurance Claims Processing System.

This module defines the strict data contracts that flow between the policy
evaluation layer, fraud detection layer, and the final decision agent.  It
is the schema boundary that every downstream component — including the API
layer and the frontend dashboard — depends on.

Architecture position
---------------------
::

    Extraction Models
        ↓
    Verification Agent
        ↓
    Policy Agent   ──→  ValidationResult  ┐
        ↓                                 │
    Fraud Agent    ──→  FraudResult       ├─→  ClaimDecision
        ↓                                 │
    Decision Agent ──→  DecisionBreakdown ┘
        ↓
    API Layer  ──→  ClaimOutcomeSummary (dashboard)

Design principles
-----------------
- Every model uses ``extra="forbid"`` so that unexpected fields injected by
  upstream agents are caught at the boundary rather than silently forwarded.
- ``validate_assignment=True`` ensures that invariants remain intact when
  agents enrich models post-construction (e.g. attaching a breakdown after
  the payout is computed).
- ``Decimal`` is the exclusive type for all monetary amounts to prevent
  floating-point rounding errors when applying co-pays and sub-limits.
- Confidence, fraud, and severity scores are validated to the closed unit
  interval ``[0.0, 1.0]`` at construction time.
- No business logic, policy rules, or calculation logic lives in this file.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
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
    Strip whitespace from a required string and reject empty values.

    Raises ``ValueError`` with an explicit field name so Pydantic surfaces a
    useful error message to the caller.
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


def _validate_unit_interval(field_name: str, value: Any) -> float:
    """
    Coerce ``value`` to ``float`` and assert it lies in ``[0.0, 1.0]``.

    Raises ``ValueError`` with a message that names the field and the
    offending value.
    """
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a float in [0.0, 1.0], got {value!r}."
        ) from exc
    if not (0.0 <= score <= 1.0):
        raise ValueError(
            f"{field_name} must be in the range [0.0, 1.0], got {score}."
        )
    return score


def _validate_nonnegative_decimal(field_name: str, value: Any) -> Decimal:
    """
    Coerce ``value`` to ``Decimal`` and assert it is >= 0.

    Accepts ``int``, ``float``, ``str``, and ``Decimal`` inputs to
    accommodate varied representations produced by upstream agents.
    """
    if isinstance(value, Decimal):
        decimal_value = value
    else:
        try:
            decimal_value = Decimal(str(value))
        except Exception as exc:
            raise ValueError(
                f"{field_name} must be a non-negative number, got {value!r}."
            ) from exc
    if decimal_value < Decimal("0"):
        raise ValueError(f"{field_name} must be >= 0, got {decimal_value}.")
    return decimal_value


# ---------------------------------------------------------------------------
# DecisionType
# ---------------------------------------------------------------------------


class DecisionType(StrEnum):
    """
    Canonical set of outcomes that the Decision Agent may produce for a claim.

    Using ``StrEnum`` guarantees that every value serialises to its plain
    string representation in JSON without a custom encoder and that
    comparisons against raw strings work transparently.

    Values
    ------
    APPROVED
        The claim satisfies all policy conditions.  The full eligible amount
        after deductions is approved for reimbursement.
    PARTIAL
        The claim is partially eligible.  One or more line items or amounts
        were excluded (e.g. a cosmetic dental procedure within an otherwise
        valid dental claim).  The approved amount is less than the claimed
        amount.
    REJECTED
        The claim does not meet one or more mandatory policy conditions (e.g.
        waiting period not elapsed, excluded condition, pre-auth missing).
        No amount is approved.
    MANUAL_REVIEW
        The system cannot reach a confident automated decision.  The claim
        must be routed to a human operations agent for review.  This outcome
        is produced when fraud signals exceed the configured threshold, when
        a required processing component failed, or when confidence is below
        the acceptable floor.
    """

    APPROVED = "APPROVED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


# ---------------------------------------------------------------------------
# ValidationStatus
# ---------------------------------------------------------------------------


class ValidationStatus(StrEnum):
    """
    Outcome of a single policy or business-rule check performed by the Policy
    Agent.

    Values
    ------
    PASSED
        The rule was evaluated and the claim satisfied it.
    FAILED
        The rule was evaluated and the claim violated it.  A failed rule
        typically blocks approval unless overridden by a higher-priority rule.
    WARNING
        The rule raised a concern that does not by itself block the claim but
        should be surfaced in the audit trail and potentially to the reviewer.
    SKIPPED
        The rule was not applicable to this claim (e.g. a pre-auth check
        skipped because the claimed amount is below the pre-auth threshold)
        or was bypassed due to a component failure.  Skipped rules do not
        count as failures.
    """

    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """
    Outcome of a single policy validation check performed by the Policy Agent.

    Each rule the Policy Agent evaluates — member eligibility, waiting period,
    coverage category, sub-limit, per-claim limit, exclusion check, network
    status, pre-authorisation — produces one ``ValidationResult`` instance.
    The list of results is stored on the final ``ClaimDecision`` and forms the
    complete explainability trace: any human reviewer or audit log can
    reconstruct exactly which rules ran, what they decided, and why.
    """

    model_config = _SHARED_CONFIG

    rule_name: str = Field(
        ...,
        description=(
            "Identifier for the policy rule that was evaluated.  Should be a "
            "stable, human-readable name (e.g. 'WAITING_PERIOD_CHECK', "
            "'PER_CLAIM_LIMIT_CHECK', 'PRE_AUTH_REQUIRED').  Must be non-empty "
            "after whitespace trimming."
        ),
        examples=[
            "WAITING_PERIOD_CHECK",
            "MEMBER_ELIGIBILITY_CHECK",
            "PER_CLAIM_LIMIT_CHECK",
            "EXCLUSION_CHECK",
            "PRE_AUTH_REQUIRED",
            "NETWORK_HOSPITAL_CHECK",
        ],
    )

    status: ValidationStatus = Field(
        ...,
        description=(
            "Outcome of evaluating this rule against the claim.  See "
            "``ValidationStatus`` for the meaning of each value."
        ),
    )

    passed: bool = Field(
        ...,
        description=(
            "Convenience boolean summarising whether this rule check should be "
            "treated as passing.  ``True`` for ``PASSED`` and ``SKIPPED`` "
            "statuses; ``False`` for ``FAILED``; ``True`` for ``WARNING`` "
            "(the claim is not blocked, but the signal is recorded).  "
            "Downstream agents may use this flag for fast aggregation without "
            "needing to switch on ``ValidationStatus``."
        ),
    )

    reason: str | None = Field(
        default=None,
        description=(
            "Human-readable explanation of why this rule passed, failed, or "
            "was skipped.  Must be specific enough to appear verbatim in a "
            "rejection notice or audit log.  ``None`` is acceptable only for "
            "rules with a ``PASSED`` status where no further explanation is "
            "needed."
        ),
        examples=[
            "Member joined on 2024-09-01; diabetes waiting period of 90 days "
            "expires on 2024-11-30.  Treatment date 2024-10-15 falls within "
            "the waiting period.",
            "Claimed amount ₹7,500 exceeds the per-claim limit of ₹5,000.",
            "MRI scan cost ₹15,000 exceeds the ₹10,000 pre-auth threshold; "
            "no pre-authorisation reference was found in the submission.",
        ],
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Arbitrary key-value data attached by the Policy Agent to support "
            "the validation outcome.  Use this to carry computed values that "
            "informed the decision (e.g. ``{'waiting_period_end_date': "
            "'2024-11-30', 'treatment_date': '2024-10-15'}``).  Empty dict "
            "when no supplementary data is relevant."
        ),
        examples=[
            {
                "join_date": "2024-09-01",
                "waiting_period_days": 90,
                "waiting_period_end_date": "2024-11-30",
                "treatment_date": "2024-10-15",
            }
        ],
    )

    @field_validator("rule_name", mode="before")
    @classmethod
    def _validate_rule_name(cls, value: Any) -> str:
        """Strip whitespace and reject empty rule names."""
        return _require_nonempty_str("rule_name", value)


# ---------------------------------------------------------------------------
# FraudSignal
# ---------------------------------------------------------------------------


class FraudSignal(BaseModel):
    """
    A single suspicious observation raised by the Fraud Detection Agent.

    Each ``FraudSignal`` documents one specific anomaly detected in the claim
    submission.  The collection of signals on a ``FraudResult`` instance
    provides an auditable record of what triggered any manual-review routing
    or fraud-score elevation.

    Common signal types include:

    - ``MULTIPLE_SAME_DAY_CLAIMS`` – the member submitted more claims in a
      single day than the policy's ``same_day_claims_limit`` permits.
    - ``HIGH_VALUE_CLAIM`` – the claimed amount exceeds the
      ``high_value_claim_threshold`` configured in the policy.
    - ``DOCUMENT_ALTERATION`` – the extraction agent detected corrections,
      crossed-out amounts, or duplicate ''ORIGINAL'' stamps.
    - ``DUPLICATE_BILL`` – a bill with an identical bill number or line-item
      fingerprint was seen on a prior claim from this member.
    """

    model_config = _SHARED_CONFIG

    signal_type: str = Field(
        ...,
        description=(
            "Short machine-readable label identifying the category of fraud "
            "signal (e.g. 'MULTIPLE_SAME_DAY_CLAIMS', 'DOCUMENT_ALTERATION').  "
            "Must be non-empty after whitespace trimming.  Downstream systems "
            "may use this value for grouping and alerting."
        ),
        examples=[
            "MULTIPLE_SAME_DAY_CLAIMS",
            "HIGH_VALUE_CLAIM",
            "DOCUMENT_ALTERATION",
            "DUPLICATE_BILL",
        ],
    )

    description: str = Field(
        ...,
        description=(
            "Human-readable explanation of the specific observation that "
            "triggered this signal.  Must be precise enough to appear in an "
            "operations review queue without additional context.  Must be "
            "non-empty after whitespace trimming."
        ),
        examples=[
            "Member EMP008 has submitted 4 claims on 2024-10-30, exceeding "
            "the same-day limit of 2.",
            "Claimed amount ₹27,000 exceeds the high-value threshold of ₹25,000.",
        ],
    )

    severity: float = Field(
        ...,
        description=(
            "Relative severity of this signal in the range [0.0, 1.0].  A "
            "score of 0.0 represents a negligible anomaly; 1.0 represents a "
            "near-certain fraudulent indicator.  The Fraud Agent uses these "
            "per-signal scores to compute the overall ``FraudResult.fraud_score``."
        ),
        examples=[0.85, 0.40, 1.0],
    )

    @field_validator("signal_type", mode="before")
    @classmethod
    def _validate_signal_type(cls, value: Any) -> str:
        """Strip whitespace and reject empty signal types."""
        return _require_nonempty_str("signal_type", value)

    @field_validator("description", mode="before")
    @classmethod
    def _validate_description(cls, value: Any) -> str:
        """Strip whitespace and reject empty descriptions."""
        return _require_nonempty_str("description", value)

    @field_validator("severity", mode="before")
    @classmethod
    def _validate_severity(cls, value: Any) -> float:
        """Reject severity values outside [0.0, 1.0]."""
        return _validate_unit_interval("severity", value)


# ---------------------------------------------------------------------------
# FraudResult
# ---------------------------------------------------------------------------


class FraudResult(BaseModel):
    """
    Complete output of the Fraud Detection Agent for a single claim.

    The ``fraud_score`` is an aggregate risk indicator derived from the
    individual ``FraudSignal`` instances in ``signals``.  The Fraud Agent is
    responsible for computing this aggregate; the model only enforces that it
    is a valid unit-interval float.

    When ``requires_manual_review`` is ``True``, the Decision Agent must
    route the claim to ``MANUAL_REVIEW`` regardless of the policy validation
    outcome.  The threshold at which this flag is set is defined in the
    policy's ``fraud_thresholds.fraud_score_manual_review_threshold``
    configuration; the model does not enforce that threshold — it only stores
    the agent's conclusion.
    """

    model_config = _SHARED_CONFIG

    fraud_score: float = Field(
        ...,
        description=(
            "Aggregate fraud risk score in the range [0.0, 1.0].  Derived by "
            "the Fraud Agent from the combination of signals detected.  A "
            "score of 0.0 indicates no suspicious activity; 1.0 indicates the "
            "highest possible risk.  The policy configuration defines the "
            "threshold above which ``requires_manual_review`` is set."
        ),
        examples=[0.0, 0.35, 0.92],
    )

    signals: list[FraudSignal] = Field(
        default_factory=list,
        description=(
            "Ordered list of suspicious observations detected by the Fraud "
            "Agent.  An empty list indicates no anomalies were found.  Each "
            "entry is independently auditable."
        ),
    )

    requires_manual_review: bool = Field(
        ...,
        description=(
            "Whether the Fraud Agent determined that this claim must be "
            "reviewed by a human before any automated decision is made.  "
            "When ``True``, the Decision Agent must produce "
            "``DecisionType.MANUAL_REVIEW`` regardless of the policy "
            "validation outcome."
        ),
    )

    reason: str | None = Field(
        default=None,
        description=(
            "Human-readable summary of the overall fraud assessment.  "
            "Populated when ``fraud_score`` is above zero or when "
            "``requires_manual_review`` is ``True``.  ``None`` when no "
            "anomalies were detected and no explanation is needed."
        ),
        examples=[
            "Member submitted 4 claims on the same day (limit: 2). "
            "Claim routed for manual review.",
            None,
        ],
    )

    @field_validator("fraud_score", mode="before")
    @classmethod
    def _validate_fraud_score(cls, value: Any) -> float:
        """Reject fraud scores outside [0.0, 1.0]."""
        return _validate_unit_interval("fraud_score", value)


# ---------------------------------------------------------------------------
# DecisionBreakdown
# ---------------------------------------------------------------------------


class DecisionBreakdown(BaseModel):
    """
    Financial breakdown explaining how the approved amount was derived.

    This model provides the line-by-line arithmetic that supports the final
    ``ClaimDecision``.  It is intentionally separate from the decision itself
    so that the API layer can render it progressively (e.g. show the
    calculation steps in the UI before displaying the final outcome).

    All monetary fields are ``Decimal`` to preserve exact arithmetic when
    network discounts and co-pay percentages are applied sequentially.

    Example calculation (TC010 — network hospital consultation):
    ::

        claimed_amount:  ₹4,500
        network_discount (20%):  −₹900   (deducted first)
        post_discount:   ₹3,600
        co_pay (10%):    −₹360   (applied on post-discount amount)
        approved_amount: ₹3,240
        deductions:      ₹1,260  (900 + 360)
    """

    model_config = _SHARED_CONFIG

    claimed_amount: Decimal = Field(
        ...,
        description=(
            "Original amount submitted by the member in the claim, in INR.  "
            "Must be >= 0."
        ),
        examples=[Decimal("4500.00"), Decimal("1500.00")],
    )

    approved_amount: Decimal = Field(
        ...,
        description=(
            "Final amount approved for reimbursement after all deductions "
            "have been applied, in INR.  Must be >= 0.  For ``REJECTED`` "
            "decisions this will be ``0``."
        ),
        examples=[Decimal("3240.00"), Decimal("0.00")],
    )

    deductions: Decimal = Field(
        ...,
        description=(
            "Total amount deducted from the claimed amount (sum of network "
            "discount, co-pay, sub-limit cap, and any excluded line-item "
            "amounts), in INR.  Must be >= 0.  "
            "``claimed_amount − deductions`` should equal ``approved_amount``; "
            "enforcement of this invariant is the Decision Agent's "
            "responsibility, not this model's."
        ),
        examples=[Decimal("1260.00"), Decimal("150.00")],
    )

    calculation_summary: str = Field(
        ...,
        description=(
            "Human-readable narrative of the calculation steps applied to "
            "derive the approved amount.  Must be specific enough to appear "
            "verbatim in a member-facing explanation or operations audit log.  "
            "Must be non-empty after whitespace trimming."
        ),
        examples=[
            "Network discount (20%) applied on ₹4,500 = ₹3,600. "
            "Co-pay (10%) applied on ₹3,600 = ₹360 deducted. "
            "Final approved: ₹3,240.",
            "10% co-pay applied on consultation category. "
            "₹150 deducted from ₹1,500. Final approved: ₹1,350.",
        ],
    )

    @field_validator("claimed_amount", mode="before")
    @classmethod
    def _validate_claimed_amount(cls, value: Any) -> Decimal:
        """Coerce to Decimal and reject negative amounts."""
        return _validate_nonnegative_decimal("claimed_amount", value)

    @field_validator("approved_amount", mode="before")
    @classmethod
    def _validate_approved_amount(cls, value: Any) -> Decimal:
        """Coerce to Decimal and reject negative amounts."""
        return _validate_nonnegative_decimal("approved_amount", value)

    @field_validator("deductions", mode="before")
    @classmethod
    def _validate_deductions(cls, value: Any) -> Decimal:
        """Coerce to Decimal and reject negative deductions."""
        return _validate_nonnegative_decimal("deductions", value)

    @field_validator("calculation_summary", mode="before")
    @classmethod
    def _validate_calculation_summary(cls, value: Any) -> str:
        """Strip whitespace and reject empty summaries."""
        return _require_nonempty_str("calculation_summary", value)


# ---------------------------------------------------------------------------
# ClaimDecision
# ---------------------------------------------------------------------------


class ClaimDecision(BaseModel):
    """
    Final output produced by the Decision Agent for a single claim.

    This is the most important model in the system.  It is the authoritative
    record of what decision was made, why it was made, what amount was
    approved, how confident the system is, and what every upstream validation
    step concluded.

    Every field required by the assignment specification is present:

    - ``decision`` — one of ``APPROVED``, ``PARTIAL``, ``REJECTED``,
      ``MANUAL_REVIEW``.
    - ``approved_amount`` — the INR amount approved for reimbursement.
    - ``confidence_score`` — the Decision Agent's confidence in the outcome.
    - ``reason`` — a non-empty explanation suitable for member communication.
    - ``validation_results`` — the full list of policy check outcomes
      (explainability trace).
    - ``fraud_result`` — the complete fraud assessment, if performed.
    - ``breakdown`` — the financial calculation steps, if the claim was
      approved or partially approved.

    Consumers
    ---------
    - The API layer serialises this model directly into the HTTP response.
    - The frontend uses ``decision``, ``approved_amount``, ``reason``, and
      ``breakdown`` for the member-facing result screen.
    - The operations dashboard uses ``validation_results``, ``fraud_result``,
      and ``confidence_score`` for the reviewer queue.
    - The audit log stores the entire model as a JSON snapshot.
    """

    model_config = _SHARED_CONFIG

    decision: DecisionType = Field(
        ...,
        description=(
            "Final decision outcome for the claim.  One of ``APPROVED``, "
            "``PARTIAL``, ``REJECTED``, or ``MANUAL_REVIEW``."
        ),
    )

    approved_amount: Decimal = Field(
        ...,
        description=(
            "Amount approved for reimbursement in INR.  Must be >= 0.  "
            "For ``REJECTED`` decisions this must be ``Decimal('0')``.  "
            "For ``MANUAL_REVIEW`` decisions this represents the best "
            "estimate pending human review and may be ``Decimal('0')`` when "
            "the system cannot produce a reliable estimate."
        ),
        examples=[Decimal("3240.00"), Decimal("1350.00"), Decimal("0.00")],
    )

    confidence_score: float = Field(
        ...,
        description=(
            "Decision Agent's confidence in the outcome, in the range "
            "[0.0, 1.0].  Reflects both the completeness of the extracted "
            "information and the clarity of the applicable policy rules.  "
            "Scores degraded by component failures or ambiguous documents "
            "are surfaced here and visible in the operations dashboard."
        ),
        examples=[0.95, 0.72, 0.40],
    )

    reason: str = Field(
        ...,
        description=(
            "Primary human-readable explanation for the decision.  This is "
            "the top-level message surfaced to the member and the operations "
            "team.  Must be non-empty, specific, and actionable — 'Claim "
            "rejected due to waiting period for diabetes (eligible from "
            "2024-11-30)' rather than 'Claim rejected'.  Must be non-empty "
            "after whitespace trimming."
        ),
        examples=[
            "Claim approved. 10% co-pay of ₹150 applied. Approved: ₹1,350.",
            "Rejected: treatment date falls within the 90-day diabetes waiting "
            "period. Eligible from 2024-11-30.",
            "MRI scan requires pre-authorisation above ₹10,000. No pre-auth "
            "reference found. Please resubmit with a valid pre-auth number.",
        ],
    )

    recommendation: str | None = Field(
        default=None,
        description=(
            "Short machine-readable routing recommendation produced by the "
            "Decision Agent for the operations layer.  Common values: "
            "``'AUTO_APPROVED'``, ``'AUTO_REJECTED'``, "
            "``'MANUAL_REVIEW_REQUIRED'``.  ``None`` when no specific routing "
            "action is needed beyond the ``decision`` itself."
        ),
        examples=["AUTO_APPROVED", "AUTO_REJECTED", "MANUAL_REVIEW_REQUIRED", None],
    )

    breakdown: DecisionBreakdown | None = Field(
        default=None,
        description=(
            "Financial calculation breakdown showing how the approved amount "
            "was derived.  Populated for ``APPROVED`` and ``PARTIAL`` "
            "decisions.  ``None`` for ``REJECTED`` decisions where no "
            "approved amount exists, and for ``MANUAL_REVIEW`` decisions "
            "where the final calculation is deferred to the human reviewer."
        ),
    )

    validation_results: list[ValidationResult] = Field(
        default_factory=list,
        description=(
            "Ordered list of all policy rule checks performed by the Policy "
            "Agent.  This list is the complete explainability trace: any "
            "reviewer or audit process can reconstruct exactly which rules "
            "ran, what they produced, and why the final decision was reached.  "
            "An empty list indicates no validation was performed (e.g. the "
            "claim was stopped at the document-verification stage)."
        ),
    )

    fraud_result: FraudResult | None = Field(
        default=None,
        description=(
            "Complete output of the Fraud Detection Agent.  ``None`` when "
            "fraud detection was not performed or was skipped due to a "
            "component failure.  When present and "
            "``fraud_result.requires_manual_review`` is ``True``, the "
            "``decision`` must be ``MANUAL_REVIEW``."
        ),
    )

    generated_at: datetime = Field(
        ...,
        description=(
            "UTC timestamp at which the Decision Agent produced this outcome.  "
            "Used for audit trail ordering and SLA tracking."
        ),
    )

    @field_validator("approved_amount", mode="before")
    @classmethod
    def _validate_approved_amount(cls, value: Any) -> Decimal:
        """Coerce to Decimal and reject negative amounts."""
        return _validate_nonnegative_decimal("approved_amount", value)

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _validate_confidence_score(cls, value: Any) -> float:
        """Reject confidence scores outside [0.0, 1.0]."""
        return _validate_unit_interval("confidence_score", value)

    @field_validator("reason", mode="before")
    @classmethod
    def _validate_reason(cls, value: Any) -> str:
        """Strip whitespace and reject empty reason strings."""
        return _require_nonempty_str("reason", value)


# ---------------------------------------------------------------------------
# ClaimOutcomeSummary
# ---------------------------------------------------------------------------


class ClaimOutcomeSummary(BaseModel):
    """
    Lightweight projection of a ``ClaimDecision`` for dashboard consumption.

    This model carries only the fields needed to render a claim row in the
    operations dashboard or a member status screen.  It deliberately omits
    the full validation trace, fraud result, and breakdown to reduce
    serialisation overhead for list endpoints that return many claims at once.

    Consumers that need the complete decision record should query the full
    ``ClaimDecision`` by claim ID.
    """

    model_config = _SHARED_CONFIG

    decision: DecisionType = Field(
        ...,
        description="Final decision outcome for the claim.",
    )

    approved_amount: Decimal = Field(
        ...,
        description=(
            "Amount approved for reimbursement in INR.  Must be >= 0.  "
            "Mirrors ``ClaimDecision.approved_amount``."
        ),
        examples=[Decimal("3240.00"), Decimal("0.00")],
    )

    confidence_score: float = Field(
        ...,
        description=(
            "Decision Agent's confidence in the outcome, in [0.0, 1.0].  "
            "Mirrors ``ClaimDecision.confidence_score``."
        ),
        examples=[0.95, 0.72],
    )

    @field_validator("approved_amount", mode="before")
    @classmethod
    def _validate_approved_amount(cls, value: Any) -> Decimal:
        """Coerce to Decimal and reject negative amounts."""
        return _validate_nonnegative_decimal("approved_amount", value)

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _validate_confidence_score(cls, value: Any) -> float:
        """Reject confidence scores outside [0.0, 1.0]."""
        return _validate_unit_interval("confidence_score", value)