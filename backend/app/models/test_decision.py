from datetime import datetime, timezone
from decimal import Decimal
import json

from backend.app.models.decision import (
    DecisionType,
    ValidationStatus,
    ValidationResult,
    FraudSignal,
    FraudResult,
    DecisionBreakdown,
    ClaimDecision,
    ClaimOutcomeSummary,
)

# --------------------------------------------------
# Enums
# --------------------------------------------------

assert DecisionType.APPROVED == "APPROVED"
assert DecisionType.MANUAL_REVIEW == "MANUAL_REVIEW"

assert ValidationStatus.PASSED == "PASSED"
assert ValidationStatus.FAILED == "FAILED"

assert json.dumps(DecisionType.APPROVED) == '"APPROVED"'

# --------------------------------------------------
# ValidationResult
# --------------------------------------------------

validation = ValidationResult(
    rule_name=" WAITING_PERIOD_CHECK ",
    status=ValidationStatus.PASSED,
    passed=True,
)

assert validation.rule_name == "WAITING_PERIOD_CHECK"

try:
    ValidationResult(
        rule_name=" ",
        status=ValidationStatus.PASSED,
        passed=True,
    )
    raise AssertionError("Blank rule name should fail")
except Exception:
    pass

# --------------------------------------------------
# FraudSignal
# --------------------------------------------------

signal = FraudSignal(
    signal_type=" HIGH_VALUE_CLAIM ",
    description=" Amount exceeds threshold ",
    severity=0.8,
)

assert signal.signal_type == "HIGH_VALUE_CLAIM"
assert signal.description == "Amount exceeds threshold"

try:
    FraudSignal(
        signal_type="X",
        description="Y",
        severity=1.5,
    )
    raise AssertionError("Severity > 1 should fail")
except Exception:
    pass

# --------------------------------------------------
# FraudResult
# --------------------------------------------------

fraud = FraudResult(
    fraud_score=0.75,
    signals=[signal],
    requires_manual_review=True,
)

assert fraud.fraud_score == 0.75

try:
    FraudResult(
        fraud_score=-0.1,
        requires_manual_review=False,
    )
    raise AssertionError("Negative fraud score should fail")
except Exception:
    pass

# --------------------------------------------------
# DecisionBreakdown
# --------------------------------------------------

breakdown = DecisionBreakdown(
    claimed_amount="4500.00",
    approved_amount="3240.00",
    deductions="1260.00",
    calculation_summary=" Network discount applied ",
)

assert breakdown.claimed_amount == Decimal("4500.00")
assert breakdown.approved_amount == Decimal("3240.00")
assert breakdown.deductions == Decimal("1260.00")
assert breakdown.calculation_summary == "Network discount applied"

try:
    DecisionBreakdown(
        claimed_amount="-1",
        approved_amount="0",
        deductions="0",
        calculation_summary="test",
    )
    raise AssertionError("Negative amount should fail")
except Exception:
    pass

# --------------------------------------------------
# ClaimDecision
# --------------------------------------------------

decision = ClaimDecision(
    decision=DecisionType.APPROVED,
    approved_amount="3240.00",
    confidence_score=0.95,
    reason=" Claim approved ",
    breakdown=breakdown,
    validation_results=[validation],
    fraud_result=fraud,
    generated_at=datetime.now(timezone.utc),
)

assert decision.approved_amount == Decimal("3240.00")
assert decision.reason == "Claim approved"
assert decision.confidence_score == 0.95

try:
    ClaimDecision(
        decision=DecisionType.APPROVED,
        approved_amount="100",
        confidence_score=1.5,
        reason="test",
        generated_at=datetime.now(timezone.utc),
    )
    raise AssertionError("Confidence > 1 should fail")
except Exception:
    pass

try:
    ClaimDecision(
        decision=DecisionType.APPROVED,
        approved_amount="-10",
        confidence_score=0.5,
        reason="test",
        generated_at=datetime.now(timezone.utc),
    )
    raise AssertionError("Negative approved amount should fail")
except Exception:
    pass

# --------------------------------------------------
# ClaimOutcomeSummary
# --------------------------------------------------

summary = ClaimOutcomeSummary(
    decision=DecisionType.APPROVED,
    approved_amount="3240.00",
    confidence_score=0.95,
)

assert summary.approved_amount == Decimal("3240.00")
assert summary.confidence_score == 0.95

try:
    ClaimOutcomeSummary(
        decision=DecisionType.APPROVED,
        approved_amount="-1",
        confidence_score=0.5,
    )
    raise AssertionError("Negative amount should fail")
except Exception:
    pass

# --------------------------------------------------
# Assignment validation
# --------------------------------------------------

try:
    decision.confidence_score = 2.0
    raise AssertionError("Assignment validation should fail")
except Exception:
    pass

print("✅ All decision assertions passed.")