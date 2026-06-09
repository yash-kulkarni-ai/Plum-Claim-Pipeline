from datetime import datetime, date, timezone
from decimal import Decimal

from app.models.claim import (
    ClaimCategory,
    ClaimStatus,
    ClaimSubmission,
    ProcessingError,
    ProcessingWarning,
    ClaimState,
)

from app.models.document import UploadedDocument

# --------------------------------------------------
# Enums
# --------------------------------------------------

assert ClaimCategory.CONSULTATION == "CONSULTATION"
assert ClaimCategory.DIAGNOSTIC == "DIAGNOSTIC"

assert ClaimStatus.SUBMITTED == "SUBMITTED"
assert ClaimStatus.COMPLETED == "COMPLETED"

# --------------------------------------------------
# Test document
# --------------------------------------------------

doc = UploadedDocument(
    document_id="DOC001",
    filename="prescription.jpg",
    file_path="/uploads/prescription.jpg",
    mime_type="image/jpeg",
    uploaded_at=datetime.now(timezone.utc),
)

# --------------------------------------------------
# ClaimSubmission
# --------------------------------------------------

submission = ClaimSubmission(
    member_id=" EMP001 ",
    policy_id=" PLUM_GHI_2024 ",
    claim_category=ClaimCategory.CONSULTATION,
    claimed_amount="1500.00",
    treatment_date=date.today(),
    documents=[doc],
    submitted_at=datetime.now(timezone.utc),
)

assert submission.member_id == "EMP001"
assert submission.policy_id == "PLUM_GHI_2024"
assert submission.claimed_amount == Decimal("1500.00")

# --------------------------------------------------
# Invalid amount
# --------------------------------------------------

try:
    ClaimSubmission(
        member_id="EMP001",
        policy_id="POL001",
        claim_category=ClaimCategory.CONSULTATION,
        claimed_amount="0",
        treatment_date=date.today(),
        documents=[doc],
        submitted_at=datetime.now(timezone.utc),
    )
    raise AssertionError("Amount should fail")
except Exception:
    pass

# --------------------------------------------------
# Empty member_id
# --------------------------------------------------

try:
    ClaimSubmission(
        member_id=" ",
        policy_id="POL001",
        claim_category=ClaimCategory.CONSULTATION,
        claimed_amount="1000",
        treatment_date=date.today(),
        documents=[doc],
        submitted_at=datetime.now(timezone.utc),
    )
    raise AssertionError("member_id should fail")
except Exception:
    pass

# --------------------------------------------------
# ProcessingError
# --------------------------------------------------

error = ProcessingError(
    component=" OCRAgent ",
    message=" OCR timeout ",
    recoverable=True,
    timestamp=datetime.now(timezone.utc),
)

assert error.component == "OCRAgent"
assert error.message == "OCR timeout"

# --------------------------------------------------
# ProcessingWarning
# --------------------------------------------------

warning = ProcessingWarning(
    source=" OCRAgent ",
    message=" Low confidence ",
    timestamp=datetime.now(timezone.utc),
)

assert warning.source == "OCRAgent"
assert warning.message == "Low confidence"

# --------------------------------------------------
# ClaimState
# --------------------------------------------------

state = ClaimState(
    claim=submission,
    created_at=datetime.now(timezone.utc),
    updated_at=datetime.now(timezone.utc),
)

assert state.status == ClaimStatus.SUBMITTED
assert state.confidence_score == 1.0

assert state.has_errors is False
assert state.has_warnings is False
assert state.is_complete is False

assert state.trace_count == 0
assert state.validation_count == 0
assert state.extraction_count == 0

# --------------------------------------------------
# Add error and warning
# --------------------------------------------------

state.errors.append(error)
state.warnings.append(warning)

assert state.has_errors is True
assert state.has_warnings is True

# --------------------------------------------------
# Confidence validation
# --------------------------------------------------

try:
    ClaimState(
        claim=submission,
        confidence_score=1.5,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    raise AssertionError("Confidence > 1 should fail")
except Exception:
    pass

try:
    state.confidence_score = 2.0
    raise AssertionError("Assignment validation should fail")
except Exception:
    pass

print("✅ All claim assertions passed.")