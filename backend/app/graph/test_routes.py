from datetime import datetime, timezone, date
from decimal import Decimal

from app.graph.routes import (
    route_after_verification,
    route_after_evaluation,
)

from app.models.claim import (
    ClaimSubmission,
)

from app.models.document import (
    UploadedDocument,
)

# --------------------------------------------------
# Readable document
# --------------------------------------------------

good_doc = UploadedDocument(
    document_id="DOC001",
    filename="prescription.jpg",
    file_path="/uploads/doc.jpg",
    mime_type="image/jpeg",
    uploaded_at=datetime.now(timezone.utc),
)

claim = ClaimSubmission(
    member_id="EMP001",
    policy_id="POL001",
    claim_category="CONSULTATION",
    claimed_amount=Decimal("1000"),
    treatment_date=date.today(),
    documents=[good_doc],
    submitted_at=datetime.now(timezone.utc),
)

state = {
    "claim": claim,
    "errors": [],
}

assert route_after_verification(state) == "ocr_extractor"

# --------------------------------------------------
# Unreadable document
# --------------------------------------------------

bad_doc = good_doc.model_copy(
    update={"is_readable": False}
)

claim_bad = claim.model_copy(
    update={"documents": [bad_doc]}
)

state_bad = {
    "claim": claim_bad,
    "errors": [],
}

assert route_after_verification(state_bad) == "decision_agent"

# --------------------------------------------------
# Errors present
# --------------------------------------------------

state_error = {
    "claim": claim,
    "errors": ["some error"],
}

assert route_after_verification(state_error) == "decision_agent"

# --------------------------------------------------
# Fraud routing always decision_agent
# --------------------------------------------------

assert route_after_evaluation(state) == "decision_agent"
assert route_after_evaluation(state_bad) == "decision_agent"
assert route_after_evaluation(state_error) == "decision_agent"

print("✅ All routes assertions passed.")