from datetime import datetime, timezone
import json

from backend.app.models.document import (
    DocumentType,
    UploadedDocument,
)

# --------------------------------------------------
# Enum tests
# --------------------------------------------------

assert DocumentType.PRESCRIPTION == "PRESCRIPTION"
assert DocumentType.UNKNOWN == "UNKNOWN"
assert json.dumps(DocumentType.PRESCRIPTION) == '"PRESCRIPTION"'

# --------------------------------------------------
# Valid document
# --------------------------------------------------

doc = UploadedDocument(
    document_id="F001",
    filename="prescription.jpg",
    file_path="/uploads/F001.jpg",
    mime_type="image/jpeg",
    uploaded_at=datetime.now(timezone.utc),
)

assert doc.document_id == "F001"
assert doc.document_type is None
assert doc.confidence is None
assert doc.is_readable is True
assert doc.is_image is True
assert doc.is_pdf is False
assert doc.is_classified is False
assert doc.metadata == {}

# --------------------------------------------------
# Whitespace trimming
# --------------------------------------------------

doc2 = UploadedDocument(
    document_id="  F002  ",
    filename="  bill.pdf  ",
    file_path="/uploads/F002.pdf",
    mime_type="application/pdf",
    uploaded_at=datetime.now(timezone.utc),
)

assert doc2.document_id == "F002"
assert doc2.filename == "bill.pdf"

# --------------------------------------------------
# Classified document
# --------------------------------------------------

doc3 = UploadedDocument(
    document_id="F003",
    filename="lab.pdf",
    file_path="/uploads/F003.pdf",
    mime_type="application/pdf",
    document_type=DocumentType.LAB_REPORT,
    confidence=0.92,
    uploaded_at=datetime.now(timezone.utc),
)

assert doc3.is_classified is True

# --------------------------------------------------
# UNKNOWN should not be classified
# --------------------------------------------------

doc4 = doc3.model_copy(update={"document_type": DocumentType.UNKNOWN})
assert doc4.is_classified is False

# --------------------------------------------------
# Confidence bounds
# --------------------------------------------------

try:
    UploadedDocument(
        document_id="F004",
        filename="x.jpg",
        file_path="/x.jpg",
        mime_type="image/png",
        confidence=1.5,
        uploaded_at=datetime.now(timezone.utc),
    )
    raise AssertionError("Expected validation error")
except Exception:
    pass

try:
    UploadedDocument(
        document_id="F005",
        filename="x.jpg",
        file_path="/x.jpg",
        mime_type="image/png",
        confidence=-0.1,
        uploaded_at=datetime.now(timezone.utc),
    )
    raise AssertionError("Expected validation error")
except Exception:
    pass

# --------------------------------------------------
# Edge confidence values
# --------------------------------------------------

for edge in (0.0, 1.0):
    d = UploadedDocument(
        document_id="F006",
        filename="x.jpg",
        file_path="/x.jpg",
        mime_type="image/jpeg",
        confidence=edge,
        uploaded_at=datetime.now(timezone.utc),
    )
    assert d.confidence == edge

# --------------------------------------------------
# Invalid MIME type
# --------------------------------------------------

try:
    UploadedDocument(
        document_id="F007",
        filename="x.gif",
        file_path="/x.gif",
        mime_type="image/gif",
        uploaded_at=datetime.now(timezone.utc),
    )
    raise AssertionError("Expected validation error")
except Exception:
    pass

# --------------------------------------------------
# Empty document_id
# --------------------------------------------------

try:
    UploadedDocument(
        document_id="   ",
        filename="x.jpg",
        file_path="/x.jpg",
        mime_type="image/jpeg",
        uploaded_at=datetime.now(timezone.utc),
    )
    raise AssertionError("Expected validation error")
except Exception:
    pass

# --------------------------------------------------
# Extra field rejection
# --------------------------------------------------

try:
    UploadedDocument(
        document_id="F008",
        filename="x.jpg",
        file_path="/x.jpg",
        mime_type="image/jpeg",
        uploaded_at=datetime.now(timezone.utc),
        unexpected_field="oops",
    )
    raise AssertionError("Expected validation error")
except Exception:
    pass

# --------------------------------------------------
# Assignment validation
# --------------------------------------------------

try:
    doc3.confidence = 2.5
    raise AssertionError("Expected assignment validation error")
except Exception:
    pass

print("✅ All assertions passed.")