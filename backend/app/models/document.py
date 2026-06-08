"""
Document models for the Plum Health Insurance Claims Processing System.

This module defines the foundational data contracts for medical documents
submitted as part of health insurance claims. It is consumed by document
verification agents, extraction agents, and the claims decision pipeline.

Design principles:
- All fields carry explicit descriptions suitable for API documentation.
- Validators are fail-fast and produce actionable error messages.
- Serialisation is deterministic: enums emit their string values, datetimes
  emit ISO-8601 strings, and unknown extra fields are rejected at construction
  time.
- Computed properties are pure and side-effect-free.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Allowed MIME types
# ---------------------------------------------------------------------------

_ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {
        "application/pdf",
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }
)

_IMAGE_MIME_TYPES: frozenset[str] = frozenset(
    {
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    }
)


# ---------------------------------------------------------------------------
# DocumentType
# ---------------------------------------------------------------------------


class DocumentType(StrEnum):
    """
    Canonical classification labels for medical documents submitted with a
    health insurance claim.

    Using ``StrEnum`` ensures that every value serialises to its plain string
    representation in JSON without requiring a custom encoder, and that
    comparisons against raw strings work transparently.

    Values
    ------
    PRESCRIPTION
        A doctor's written or printed medication order, carrying diagnosis,
        prescribed medicines, dosages, and the treating physician's details.
    HOSPITAL_BILL
        An itemised invoice issued by a hospital, clinic, or day-care centre
        for services rendered during the episode of care.
    PHARMACY_BILL
        A receipt from a licensed pharmacy listing dispensed medicines, batch
        numbers, MRP, quantities, and net amounts charged.
    LAB_REPORT
        A diagnostic laboratory result sheet containing test names, observed
        values, reference ranges, and pathologist sign-off.
    DENTAL_REPORT
        A treatment summary or procedure note issued by a dental practitioner,
        required specifically for dental category claims.
    DIAGNOSTIC_REPORT
        A clinical imaging or special investigation report (e.g. MRI, CT, PET,
        ultrasound) that is distinct from routine lab work.
    DISCHARGE_SUMMARY
        A document issued at the end of an inpatient admission summarising the
        diagnosis, procedures performed, and follow-up instructions.
    UNKNOWN
        The document could not be classified with sufficient confidence, or its
        type falls outside the supported taxonomy.  Documents in this state
        must not be treated as satisfying any required-document slot.
    """

    PRESCRIPTION = "PRESCRIPTION"
    HOSPITAL_BILL = "HOSPITAL_BILL"
    PHARMACY_BILL = "PHARMACY_BILL"
    LAB_REPORT = "LAB_REPORT"
    DENTAL_REPORT = "DENTAL_REPORT"
    DIAGNOSTIC_REPORT = "DIAGNOSTIC_REPORT"
    DISCHARGE_SUMMARY = "DISCHARGE_SUMMARY"
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# UploadedDocument
# ---------------------------------------------------------------------------


class UploadedDocument(BaseModel):
    """
    Represents a single medical document uploaded as part of a health
    insurance claim submission.

    This model is the shared contract between the ingestion layer, document
    verification agents, extraction agents, and the claims decision engine.
    It carries both the raw upload metadata (file path, MIME type) and the
    enriched classification state produced by downstream agents
    (``document_type``, ``confidence``, ``is_readable``).

    Lifecycle
    ---------
    1. **Ingestion** – The API layer constructs the model with ``document_type``
       and ``confidence`` left as ``None``.  ``is_readable`` defaults to
       ``True`` (optimistic; unverified).
    2. **Verification** – The document-verification agent updates
       ``document_type``, ``confidence``, ``is_readable``, and
       ``classification_reason`` in-place.
    3. **Extraction** – The extraction agent populates ``metadata`` with
       structured fields parsed from the document content.
    4. **Policy check** – The policy engine reads ``document_type`` and
       ``metadata`` to evaluate claim eligibility.

    Configuration
    -------------
    - ``extra="forbid"`` prevents silently swallowed fields from upstream
      services.
    - ``validate_assignment=True`` keeps invariants intact when agent code
      mutates fields after construction.
    - ``populate_by_name=True`` allows both alias and field name in
      ``model_validate`` calls.
    """

    model_config = {
        "extra": "forbid",
        "validate_assignment": True,
        "frozen": False,
        "populate_by_name": True,
    }

    # ------------------------------------------------------------------
    # Identity & storage
    # ------------------------------------------------------------------

    document_id: str = Field(
        ...,
        description=(
            "Globally unique identifier for this document, assigned at upload "
            "time.  Must be a non-empty string after whitespace trimming."
        ),
        examples=["F001", "doc_7f3a1b2c-4d5e-6f78-90ab-cdef01234567"],
    )

    filename: str = Field(
        ...,
        description=(
            "Original filename provided by the uploading client, including the "
            "file extension (e.g. 'prescription.jpg').  Stored for display and "
            "audit purposes only; do not rely on it for type inference."
        ),
        examples=["dr_sharma_prescription.jpg", "hospital_bill_nov2024.pdf"],
    )

    file_path: str = Field(
        ...,
        description=(
            "Absolute or relative path to the stored file on the backing "
            "object store or local file system.  Must be non-empty.  This "
            "value is used by extraction agents to read file content."
        ),
        examples=[
            "/uploads/claims/EMP001/F001/dr_sharma_prescription.jpg",
            "s3://plum-claims/2024/11/F001.pdf",
        ],
    )

    mime_type: str = Field(
        ...,
        description=(
            "MIME content-type of the uploaded file.  Only the following "
            "values are accepted: "
            "'application/pdf', 'image/jpeg', 'image/jpg', 'image/png', "
            "'image/webp'.  Any other value is rejected at construction time."
        ),
        examples=["application/pdf", "image/jpeg", "image/png"],
    )

    # ------------------------------------------------------------------
    # Classification state  (populated by verification agent)
    # ------------------------------------------------------------------

    document_type: DocumentType | None = Field(
        default=None,
        description=(
            "Classified document type assigned by the document-verification "
            "agent.  ``None`` indicates the document has not yet been "
            "classified.  A value of ``DocumentType.UNKNOWN`` indicates "
            "classification was attempted but could not produce a confident "
            "result.  Either state prevents the document from satisfying a "
            "required-document slot."
        ),
    )

    confidence: float | None = Field(
        default=None,
        description=(
            "Classification confidence score in the range [0.0, 1.0] produced "
            "by the verification agent.  ``None`` means classification has not "
            "been run.  Scores below 0.5 should be treated as unreliable by "
            "downstream consumers."
        ),
        examples=[0.97, 0.42, None],
    )

    is_readable: bool = Field(
        default=True,
        description=(
            "Whether the document content is sufficiently legible for "
            "automated extraction.  Set to ``False`` by the verification agent "
            "when the document is too blurry, too dark, partially cropped, or "
            "otherwise unsuitable for reliable OCR.  A claim containing an "
            "unreadable required document must prompt the member to re-upload "
            "before processing continues."
        ),
    )

    classification_reason: str | None = Field(
        default=None,
        description=(
            "Human-readable explanation of why the document was assigned its "
            "current ``document_type`` and ``confidence``.  Populated by the "
            "verification agent.  Used to generate specific, actionable error "
            "messages for members and for audit trails."
        ),
        examples=[
            "Document header matches standard prescription template; "
            "doctor registration number KA/45678/2015 detected.",
            "Image is too blurry to extract any text reliably (estimated "
            "sharpness score: 0.08).",
        ],
    )

    # ------------------------------------------------------------------
    # Timestamps
    # ------------------------------------------------------------------

    uploaded_at: datetime = Field(
        ...,
        description=(
            "UTC timestamp at which the document was received by the ingestion "
            "layer.  Serialised as an ISO-8601 string in ``model_dump`` output."
        ),
    )

    # ------------------------------------------------------------------
    # Extracted content
    # ------------------------------------------------------------------

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Structured key-value data extracted from the document by the "
            "extraction agent.  The schema is document-type-specific and is "
            "intentionally untyped here to decouple the model from individual "
            "extractor implementations.  Consumers should treat this as an "
            "opaque bag until a typed extraction model is available."
        ),
        examples=[
            {
                "doctor_name": "Dr. Arun Sharma",
                "doctor_registration": "KA/45678/2015",
                "patient_name": "Rajesh Kumar",
                "diagnosis": "Viral Fever",
            }
        ],
    )

    # ------------------------------------------------------------------
    # Field validators
    # ------------------------------------------------------------------

    @field_validator("document_id", mode="before")
    @classmethod
    def _validate_document_id(cls, value: Any) -> str:
        """Strip whitespace and reject empty document identifiers."""
        if not isinstance(value, str):
            raise ValueError(
                f"document_id must be a string, got {type(value).__name__!r}."
            )
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "document_id must not be empty or contain only whitespace."
            )
        return stripped

    @field_validator("filename", mode="before")
    @classmethod
    def _validate_filename(cls, value: Any) -> str:
        """Strip whitespace and reject empty filenames."""
        if not isinstance(value, str):
            raise ValueError(
                f"filename must be a string, got {type(value).__name__!r}."
            )
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "filename must not be empty or contain only whitespace."
            )
        return stripped

    @field_validator("file_path", mode="before")
    @classmethod
    def _validate_file_path(cls, value: Any) -> str:
        """Reject empty file paths."""
        if not isinstance(value, str):
            raise ValueError(
                f"file_path must be a string, got {type(value).__name__!r}."
            )
        if not value:
            raise ValueError("file_path must not be empty.")
        return value

    @field_validator("mime_type", mode="before")
    @classmethod
    def _validate_mime_type(cls, value: Any) -> str:
        """
        Reject MIME types that the system cannot process.

        Only the five types listed in ``_ALLOWED_MIME_TYPES`` are accepted.
        Normalisation (e.g. lowercasing) is intentionally not performed: the
        caller is responsible for providing a canonical MIME type string.
        """
        if not isinstance(value, str):
            raise ValueError(
                f"mime_type must be a string, got {type(value).__name__!r}."
            )
        if value not in _ALLOWED_MIME_TYPES:
            allowed = ", ".join(sorted(_ALLOWED_MIME_TYPES))
            raise ValueError(
                f"Unsupported MIME type {value!r}.  "
                f"Accepted values are: {allowed}."
            )
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_confidence(cls, value: Any) -> float | None:
        """
        Accept ``None`` (unclassified) or a float in [0.0, 1.0].

        Raises ``ValueError`` if a non-None value falls outside the unit
        interval, which would indicate a bug in the classification agent.
        """
        if value is None:
            return None
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"confidence must be a float or None, got {value!r}."
            ) from exc
        if not (0.0 <= score <= 1.0):
            raise ValueError(
                f"confidence must be in the range [0.0, 1.0], got {score}."
            )
        return score

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def is_image(self) -> bool:
        """
        Return ``True`` when the document was uploaded as a raster image.

        Specifically, returns ``True`` for MIME types:
        ``image/jpeg``, ``image/jpg``, ``image/png``, ``image/webp``.

        Extraction agents may use this to select an appropriate parsing
        strategy (vision model vs. PDF text extraction).
        """
        return self.mime_type in _IMAGE_MIME_TYPES

    @property
    def is_pdf(self) -> bool:
        """
        Return ``True`` when the document was uploaded as a PDF.

        Specifically, returns ``True`` for ``application/pdf`` only.
        """
        return self.mime_type == "application/pdf"

    @property
    def is_classified(self) -> bool:
        """
        Return ``True`` when the document has been assigned a concrete,
        known type by the verification agent.

        A document is considered *classified* only when ``document_type`` is
        not ``None`` **and** is not ``DocumentType.UNKNOWN``.  Both of the
        following states are treated as *unclassified*:

        - ``document_type is None`` – verification agent has not run yet.
        - ``document_type == DocumentType.UNKNOWN`` – agent ran but could not
          produce a confident classification.

        Downstream consumers (e.g. the document-slot verification step) should
        gate on this property before treating the document as satisfying a
        required-document requirement.
        """
        return (
            self.document_type is not None
            and self.document_type is not DocumentType.UNKNOWN
        )