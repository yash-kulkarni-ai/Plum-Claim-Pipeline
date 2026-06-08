"""
Extraction schema contracts for the Plum Health Insurance Claims Processing System.

This module defines the strict Pydantic models that normalise and validate
structured data produced by Gemini Vision OCR before that data is consumed by
verification, policy, and decision agents.

Architecture position
---------------------
::

    Gemini Vision
        ↓
    Extraction Models  ← this module
        ↓
    Verification Agent
        ↓
    Policy Agent
        ↓
    Decision Agent

Design principles
-----------------
- Every model uses ``extra="forbid"`` so that unexpected fields emitted by the
  LLM are caught immediately rather than propagated silently.
- ``validate_assignment=True`` ensures invariants hold when agents mutate
  fields post-construction.
- String fields are whitespace-stripped at the boundary so that downstream
  agents receive clean values regardless of OCR formatting quirks.
- ``Decimal`` is used for all monetary amounts to prevent floating-point
  rounding errors in financial calculations.
- ``None`` is the canonical sentinel for "field not found in document";
  agents must not infer absent data.
- No business logic, policy rules, or claim decision logic lives in this file.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
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
# Shared string-stripping helper
# ---------------------------------------------------------------------------


def _strip_or_none(value: Any) -> str | None:
    """
    Normalise a nullable string field from OCR output.

    - If the value is ``None``, return ``None``.
    - If the value is a non-empty string after stripping, return the stripped
      string.
    - If the value is an all-whitespace string, return ``None`` so that
      downstream agents can treat it as absent.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Expected str or None, got {type(value).__name__!r}.")
    stripped = value.strip()
    return stripped if stripped else None


# ---------------------------------------------------------------------------
# PatientInfo
# ---------------------------------------------------------------------------


class PatientInfo(BaseModel):
    """
    Patient or insured-member details extracted from a medical document.

    This sub-model is embedded in every document-level extraction model so
    that verification agents can cross-reference the patient across multiple
    uploaded documents within a single claim.

    Fields may be ``None`` when the corresponding information was not legible
    or not present in the source document; agents must not infer or fabricate
    absent values.
    """

    model_config = _SHARED_CONFIG

    name: str | None = Field(
        default=None,
        description=(
            "Full name of the patient as it appears on the document.  "
            "Whitespace-stripped.  ``None`` if not present or illegible."
        ),
        examples=["Rajesh Kumar", "Priya Singh"],
    )

    age: int | None = Field(
        default=None,
        description=(
            "Age of the patient in years at the time of the visit, as stated "
            "on the document.  Must be >= 0 when present.  ``None`` if not "
            "stated."
        ),
        examples=[39, 25, None],
    )

    gender: str | None = Field(
        default=None,
        description=(
            "Gender of the patient as recorded on the document (e.g. 'M', "
            "'F', 'Male', 'Female').  No normalisation is applied here; "
            "downstream agents are responsible for standardisation.  ``None`` "
            "if not stated."
        ),
        examples=["M", "Female", None],
    )

    member_id: str | None = Field(
        default=None,
        description=(
            "Insurer or employer member ID found on the document, if printed.  "
            "This value is not always present; most Indian OPD documents do "
            "not carry it.  ``None`` when absent."
        ),
        examples=["EMP001", None],
    )

    @field_validator("name", "gender", "member_id", mode="before")
    @classmethod
    def _strip_string_fields(cls, value: Any) -> str | None:
        """Strip whitespace; return ``None`` for blank or absent strings."""
        return _strip_or_none(value)

    @field_validator("age", mode="before")
    @classmethod
    def _validate_age(cls, value: Any) -> int | None:
        """Reject negative ages; accept ``None`` for absent values."""
        if value is None:
            return None
        try:
            age = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"age must be an integer or None, got {value!r}."
            ) from exc
        if age < 0:
            raise ValueError(f"age must be >= 0, got {age}.")
        return age


# ---------------------------------------------------------------------------
# DoctorInfo
# ---------------------------------------------------------------------------


class DoctorInfo(BaseModel):
    """
    Medical practitioner details extracted from a prescription or report.

    Registration number formats are state-specific in India (e.g.
    ``KA/45678/2015`` for Karnataka).  Validation of format correctness is
    intentionally deferred to a dedicated verification agent and is **not**
    performed here.
    """

    model_config = _SHARED_CONFIG

    name: str | None = Field(
        default=None,
        description=(
            "Full name of the treating or referring doctor as printed on the "
            "document, including title (e.g. 'Dr. Arun Sharma').  ``None`` "
            "if not legible."
        ),
        examples=["Dr. Arun Sharma", "Vaidya T. Krishnan", None],
    )

    registration_number: str | None = Field(
        default=None,
        description=(
            "Medical Council registration number of the practitioner as "
            "stated on the document.  Common Indian formats follow the pattern "
            "``<STATE>/<NUMBER>/<YEAR>`` (e.g. ``KA/45678/2015``).  Format "
            "validation is performed by downstream agents, not here.  ``None`` "
            "if absent or obscured."
        ),
        examples=["KA/45678/2015", "AYUR/KL/2345/2019", None],
    )

    specialization: str | None = Field(
        default=None,
        description=(
            "Medical specialization or qualification as printed on the "
            "letterhead or stamp (e.g. 'MBBS, MD (Internal Medicine)').  "
            "``None`` if not stated."
        ),
        examples=["MBBS, MD (Internal Medicine)", "BDS", None],
    )

    @field_validator("name", "registration_number", "specialization", mode="before")
    @classmethod
    def _strip_string_fields(cls, value: Any) -> str | None:
        """Strip whitespace; return ``None`` for blank or absent strings."""
        return _strip_or_none(value)


# ---------------------------------------------------------------------------
# HospitalInfo
# ---------------------------------------------------------------------------


class HospitalInfo(BaseModel):
    """
    Hospital or clinic details extracted from a bill or report header.

    Small clinics in India frequently omit the GSTIN on invoices; ``None``
    is the expected value for that field in most OPD scenarios.
    """

    model_config = _SHARED_CONFIG

    name: str | None = Field(
        default=None,
        description=(
            "Name of the hospital, clinic, or diagnostic centre as printed on "
            "the document.  ``None`` if not legible or absent."
        ),
        examples=["Apollo Hospitals", "City Medical Centre, Bengaluru", None],
    )

    address: str | None = Field(
        default=None,
        description=(
            "Full address of the facility as printed on the document.  May "
            "span multiple lines; newlines are preserved.  ``None`` if absent."
        ),
        examples=["12 MG Road, Bengaluru – 560001", None],
    )

    gstin: str | None = Field(
        default=None,
        description=(
            "Goods and Services Tax Identification Number of the facility, if "
            "printed on the invoice (format: ``29XXXXX1234X1ZX``).  Many "
            "small clinics and pharmacies are exempt or do not print it.  "
            "``None`` when absent."
        ),
        examples=["29XXXXX1234X1ZX", None],
    )

    @field_validator("name", "address", "gstin", mode="before")
    @classmethod
    def _strip_string_fields(cls, value: Any) -> str | None:
        """Strip whitespace; return ``None`` for blank or absent strings."""
        return _strip_or_none(value)


# ---------------------------------------------------------------------------
# BillLineItem
# ---------------------------------------------------------------------------


class BillLineItem(BaseModel):
    """
    A single line item from a hospital bill, pharmacy bill, or clinic invoice.

    Examples of descriptions seen in Indian medical bills:

    - ``"Consultation Fee (OPD)"``
    - ``"MRI Lumbar Spine"``
    - ``"CBC (Complete Blood Count)"``
    - ``"Paracetamol 650mg × 15"``
    - ``"Teeth Whitening"``

    ``Decimal`` is used for ``amount`` instead of ``float`` to avoid
    floating-point rounding errors when the policy engine aggregates line
    items against sub-limits and co-pay thresholds.
    """

    model_config = _SHARED_CONFIG

    description: str = Field(
        ...,
        description=(
            "Human-readable description of the service, procedure, or product "
            "as printed on the bill.  Must be a non-empty string after "
            "whitespace trimming."
        ),
        examples=["Consultation Fee (OPD)", "MRI Lumbar Spine", "Paracetamol 650mg"],
    )

    amount: Decimal = Field(
        ...,
        description=(
            "Amount charged for this line item in Indian Rupees (INR).  "
            "Must be >= 0.  Stored as ``Decimal`` to preserve exact monetary "
            "precision."
        ),
        examples=[Decimal("1000.00"), Decimal("300.00"), Decimal("0.00")],
    )

    @field_validator("description", mode="before")
    @classmethod
    def _validate_description(cls, value: Any) -> str:
        """Strip whitespace and reject empty or absent descriptions."""
        if not isinstance(value, str):
            raise ValueError(
                f"description must be a non-empty string, got {type(value).__name__!r}."
            )
        stripped = value.strip()
        if not stripped:
            raise ValueError("description must not be empty or contain only whitespace.")
        return stripped

    @field_validator("amount", mode="before")
    @classmethod
    def _validate_amount(cls, value: Any) -> Decimal:
        """
        Coerce numeric types to ``Decimal`` and reject negative amounts.

        Accepts ``int``, ``float``, ``str``, and ``Decimal`` inputs to
        accommodate the varied representations that Gemini may emit.
        """
        if isinstance(value, Decimal):
            decimal_value = value
        else:
            try:
                decimal_value = Decimal(str(value))
            except Exception as exc:
                raise ValueError(
                    f"amount must be a non-negative number, got {value!r}."
                ) from exc
        if decimal_value < Decimal("0"):
            raise ValueError(f"amount must be >= 0, got {decimal_value}.")
        return decimal_value


# ---------------------------------------------------------------------------
# LabTestResult
# ---------------------------------------------------------------------------


class LabTestResult(BaseModel):
    """
    A single diagnostic test result extracted from a laboratory report.

    Each row in the results table of a lab report maps to one instance of
    this model.  Only ``test_name`` is required; the remaining fields are
    ``None`` when the corresponding column is absent or illegible.
    """

    model_config = _SHARED_CONFIG

    test_name: str = Field(
        ...,
        description=(
            "Name of the diagnostic test as printed on the report (e.g. "
            "'Hemoglobin', 'Dengue NS1 Antigen', 'MRI Lumbar Spine').  Must "
            "be a non-empty string after whitespace trimming."
        ),
        examples=["Hemoglobin", "WBC Count", "Dengue NS1 Antigen"],
    )

    result: str | None = Field(
        default=None,
        description=(
            "Observed result value as printed, including units if embedded in "
            "the result cell (e.g. '13.2', 'NEGATIVE', 'Positive').  Raw "
            "string; no parsing or unit extraction is performed here.  "
            "``None`` if the cell was blank or illegible."
        ),
        examples=["13.2", "NEGATIVE", "Positive", None],
    )

    unit: str | None = Field(
        default=None,
        description=(
            "Unit of measurement for the result (e.g. 'g/dL', '/μL', "
            "'mg/dL').  ``None`` for qualitative tests or when absent."
        ),
        examples=["g/dL", "/μL", None],
    )

    normal_range: str | None = Field(
        default=None,
        description=(
            "Reference range as printed in the report (e.g. '13.0 – 17.0', "
            "'150,000 – 450,000').  Raw string; no parsing is performed.  "
            "``None`` for tests without a tabulated reference range."
        ),
        examples=["13.0 – 17.0", "4,500 – 11,000", None],
    )

    @field_validator("test_name", mode="before")
    @classmethod
    def _validate_test_name(cls, value: Any) -> str:
        """Strip whitespace and reject empty test names."""
        if not isinstance(value, str):
            raise ValueError(
                f"test_name must be a non-empty string, got {type(value).__name__!r}."
            )
        stripped = value.strip()
        if not stripped:
            raise ValueError("test_name must not be empty or contain only whitespace.")
        return stripped

    @field_validator("result", "unit", "normal_range", mode="before")
    @classmethod
    def _strip_optional_strings(cls, value: Any) -> str | None:
        """Strip whitespace; return ``None`` for blank or absent strings."""
        return _strip_or_none(value)


# ---------------------------------------------------------------------------
# PrescriptionExtraction
# ---------------------------------------------------------------------------


class PrescriptionExtraction(BaseModel):
    """
    Structured output produced by Gemini Vision OCR on a medical prescription.

    A prescription is the primary document for most OPD claim categories.
    It links the patient to a treating doctor, records the diagnosis, and
    authorises medicines and investigations.

    Medicines and test names are stored as raw strings rather than typed
    objects because prescription formats vary widely across Indian clinics;
    structured parsing is delegated to downstream agents that have
    domain-specific normalisation logic.
    """

    model_config = _SHARED_CONFIG

    patient: PatientInfo = Field(
        ...,
        description=(
            "Patient details as extracted from the prescription header or "
            "patient info block."
        ),
    )

    doctor: DoctorInfo = Field(
        ...,
        description=(
            "Prescribing doctor details extracted from the letterhead, stamp, "
            "or signature block."
        ),
    )

    diagnosis: str | None = Field(
        default=None,
        description=(
            "Primary diagnosis or chief complaint as written by the doctor.  "
            "May include medical shorthand (e.g. 'T2DM', 'HTN', 'URI').  "
            "Whitespace-stripped.  ``None`` if not stated or illegible."
        ),
        examples=["Viral Fever", "Type 2 Diabetes Mellitus", "Acute Bronchitis", None],
    )

    medicines: list[str] = Field(
        default_factory=list,
        description=(
            "List of medicine names and dosage instructions as written on the "
            "prescription (e.g. 'Tab Paracetamol 650mg — 1-1-1 x 5 days').  "
            "Each entry is a raw string; no parsing of dose or frequency is "
            "performed here.  Empty list when no medicines are prescribed."
        ),
        examples=[["Tab Paracetamol 650mg — 1-1-1 x 5 days", "Vitamin C 500mg — 0-0-1 x 7 days"]],
    )

    tests_ordered: list[str] = Field(
        default_factory=list,
        description=(
            "List of investigations or diagnostic tests ordered by the doctor "
            "(e.g. 'CBC', 'Dengue NS1', 'MRI Lumbar Spine').  Empty list "
            "when no tests are ordered."
        ),
        examples=[["CBC", "Dengue NS1 Antigen"]],
    )

    visit_date: date | None = Field(
        default=None,
        description=(
            "Date of the consultation as printed on the prescription in the "
            "format ``YYYY-MM-DD`` after parsing.  ``None`` if absent or "
            "illegible."
        ),
        examples=["2024-11-01", None],
    )

    @field_validator("diagnosis", mode="before")
    @classmethod
    def _strip_diagnosis(cls, value: Any) -> str | None:
        """Strip whitespace; return ``None`` for blank or absent values."""
        return _strip_or_none(value)

    @field_validator("medicines", "tests_ordered", mode="before")
    @classmethod
    def _strip_string_list(cls, value: Any) -> list[str]:
        """
        Strip whitespace from each entry and remove blank strings produced
        by OCR artefacts.
        """
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"Expected a list of strings, got {type(value).__name__!r}."
            )
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(
                    f"Each entry must be a string, got {type(item).__name__!r}."
                )
            stripped = item.strip()
            if stripped:
                cleaned.append(stripped)
        return cleaned


# ---------------------------------------------------------------------------
# HospitalBillExtraction
# ---------------------------------------------------------------------------


class HospitalBillExtraction(BaseModel):
    """
    Structured output produced by Gemini Vision OCR on a hospital or clinic bill.

    Hospital bills are required for consultation, diagnostic, dental, vision,
    and alternative-medicine claim categories.  The line items allow the
    policy engine to identify and exclude non-covered services (e.g. cosmetic
    procedures) at the item level before computing the approved amount.
    """

    model_config = _SHARED_CONFIG

    patient: PatientInfo = Field(
        ...,
        description="Patient details extracted from the bill header.",
    )

    hospital: HospitalInfo = Field(
        ...,
        description="Hospital or clinic details extracted from the bill header.",
    )

    bill_date: date | None = Field(
        default=None,
        description=(
            "Date of the bill as printed, parsed to ``YYYY-MM-DD``.  ``None`` "
            "if absent or illegible."
        ),
        examples=["2024-11-01", None],
    )

    line_items: list[BillLineItem] = Field(
        default_factory=list,
        description=(
            "Itemised list of services, procedures, or products charged on "
            "the bill.  Empty list when no line items could be extracted "
            "(e.g. the bill shows only a lump-sum total without breakdown)."
        ),
    )

    total_amount: Decimal | None = Field(
        default=None,
        description=(
            "Total amount payable as printed on the bill in INR.  Must be "
            ">= 0 when present.  ``None`` if not legible."
        ),
        examples=[Decimal("1500.00"), None],
    )

    @field_validator("total_amount", mode="before")
    @classmethod
    def _validate_total_amount(cls, value: Any) -> Decimal | None:
        """Coerce to ``Decimal`` and reject negative totals."""
        if value is None:
            return None
        if isinstance(value, Decimal):
            decimal_value = value
        else:
            try:
                decimal_value = Decimal(str(value))
            except Exception as exc:
                raise ValueError(
                    f"total_amount must be a non-negative number or None, got {value!r}."
                ) from exc
        if decimal_value < Decimal("0"):
            raise ValueError(f"total_amount must be >= 0, got {decimal_value}.")
        return decimal_value


# ---------------------------------------------------------------------------
# PharmacyBillExtraction
# ---------------------------------------------------------------------------


class PharmacyBillExtraction(BaseModel):
    """
    Structured output produced by Gemini Vision OCR on a pharmacy bill.

    Pharmacy bills are the primary supporting document for the ``PHARMACY``
    claim category.  Line items typically correspond to individual medicines
    with their batch numbers, expiry dates, quantities, MRP, and amounts;
    however, many small pharmacies issue a single-line total rather than an
    itemised breakdown.
    """

    model_config = _SHARED_CONFIG

    patient: PatientInfo = Field(
        ...,
        description="Patient details extracted from the pharmacy bill.",
    )

    pharmacy_name: str | None = Field(
        default=None,
        description=(
            "Name of the dispensing pharmacy as printed on the bill.  "
            "Whitespace-stripped.  ``None`` if absent or illegible."
        ),
        examples=["Health First Pharmacy", None],
    )

    bill_date: date | None = Field(
        default=None,
        description=(
            "Date of the pharmacy bill, parsed to ``YYYY-MM-DD``.  ``None`` "
            "if absent or illegible."
        ),
        examples=["2024-11-01", None],
    )

    line_items: list[BillLineItem] = Field(
        default_factory=list,
        description=(
            "Itemised list of medicines dispensed.  Each entry corresponds to "
            "one row on the pharmacy bill.  Empty list when only a total is "
            "available."
        ),
    )

    total_amount: Decimal | None = Field(
        default=None,
        description=(
            "Net amount charged (after any pharmacy discount) as printed on "
            "the bill, in INR.  Must be >= 0 when present.  ``None`` if not "
            "legible."
        ),
        examples=[Decimal("73.62"), None],
    )

    @field_validator("pharmacy_name", mode="before")
    @classmethod
    def _strip_pharmacy_name(cls, value: Any) -> str | None:
        """Strip whitespace; return ``None`` for blank or absent values."""
        return _strip_or_none(value)

    @field_validator("total_amount", mode="before")
    @classmethod
    def _validate_total_amount(cls, value: Any) -> Decimal | None:
        """Coerce to ``Decimal`` and reject negative totals."""
        if value is None:
            return None
        if isinstance(value, Decimal):
            decimal_value = value
        else:
            try:
                decimal_value = Decimal(str(value))
            except Exception as exc:
                raise ValueError(
                    f"total_amount must be a non-negative number or None, got {value!r}."
                ) from exc
        if decimal_value < Decimal("0"):
            raise ValueError(f"total_amount must be >= 0, got {decimal_value}.")
        return decimal_value


# ---------------------------------------------------------------------------
# LabReportExtraction
# ---------------------------------------------------------------------------


class LabReportExtraction(BaseModel):
    """
    Structured output produced by Gemini Vision OCR on a diagnostic lab report.

    Lab reports are required for ``DIAGNOSTIC`` claims and are optional
    supporting documents for consultation claims.  The ``test_results`` list
    maps directly to the rows in the results table of the report.
    """

    model_config = _SHARED_CONFIG

    patient: PatientInfo = Field(
        ...,
        description="Patient details extracted from the lab report header.",
    )

    doctor: DoctorInfo | None = Field(
        default=None,
        description=(
            "Referring or ordering doctor details if printed on the report.  "
            "``None`` when absent (walk-in lab visits may not carry a referral "
            "doctor)."
        ),
    )

    test_results: list[LabTestResult] = Field(
        default_factory=list,
        description=(
            "List of individual test results extracted from the report.  "
            "Each entry corresponds to one test row.  Empty list if no "
            "structured results table could be extracted."
        ),
    )

    remarks: str | None = Field(
        default=None,
        description=(
            "Free-text remarks, interpretation, or clinical correlation notes "
            "appended by the pathologist.  Whitespace-stripped.  ``None`` if "
            "absent."
        ),
        examples=[
            "WBC count is towards upper normal limit.  Clinical correlation advised.",
            None,
        ],
    )

    @field_validator("remarks", mode="before")
    @classmethod
    def _strip_remarks(cls, value: Any) -> str | None:
        """Strip whitespace; return ``None`` for blank or absent values."""
        return _strip_or_none(value)


# ---------------------------------------------------------------------------
# DentalReportExtraction
# ---------------------------------------------------------------------------


class DentalReportExtraction(BaseModel):
    """
    Structured output produced by Gemini Vision OCR on a dental treatment report.

    Dental reports are optional supporting documents for ``DENTAL`` claims.
    The ``procedures`` list is used by the policy engine to distinguish covered
    procedures (e.g. Root Canal Treatment) from cosmetic exclusions (e.g. Teeth
    Whitening) at the item level.
    """

    model_config = _SHARED_CONFIG

    patient: PatientInfo = Field(
        ...,
        description="Patient details extracted from the dental report.",
    )

    doctor: DoctorInfo | None = Field(
        default=None,
        description=(
            "Treating dentist's details if present on the report.  ``None`` "
            "when absent."
        ),
    )

    diagnosis: str | None = Field(
        default=None,
        description=(
            "Dental diagnosis or clinical finding as stated by the dentist "
            "(e.g. 'Periapical Abscess', 'Dental Caries – Upper Left 6').  "
            "Whitespace-stripped.  ``None`` if not stated."
        ),
        examples=["Periapical Abscess", "Dental Caries – Upper Left 6", None],
    )

    procedures: list[str] = Field(
        default_factory=list,
        description=(
            "List of dental procedures performed as recorded in the report "
            "(e.g. 'Root Canal Treatment', 'Teeth Whitening').  These values "
            "are matched against the policy's covered and excluded procedure "
            "lists by the policy engine.  Empty list if no procedures are "
            "stated."
        ),
        examples=[["Root Canal Treatment", "Teeth Whitening"]],
    )

    @field_validator("diagnosis", mode="before")
    @classmethod
    def _strip_diagnosis(cls, value: Any) -> str | None:
        """Strip whitespace; return ``None`` for blank or absent values."""
        return _strip_or_none(value)

    @field_validator("procedures", mode="before")
    @classmethod
    def _strip_procedures(cls, value: Any) -> list[str]:
        """Strip whitespace from each entry; discard blank strings."""
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError(
                f"procedures must be a list of strings, got {type(value).__name__!r}."
            )
        cleaned: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(
                    f"Each procedure must be a string, got {type(item).__name__!r}."
                )
            stripped = item.strip()
            if stripped:
                cleaned.append(stripped)
        return cleaned


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------


class ExtractionResult(BaseModel):
    """
    Normalised OCR output envelope returned by every extraction agent.

    Every agent in the extraction layer — regardless of document type — wraps
    its output in this model before passing it to the verification agent.
    This provides a single, uniform contract that the rest of the pipeline
    can depend on without needing to branch on document type.

    The ``structured_data`` field carries the serialised payload of the
    appropriate typed extraction model (e.g. ``PrescriptionExtraction``,
    ``HospitalBillExtraction``) as a plain ``dict``.  Consumers that need the
    typed model should call ``model_validate`` on the appropriate class using
    this dict.

    The ``warnings`` list records non-fatal issues encountered during
    extraction (e.g. low-confidence fields, partially illegible sections,
    detected document alterations) so that downstream agents and human
    reviewers have full visibility into extraction quality without needing
    to reprocess the raw document.
    """

    model_config = _SHARED_CONFIG

    document_id: str = Field(
        ...,
        description=(
            "Identifier of the source document, matching the ``document_id`` "
            "field of the corresponding ``UploadedDocument``.  Must be a "
            "non-empty string after whitespace trimming."
        ),
        examples=["F001", "F007"],
    )

    document_type: str = Field(
        ...,
        description=(
            "String representation of the document type classification "
            "(e.g. ``'PRESCRIPTION'``, ``'HOSPITAL_BILL'``).  Should match a "
            "``DocumentType`` enum value but is stored as a plain string here "
            "to avoid a circular import dependency between the extraction and "
            "document modules."
        ),
        examples=["PRESCRIPTION", "HOSPITAL_BILL", "LAB_REPORT"],
    )

    structured_data: dict[str, Any] = Field(
        ...,
        description=(
            "Serialised payload of the typed extraction model (e.g. the "
            "``model_dump()`` output of a ``PrescriptionExtraction`` instance).  "
            "The exact schema is determined by ``document_type``.  Consumers "
            "must validate this dict against the appropriate typed model before "
            "use."
        ),
    )

    confidence: float = Field(
        ...,
        description=(
            "Overall extraction confidence score in the range [0.0, 1.0] "
            "assigned by the extraction agent.  Reflects the proportion of "
            "required fields successfully extracted and the OCR quality of "
            "the source document.  Scores below 0.5 should trigger manual "
            "review by downstream agents."
        ),
        examples=[0.95, 0.62, 0.30],
    )

    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Non-fatal issues encountered during extraction.  Each entry is a "
            "human-readable message suitable for display to operations staff "
            "and for inclusion in claim audit trails.  Empty list indicates a "
            "clean extraction with no anomalies detected."
        ),
        examples=[
            [
                "Registration number partially obscured by rubber stamp; "
                "extracted value may be incomplete.",
                "Bill total (₹1,500) does not match sum of line items (₹1,450).",
            ]
        ],
    )

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

    @field_validator("document_type", mode="before")
    @classmethod
    def _validate_document_type(cls, value: Any) -> str:
        """Strip whitespace and reject empty document type strings."""
        if not isinstance(value, str):
            raise ValueError(
                f"document_type must be a string, got {type(value).__name__!r}."
            )
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                "document_type must not be empty or contain only whitespace."
            )
        return stripped

    @field_validator("confidence", mode="before")
    @classmethod
    def _validate_confidence(cls, value: Any) -> float:
        """Reject confidence values outside [0.0, 1.0]."""
        try:
            score = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"confidence must be a float in [0.0, 1.0], got {value!r}."
            ) from exc
        if not (0.0 <= score <= 1.0):
            raise ValueError(
                f"confidence must be in the range [0.0, 1.0], got {score}."
            )
        return score