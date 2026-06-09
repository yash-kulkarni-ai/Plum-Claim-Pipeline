from decimal import Decimal
from datetime import date

from app.models.extraction import (
    PatientInfo,
    DoctorInfo,
    HospitalInfo,
    BillLineItem,
    LabTestResult,
    PrescriptionExtraction,
    HospitalBillExtraction,
    ExtractionResult,
)

# --------------------------------------------------
# PatientInfo
# --------------------------------------------------

patient = PatientInfo(
    name="  Yash Kulkarni  ",
    age=21,
    gender=" Male ",
    member_id=" EMP001 ",
)

assert patient.name == "Yash Kulkarni"
assert patient.gender == "Male"
assert patient.member_id == "EMP001"

try:
    PatientInfo(age=-1)
    raise AssertionError("Negative age should fail")
except Exception:
    pass

# --------------------------------------------------
# DoctorInfo
# --------------------------------------------------

doctor = DoctorInfo(
    name=" Dr. Sharma ",
    registration_number=" KA/12345/2020 ",
)

assert doctor.name == "Dr. Sharma"
assert doctor.registration_number == "KA/12345/2020"

# --------------------------------------------------
# HospitalInfo
# --------------------------------------------------

hospital = HospitalInfo(
    name=" Apollo Hospitals ",
    address=" MG Road ",
)

assert hospital.name == "Apollo Hospitals"
assert hospital.address == "MG Road"

# --------------------------------------------------
# BillLineItem
# --------------------------------------------------

item = BillLineItem(
    description=" Consultation Fee ",
    amount="500.00",
)

assert item.description == "Consultation Fee"
assert item.amount == Decimal("500.00")

try:
    BillLineItem(
        description="Consultation",
        amount="-1",
    )
    raise AssertionError("Negative amount should fail")
except Exception:
    pass

# --------------------------------------------------
# LabTestResult
# --------------------------------------------------

test = LabTestResult(
    test_name=" Hemoglobin ",
    result="13.5",
    unit="g/dL",
)

assert test.test_name == "Hemoglobin"

try:
    LabTestResult(test_name=" ")
    raise AssertionError("Blank test name should fail")
except Exception:
    pass

# --------------------------------------------------
# PrescriptionExtraction
# --------------------------------------------------

prescription = PrescriptionExtraction(
    patient=patient,
    doctor=doctor,
    diagnosis=" Viral Fever ",
    medicines=[
        " Paracetamol ",
        " ",
        " Vitamin C ",
    ],
    tests_ordered=[
        " CBC ",
        "",
    ],
)

assert prescription.diagnosis == "Viral Fever"
assert prescription.medicines == ["Paracetamol", "Vitamin C"]
assert prescription.tests_ordered == ["CBC"]

# --------------------------------------------------
# HospitalBillExtraction
# --------------------------------------------------

bill = HospitalBillExtraction(
    patient=patient,
    hospital=hospital,
    bill_date=date.today(),
    line_items=[item],
    total_amount="500.00",
)

assert bill.total_amount == Decimal("500.00")

try:
    HospitalBillExtraction(
        patient=patient,
        hospital=hospital,
        total_amount="-10",
    )
    raise AssertionError("Negative total should fail")
except Exception:
    pass

# --------------------------------------------------
# ExtractionResult
# --------------------------------------------------

result = ExtractionResult(
    document_id=" DOC001 ",
    document_type=" PRESCRIPTION ",
    structured_data=prescription.model_dump(),
    confidence=0.95,
)

assert result.document_id == "DOC001"
assert result.document_type == "PRESCRIPTION"
assert result.confidence == 0.95

try:
    ExtractionResult(
        document_id="DOC002",
        document_type="PRESCRIPTION",
        structured_data={},
        confidence=1.5,
    )
    raise AssertionError("Confidence > 1 should fail")
except Exception:
    pass

try:
    ExtractionResult(
        document_id=" ",
        document_type="PRESCRIPTION",
        structured_data={},
        confidence=0.5,
    )
    raise AssertionError("Blank document_id should fail")
except Exception:
    pass

# --------------------------------------------------
# Assignment validation
# --------------------------------------------------

try:
    result.confidence = 2.0
    raise AssertionError("Assignment validation should fail")
except Exception:
    pass

print("✅ All extraction assertions passed.")