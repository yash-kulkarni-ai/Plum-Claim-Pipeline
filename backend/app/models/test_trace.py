from datetime import datetime, timezone

from backend.app.observability.trace import (
    TraceStatus,
    TraceSeverity,
    TraceEvent,
    AgentExecutionSummary,
    TraceTimeline,
    TraceSummary,
)

# --------------------------------------------------
# Enums
# --------------------------------------------------

assert TraceStatus.PASSED == "PASSED"
assert TraceStatus.FAILED == "FAILED"

assert TraceSeverity.INFO == "INFO"
assert TraceSeverity.CRITICAL == "CRITICAL"

# --------------------------------------------------
# TraceEvent
# --------------------------------------------------

event = TraceEvent(
    agent_name=" OCRAgent ",
    action=" extract_text ",
    status=TraceStatus.PASSED,
    severity=TraceSeverity.INFO,
    message=" OCR completed ",
    timestamp=datetime.now(timezone.utc),
)

assert event.agent_name == "OCRAgent"
assert event.action == "extract_text"
assert event.message == "OCR completed"

assert event.is_failure is False
assert event.is_warning is False
assert event.is_critical is False

# --------------------------------------------------
# Failure event
# --------------------------------------------------

failure_event = TraceEvent(
    agent_name="OCRAgent",
    action="extract_text",
    status=TraceStatus.FAILED,
    severity=TraceSeverity.ERROR,
    message="OCR timeout",
    timestamp=datetime.now(timezone.utc),
)

assert failure_event.is_failure is True

# --------------------------------------------------
# Warning event
# --------------------------------------------------

warning_event = TraceEvent(
    agent_name="OCRAgent",
    action="extract_text",
    status=TraceStatus.PARTIAL,
    severity=TraceSeverity.WARNING,
    message="Low confidence",
    timestamp=datetime.now(timezone.utc),
)

assert warning_event.is_warning is True

# --------------------------------------------------
# Critical event
# --------------------------------------------------

critical_event = TraceEvent(
    agent_name="FraudAgent",
    action="fraud_check",
    status=TraceStatus.FAILED,
    severity=TraceSeverity.CRITICAL,
    message="Manual review required",
    timestamp=datetime.now(timezone.utc),
)

assert critical_event.is_critical is True

# --------------------------------------------------
# Invalid empty fields
# --------------------------------------------------

try:
    TraceEvent(
        agent_name=" ",
        action="x",
        status=TraceStatus.PASSED,
        severity=TraceSeverity.INFO,
        message="x",
        timestamp=datetime.now(timezone.utc),
    )
    raise AssertionError("Blank agent name should fail")
except Exception:
    pass

# --------------------------------------------------
# AgentExecutionSummary
# --------------------------------------------------

summary = AgentExecutionSummary(
    agent_name="OCRAgent",
    status=TraceStatus.PASSED,
    duration_ms=100,
    events_count=5,
)

assert summary.duration_ms == 100
assert summary.events_count == 5

try:
    AgentExecutionSummary(
        agent_name="OCRAgent",
        status=TraceStatus.PASSED,
        duration_ms=-1,
        events_count=1,
    )
    raise AssertionError("Negative duration should fail")
except Exception:
    pass

# --------------------------------------------------
# TraceTimeline
# --------------------------------------------------

timeline = TraceTimeline(
    events=[event, failure_event],
    generated_at=datetime.now(timezone.utc),
)

assert timeline.total_events == 2

# --------------------------------------------------
# TraceSummary
# --------------------------------------------------

trace_summary = TraceSummary(
    total_events=10,
    passed_events=8,
    failed_events=1,
    warning_events=1,
    critical_events=0,
)

assert trace_summary.total_events == 10

try:
    TraceSummary(
        total_events=-1,
        passed_events=0,
        failed_events=0,
        warning_events=0,
        critical_events=0,
    )
    raise AssertionError("Negative count should fail")
except Exception:
    pass

# --------------------------------------------------
# Assignment validation
# --------------------------------------------------

try:
    summary.duration_ms = -100
    raise AssertionError("Assignment validation should fail")
except Exception:
    pass

print("✅ All trace assertions passed.")