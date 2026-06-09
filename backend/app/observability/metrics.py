"""
Pure metrics calculation functions for the Plum Health Insurance Claims
Processing System observability layer.

This module contains two deterministic, side-effect-free functions that
derive observability metrics from the accumulated ``TraceEvent`` list in
``OverallState``.  They are consumed by the Explainability Agent at the end
of the pipeline to produce the ``TraceSummary`` and duration information that
appear in the operations dashboard and the claim audit report.

Design principles
-----------------
- Both functions are module-level functions rather than class methods because
  they have no shared state and no instance behaviour.
- Neither function mutates the state it receives or creates any side effects.
- Timestamps are read exclusively from ``TraceEvent.timestamp`` values; the
  system clock is never consulted.  This makes both functions deterministic
  and safe to call in any context, including replay of historical events.
- ``get_pipeline_duration_ms`` returns ``None`` rather than zero when no
  events exist, enabling callers to distinguish "pipeline produced no events"
  from "pipeline completed in zero time" (the latter only happens when there
  is exactly one event).
"""

from __future__ import annotations

from app.graph.state import OverallState
from app.models.trace import TraceSeverity, TraceStatus, TraceSummary

__all__ = ["calculate_execution_metrics", "get_pipeline_duration_ms"]


def calculate_execution_metrics(state: OverallState) -> TraceSummary:
    """
    Derive a ``TraceSummary`` from the accumulated trace events in state.

    Iterates over ``state["trace_events"]`` once and counts events by their
    ``status`` and ``severity`` according to the following rules:

    - ``passed_events``  — events whose ``status`` is ``TraceStatus.PASSED``.
    - ``failed_events``  — events whose ``status`` is ``TraceStatus.FAILED``.
    - ``warning_events`` — events whose ``severity`` is
      ``TraceSeverity.WARNING`` (regardless of status).
    - ``critical_events`` — events whose ``severity`` is
      ``TraceSeverity.CRITICAL`` (regardless of status).
    - ``total_events``   — ``len(state["trace_events"])``.

    Note that the four sub-counts are not guaranteed to sum to
    ``total_events``: a single event may be counted in both
    ``failed_events`` (by status) and ``critical_events`` (by severity), and
    events with ``STARTED``, ``SKIPPED``, or ``PARTIAL`` statuses that carry
    ``INFO`` or ``ERROR`` severity are counted only in ``total_events``.

    Parameters
    ----------
    state:
        The current ``OverallState``.  Only ``state["trace_events"]`` is
        read; no other fields are accessed.

    Returns
    -------
    TraceSummary
        A fully-populated ``TraceSummary`` reflecting the event counts at
        the moment this function was called.  Returns a zero-count summary
        when ``state["trace_events"]`` is empty.

    Examples
    --------
    Given a state with 12 events of which 8 have ``status=PASSED``, 1 has
    ``status=FAILED``, 2 have ``severity=WARNING``, and 1 has
    ``severity=CRITICAL``::

        summary = calculate_execution_metrics(state)
        assert summary.total_events == 12
        assert summary.passed_events == 8
        assert summary.failed_events == 1
        assert summary.warning_events == 2
        assert summary.critical_events == 1
    """
    events = state["trace_events"]

    total_events: int = len(events)
    passed_events: int = 0
    failed_events: int = 0
    warning_events: int = 0
    critical_events: int = 0

    for event in events:
        if event.status is TraceStatus.PASSED:
            passed_events += 1
        elif event.status is TraceStatus.FAILED:
            failed_events += 1

        if event.severity is TraceSeverity.WARNING:
            warning_events += 1
        elif event.severity is TraceSeverity.CRITICAL:
            critical_events += 1

    return TraceSummary(
        total_events=total_events,
        passed_events=passed_events,
        failed_events=failed_events,
        warning_events=warning_events,
        critical_events=critical_events,
    )


def get_pipeline_duration_ms(state: OverallState) -> int | None:
    """
    Return the total elapsed pipeline time in milliseconds.

    Computes the duration as the difference between the ``timestamp`` of the
    earliest event and the ``timestamp`` of the latest event found in
    ``state["trace_events"]``.

    Special cases
    -------------
    - **No events**: returns ``None`` — the pipeline produced no telemetry,
      so no duration can be calculated.
    - **Exactly one event**: returns ``0`` — there is a start point but no
      end point, so elapsed time is treated as zero rather than ``None`` to
      signal that the pipeline at least began.
    - **Two or more events**: returns the integer millisecond difference
      between the earliest and latest timestamps.  The result is always
      non-negative because ``min`` and ``max`` are used on the full event
      list rather than assuming ordering.

    Parameters
    ----------
    state:
        The current ``OverallState``.  Only ``state["trace_events"]`` is
        read; the system clock is never consulted.

    Returns
    -------
    int | None
        Elapsed pipeline time in whole milliseconds, or ``None`` when no
        events are present.

    Notes
    -----
    The function uses ``min`` / ``max`` over all event timestamps rather than
    assuming the list is sorted.  This is intentional: LangGraph's parallel
    node execution or out-of-order state merges could, in theory, produce an
    unsorted event list.

    Examples
    --------
    ::

        # No events
        assert get_pipeline_duration_ms(state_with_no_events) is None

        # One event
        assert get_pipeline_duration_ms(state_with_one_event) == 0

        # Two events 842 ms apart
        assert get_pipeline_duration_ms(state) == 842
    """
    events = state["trace_events"]

    if not events:
        return None

    if len(events) == 1:
        return 0

    earliest = min(event.timestamp for event in events)
    latest = max(event.timestamp for event in events)

    delta = latest - earliest
    # Convert timedelta to whole milliseconds.
    # total_seconds() is used rather than .seconds to correctly handle
    # durations that span multiple hours (where .seconds wraps at 3600).
    return int(delta.total_seconds() * 1000)