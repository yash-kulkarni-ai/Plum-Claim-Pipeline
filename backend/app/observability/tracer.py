"""
Standardised ``TraceEvent`` factory for the Plum Health Insurance Claims
Processing System observability layer.

This module exposes ``AgentTracer``, a stateless class whose methods produce
fully-populated ``TraceEvent`` objects in a single call.  Agents use it to
emit telemetry without knowing the internals of ``TraceEvent`` construction —
UUID generation, timestamp injection, and status/severity mapping are all
handled here.

Usage pattern
-------------
Every agent node in the LangGraph graph should emit events like this::

    from app.observability.tracer import AgentTracer
    from app.observability.events import AgentActions, EventMessages

    # At the start of agent execution:
    start_evt = AgentTracer.start_event(
        agent_name="PolicyValidationAgent",
        action=AgentActions.POLICY_VALIDATION,
        message=EventMessages.agent_started("PolicyValidationAgent"),
        metadata={"claim_id": state["claim"].claim_id},
    )

    # On successful completion:
    done_evt = AgentTracer.success_event(
        agent_name="PolicyValidationAgent",
        action=AgentActions.POLICY_VALIDATION,
        message=EventMessages.agent_completed("PolicyValidationAgent"),
        metadata={"rules_evaluated": 7, "rules_failed": 0},
    )

    # Return from the LangGraph node:
    return {"trace_events": [start_evt, done_evt], ...}

Design principles
-----------------
- ``AgentTracer`` never returns raw ``dict`` objects.  Every method returns
  a fully-typed ``TraceEvent`` instance so that static analysis catches
  misuse at the call site.
- ``event_id`` and ``timestamp`` are always auto-generated inside
  ``create_event``; callers must never supply them.  This eliminates an
  entire class of bugs where two events share an ID or carry a stale
  timestamp.
- The convenience methods (``start_event``, ``success_event``, etc.) are
  the recommended entry points.  ``create_event`` is the escape hatch for
  situations where the caller needs explicit control over status/severity.
- All methods are ``@staticmethod`` because ``AgentTracer`` is stateless;
  there is no instance state to manage.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from app.models.trace import TraceEvent, TraceSeverity, TraceStatus

__all__ = ["AgentTracer"]


class AgentTracer:
    """
    Stateless factory for constructing fully-populated ``TraceEvent`` objects.

    Every method is a ``@staticmethod``; the class is never instantiated.
    Agents import ``AgentTracer`` and call its methods directly::

        event = AgentTracer.success_event(
            agent_name="OCRAgent",
            action="extract_text",
            message="Successfully extracted prescription",
        )

    Status / severity mapping
    -------------------------
    +------------------+-----------------+--------------------+
    | Method           | TraceStatus     | TraceSeverity      |
    +==================+=================+====================+
    | ``start_event``  | STARTED         | INFO               |
    +------------------+-----------------+--------------------+
    | ``success_event``| PASSED          | INFO               |
    +------------------+-----------------+--------------------+
    | ``fail_event``   | FAILED          | ERROR              |
    +------------------+-----------------+--------------------+
    | ``warning_event``| PARTIAL         | WARNING            |
    +------------------+-----------------+--------------------+
    | ``critical_event``| FAILED         | CRITICAL           |
    +------------------+-----------------+--------------------+
    """

    @staticmethod
    def create_event(
        agent_name: str,
        action: str,
        status: TraceStatus,
        severity: TraceSeverity,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> TraceEvent:
        """
        Core factory method: build a ``TraceEvent`` with auto-generated
        identity and timestamp.

        This is the canonical construction path.  The convenience methods
        (``start_event``, ``success_event``, etc.) delegate to this method
        after applying their fixed status/severity mapping.

        Parameters
        ----------
        agent_name:
            Name of the agent emitting the event.  Must be a non-empty
            string (validated by ``TraceEvent``).
        action:
            Snake_case verb phrase identifying the operation being traced
            (e.g. ``"validate_waiting_period"``).  Must be non-empty.
        status:
            Execution outcome for this event (``STARTED``, ``PASSED``,
            ``FAILED``, ``PARTIAL``, ``SKIPPED``).
        severity:
            Importance level for this event (``INFO``, ``WARNING``,
            ``ERROR``, ``CRITICAL``).
        message:
            Human-readable explanation that stands alone in an audit log
            or operations dashboard.  Must be non-empty.
        metadata:
            Optional dict of supplementary key-value pairs (e.g.
            ``{"claim_id": "CLM_001", "processing_time_ms": 432}``).
            Defaults to an empty dict when ``None`` is supplied.

        Returns
        -------
        TraceEvent
            A fully populated ``TraceEvent`` with a fresh UUID4
            ``event_id`` and a UTC ``timestamp`` set to the moment this
            method was called.

        Notes
        -----
        ``event_id`` and ``timestamp`` are always generated here.  Callers
        must not attempt to supply them; doing so would require calling
        ``TraceEvent`` directly, bypassing this factory.
        """
        return TraceEvent(
            event_id=str(uuid.uuid4()),
            agent_name=agent_name,
            action=action,
            status=status,
            severity=severity,
            message=message,
            timestamp=datetime.now(timezone.utc),
            metadata=metadata if metadata is not None else {},
        )

    @staticmethod
    def start_event(
        agent_name: str,
        action: str,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> TraceEvent:
        """
        Create a ``STARTED / INFO`` event indicating an agent has begun work.

        Emit this event at the entry point of every agent node so that the
        UI can render a live "in progress" indicator and so that timeouts
        can be detected by comparing the ``STARTED`` timestamp against wall
        clock time in the Explainability Agent.

        Parameters
        ----------
        agent_name:
            Name of the starting agent.
        action:
            Action being started (use an ``AgentActions`` constant).
        message:
            Human-readable start message (use
            ``EventMessages.agent_started``).
        metadata:
            Optional supplementary context (e.g. ``{"claim_id": "CLM_001",
            "document_count": 2}``).

        Returns
        -------
        TraceEvent
            Status ``STARTED``, severity ``INFO``.
        """
        return AgentTracer.create_event(
            agent_name=agent_name,
            action=action,
            status=TraceStatus.STARTED,
            severity=TraceSeverity.INFO,
            message=message,
            metadata=metadata,
        )

    @staticmethod
    def success_event(
        agent_name: str,
        action: str,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> TraceEvent:
        """
        Create a ``PASSED / INFO`` event indicating successful completion.

        Emit this event when an agent finishes its work without errors and
        all expected outputs were produced.

        Parameters
        ----------
        agent_name:
            Name of the completing agent.
        action:
            Action that completed (use an ``AgentActions`` constant).
        message:
            Human-readable completion message (use
            ``EventMessages.agent_completed``).
        metadata:
            Optional supplementary context (e.g. ``{"confidence": 0.97,
            "document_type": "PRESCRIPTION"}``).

        Returns
        -------
        TraceEvent
            Status ``PASSED``, severity ``INFO``.
        """
        return AgentTracer.create_event(
            agent_name=agent_name,
            action=action,
            status=TraceStatus.PASSED,
            severity=TraceSeverity.INFO,
            message=message,
            metadata=metadata,
        )

    @staticmethod
    def fail_event(
        agent_name: str,
        action: str,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> TraceEvent:
        """
        Create a ``FAILED / ERROR`` event indicating a recoverable component
        failure.

        Emit this event when an agent encounters an error that degrades
        output quality but does not crash the pipeline.  The pipeline
        continues with whatever partial output is available; the confidence
        score should be decreased by the emitting agent.

        Parameters
        ----------
        agent_name:
            Name of the agent that failed.
        action:
            Action that failed (use an ``AgentActions`` constant).
        message:
            Specific, actionable error message (use
            ``EventMessages.processing_error``).
        metadata:
            Optional context (e.g. ``{"error_type": "LLMTimeout",
            "document_id": "F004", "retry_count": 2}``).

        Returns
        -------
        TraceEvent
            Status ``FAILED``, severity ``ERROR``.
        """
        return AgentTracer.create_event(
            agent_name=agent_name,
            action=action,
            status=TraceStatus.FAILED,
            severity=TraceSeverity.ERROR,
            message=message,
            metadata=metadata,
        )

    @staticmethod
    def warning_event(
        agent_name: str,
        action: str,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> TraceEvent:
        """
        Create a ``PARTIAL / WARNING`` event indicating degraded but usable
        output.

        Emit this event when an agent completes but with reduced reliability —
        for example, when a document is only partially readable, when a
        required field was extracted with low confidence, or when a document
        alteration was detected below the fraud threshold.

        Parameters
        ----------
        agent_name:
            Name of the agent emitting the warning.
        action:
            Action that produced degraded output.
        message:
            Specific description of the degradation (use
            ``EventMessages.ocr_warning``).
        metadata:
            Optional context (e.g. ``{"field": "registration_number",
            "confidence": 0.41, "document_id": "F007"}``).

        Returns
        -------
        TraceEvent
            Status ``PARTIAL``, severity ``WARNING``.
        """
        return AgentTracer.create_event(
            agent_name=agent_name,
            action=action,
            status=TraceStatus.PARTIAL,
            severity=TraceSeverity.WARNING,
            message=message,
            metadata=metadata,
        )

    @staticmethod
    def critical_event(
        agent_name: str,
        action: str,
        message: str,
        metadata: dict[str, object] | None = None,
    ) -> TraceEvent:
        """
        Create a ``FAILED / CRITICAL`` event indicating a condition that
        requires immediate manual intervention.

        Emit this event when a failure is severe enough that the claim cannot
        be processed reliably — for example, when a member is not found in
        the policy roster, when no readable documents are available for a
        required type, or when the fraud score exceeds the mandatory review
        threshold.

        Downstream consumers (the Decision Agent, the routing functions)
        should inspect ``is_critical`` on ``TraceEvent`` to detect these
        events and respond by setting ``ClaimStatus.MANUAL_REVIEW``.

        Parameters
        ----------
        agent_name:
            Name of the agent emitting the critical event.
        action:
            Action that triggered the critical condition.
        message:
            Specific, actionable description (use
            ``EventMessages.manual_review_triggered``).
        metadata:
            Optional context (e.g. ``{"fraud_score": 0.92,
            "threshold": 0.80, "claim_id": "CLM_001"}``).

        Returns
        -------
        TraceEvent
            Status ``FAILED``, severity ``CRITICAL``.
        """
        return AgentTracer.create_event(
            agent_name=agent_name,
            action=action,
            status=TraceStatus.FAILED,
            severity=TraceSeverity.CRITICAL,
            message=message,
            metadata=metadata,
        )