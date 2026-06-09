"""
Centralised event naming constants and semantic message builders for the
Plum Health Insurance Claims Processing System observability layer.

This module provides two purely declarative utilities that every agent uses
to produce consistent, human-readable telemetry without embedding raw strings
in business logic:

``AgentActions``
    Class-level string constants for every traceable action in the pipeline.
    Agents reference these constants when constructing ``TraceEvent`` objects
    via ``AgentTracer``, eliminating magic strings across the codebase.

``EventMessages``
    Static-method factory that returns formatted, human-readable strings for
    common telemetry scenarios.  All methods are pure functions: they accept
    context strings and return formatted messages with no side effects.

Design principles
-----------------
- This module **never** creates ``TraceEvent`` objects.  Event construction
  is the sole responsibility of ``AgentTracer`` in ``tracer.py``.  The
  separation keeps concerns clean: constants + messages here; event
  lifecycle there.
- All constants are plain ``str`` values so they serialise to JSON
  transparently without custom encoders.
- ``EventMessages`` methods follow a consistent ``{context}: {detail}``
  format so that operations dashboards can parse and group messages
  programmatically.
"""

from __future__ import annotations

__all__ = ["AgentActions", "EventMessages"]


# ---------------------------------------------------------------------------
# AgentActions
# ---------------------------------------------------------------------------


class AgentActions:
    """
    Canonical string constants for every traceable action in the pipeline.

    Agents pass these constants as the ``action`` argument when creating
    ``TraceEvent`` objects via ``AgentTracer``.  Using constants instead of
    inline strings prevents typos, enables IDE autocomplete, and makes it
    trivial to search the codebase for all uses of a particular action.

    Convention
    ----------
    Constants follow the pattern ``VERB_NOUN`` in ``UPPER_SNAKE_CASE``.
    Each constant corresponds to one agent's primary responsibility.  An
    agent may use a constant as its baseline action name and append a
    qualifier (e.g. ``AgentActions.OCR_EXTRACTION + "_prescription"``) when
    it needs to distinguish sub-steps, though in most cases the base constant
    is sufficient.
    """

    DOCUMENT_CLASSIFICATION: str = "classify_document"
    """
    Action emitted by the Document Classification Agent when it attempts to
    identify the type of an uploaded document via Gemini Vision.
    """

    DOCUMENT_VERIFICATION: str = "verify_document"
    """
    Action emitted by the Document Verification Agent when it checks that
    the correct document types are present for the submitted claim category.
    """

    OCR_EXTRACTION: str = "extract_ocr"
    """
    Action emitted by the OCR Extraction Agent when it sends a document
    through Gemini Vision and attempts to extract structured fields.
    """

    POLICY_VALIDATION: str = "validate_policy"
    """
    Action emitted by the Policy Validation Agent when it evaluates a single
    policy rule (eligibility, waiting period, sub-limit, exclusion, etc.).
    """

    FRAUD_DETECTION: str = "detect_fraud"
    """
    Action emitted by the Fraud Detection Agent when it evaluates the claim
    for suspicious patterns and computes the aggregate fraud score.
    """

    DECISION_GENERATION: str = "generate_decision"
    """
    Action emitted by the Decision Agent when it synthesises all upstream
    outputs into a final ``ClaimDecision``.
    """

    EXPLAINABILITY_GENERATION: str = "generate_explainability"
    """
    Action emitted by the Explainability Agent when it builds the human-readable
    audit summary and ``TraceSummary`` from the completed pipeline state.
    """


# ---------------------------------------------------------------------------
# EventMessages
# ---------------------------------------------------------------------------


class EventMessages:
    """
    Static-method factory that returns consistently formatted, human-readable
    event message strings.

    Every method is a pure function: it accepts one or more context strings,
    formats them into a message, and returns the result.  No state is read or
    written; no objects are created.

    Usage by agents
    ---------------
    Agents pass the return value of an ``EventMessages`` method as the
    ``message`` argument to ``AgentTracer``::

        event = AgentTracer.success_event(
            agent_name="DocumentClassificationAgent",
            action=AgentActions.DOCUMENT_CLASSIFICATION,
            message=EventMessages.agent_completed("DocumentClassificationAgent"),
            metadata={"document_id": "F001", "confidence": 0.97},
        )

    Message format contract
    -----------------------
    All messages follow a ``{subject}: {detail}`` pattern so that the
    operations dashboard can split on ``: `` and display subject and detail
    in separate columns when needed.
    """

    @staticmethod
    def agent_started(agent_name: str) -> str:
        """
        Return a message indicating that an agent has begun processing.

        Parameters
        ----------
        agent_name:
            Human-readable name of the agent (e.g. ``"OCRAgent"``).

        Returns
        -------
        str
            Formatted start message.

        Examples
        --------
        >>> EventMessages.agent_started("PolicyValidationAgent")
        'PolicyValidationAgent: started processing.'
        """
        return f"{agent_name}: started processing."

    @staticmethod
    def agent_completed(agent_name: str) -> str:
        """
        Return a message indicating that an agent completed successfully.

        Parameters
        ----------
        agent_name:
            Human-readable name of the agent.

        Returns
        -------
        str
            Formatted completion message.

        Examples
        --------
        >>> EventMessages.agent_completed("FraudDetectionAgent")
        'FraudDetectionAgent: completed successfully.'
        """
        return f"{agent_name}: completed successfully."

    @staticmethod
    def validation_passed(validation_name: str) -> str:
        """
        Return a message indicating that a named validation rule passed.

        Parameters
        ----------
        validation_name:
            Identifier of the rule that passed (e.g.
            ``"WAITING_PERIOD_CHECK"``).

        Returns
        -------
        str
            Formatted validation-passed message.

        Examples
        --------
        >>> EventMessages.validation_passed("MEMBER_ELIGIBILITY_CHECK")
        'Validation passed: MEMBER_ELIGIBILITY_CHECK.'
        """
        return f"Validation passed: {validation_name}."

    @staticmethod
    def validation_failed(validation_name: str) -> str:
        """
        Return a message indicating that a named validation rule failed.

        Parameters
        ----------
        validation_name:
            Identifier of the rule that failed (e.g.
            ``"PRE_AUTH_REQUIRED"``).

        Returns
        -------
        str
            Formatted validation-failed message.

        Examples
        --------
        >>> EventMessages.validation_failed("PER_CLAIM_LIMIT_CHECK")
        'Validation failed: PER_CLAIM_LIMIT_CHECK.'
        """
        return f"Validation failed: {validation_name}."

    @staticmethod
    def ocr_warning(message: str) -> str:
        """
        Return a message describing a non-fatal OCR quality issue.

        Parameters
        ----------
        message:
            Description of the specific OCR issue (e.g. ``"registration
            number obscured by rubber stamp on document F007"``).

        Returns
        -------
        str
            Formatted OCR warning message.

        Examples
        --------
        >>> EventMessages.ocr_warning("doctor registration number obscured by rubber stamp")
        'OCR warning: doctor registration number obscured by rubber stamp.'
        """
        return f"OCR warning: {message}."

    @staticmethod
    def processing_error(message: str) -> str:
        """
        Return a message describing a recoverable pipeline processing error.

        Parameters
        ----------
        message:
            Description of the error (e.g. ``"Gemini Vision API timed out
            after 30s on document F004"``).

        Returns
        -------
        str
            Formatted processing error message.

        Examples
        --------
        >>> EventMessages.processing_error("Gemini Vision API timed out after 30s")
        'Processing error: Gemini Vision API timed out after 30s.'
        """
        return f"Processing error: {message}."

    @staticmethod
    def manual_review_triggered(reason: str) -> str:
        """
        Return a message indicating that the claim has been escalated to
        manual review.

        Parameters
        ----------
        reason:
            Human-readable explanation of why manual review was triggered
            (e.g. ``"fraud score 0.87 exceeds threshold 0.80"``).

        Returns
        -------
        str
            Formatted manual-review escalation message.

        Examples
        --------
        >>> EventMessages.manual_review_triggered("fraud score 0.87 exceeds threshold 0.80")
        'Manual review triggered: fraud score 0.87 exceeds threshold 0.80.'
        """
        return f"Manual review triggered: {reason}."