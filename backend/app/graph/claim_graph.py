"""
LangGraph claim-processing graph for the Plum Health Insurance Claims
Processing System.

This module constructs, wires, and compiles the ``StateGraph`` that
orchestrates the multi-agent claim-processing pipeline.  It is the single
place in the codebase where the graph topology is defined: nodes are
registered, edges are declared, and the compiled runnable is exported.

Pipeline topology
-----------------
::

    START
      ↓
    document_classifier
      ↓
    document_verifier
      ↓ (conditional)
      ├── [unreadable doc or errors] → decision_agent
      └── [all clear]                → ocr_extractor
                                           ↓
                                     policy_validator
                                           ↓
                                     fraud_detector
                                           ↓ (conditional — always)
                                     decision_agent
                                           ↓
                                     explainability_agent
                                           ↓
                                         END

Design principles
-----------------
- Nodes are thin skeleton functions that accept ``OverallState`` and return
  ``dict``.  Business logic lives in dedicated agent modules that will be
  injected into these stubs in subsequent implementation steps.
- Routing functions are imported from ``app.graph.routes`` and kept separate
  from the graph wiring to make them independently testable.
- The compiled ``app_graph`` is the only public export of this module.
  Callers (the FastAPI layer) invoke it via ``app_graph.invoke()`` or
  ``app_graph.astream()``.

Exports
-------
``app_graph``
    The compiled LangGraph ``CompiledGraph`` runnable.
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.graph.routes import route_after_evaluation, route_after_verification
from app.graph.state import OverallState
from app.agents.document_classifier import classify_documents_agent

# ---------------------------------------------------------------------------
# Node skeleton functions
# ---------------------------------------------------------------------------
# Every node accepts the full OverallState and returns a dict of field
# updates to be merged by LangGraph using the per-field reducers defined in
# app.graph.state.  Returning {} is valid and means the node made no state
# changes.  Nodes must not mutate the state object they receive.
# ---------------------------------------------------------------------------


def document_classifier(
    state: OverallState,
) -> dict[str, object]:
    """
    Classify each uploaded document by its type.

    Delegates to ``classify_documents_agent`` in
    ``app.agents.document_classifier``.  See that module for the full
    implementation contract.

    Parameters
    ----------
    state:
        Shared pipeline state at graph entry.  The ``claim.documents``
        list contains the raw uploaded documents awaiting classification.

    Returns
    -------
    dict[str, object]
        Partial state update containing ``document_results``,
        ``trace_events``, and optionally ``errors``.
    """
    return classify_documents_agent(state)


def document_verifier(state: OverallState) -> dict[str, object]:
    """
    Verify that the correct document types are present for the claim category.

    Responsibilities (to be implemented in the agent module):
    - Cross-reference the classified document types against the policy's
      ``document_requirements`` for ``state["claim"].claim_category``.
    - Flag any required document type that is absent or represented only by
      an ``UNKNOWN``-classified document.
    - Set ``is_readable = False`` on documents whose OCR confidence fell
      below the readability threshold during classification.
    - Produce specific, actionable error messages (e.g. "Uploaded two
      PRESCRIPTION documents; a HOSPITAL_BILL is required") rather than
      generic failures.
    - Append ``TraceEvent`` objects for each verification check performed.

    Parameters
    ----------
    state:
        Shared pipeline state after document classification.

    Returns
    -------
    dict[str, object]
        Partial state update.  Expected keys when implemented:
        ``document_results``, ``trace_events``, ``errors``, ``warnings``,
        ``status``.
    """
    return {}


def ocr_extractor(state: OverallState) -> dict[str, object]:
    """
    Extract structured information from each verified document via OCR.

    Responsibilities (to be implemented in the agent module):
    - Send each readable ``UploadedDocument`` through Gemini Vision with a
      document-type-specific extraction prompt.
    - Validate the raw LLM response against the appropriate
      ``*Extraction`` Pydantic model (``PrescriptionExtraction``,
      ``HospitalBillExtraction``, etc.).
    - Wrap each validated extraction in an ``ExtractionResult`` and append
      it to ``extractions``.
    - Decrease ``confidence_score`` proportionally for documents with low
      extraction confidence or partially unreadable fields.
    - Record ``ProcessingError`` for LLM timeouts or schema validation
      failures; record ``ProcessingWarning`` for low-confidence fields.

    Parameters
    ----------
    state:
        Shared pipeline state after successful document verification.

    Returns
    -------
    dict[str, object]
        Partial state update.  Expected keys when implemented:
        ``extractions``, ``confidence_score``, ``trace_events``,
        ``errors``, ``warnings``, ``status``.
    """
    return {}


def policy_validator(state: OverallState) -> dict[str, object]:
    """
    Evaluate the claim against all applicable policy rules.

    Responsibilities (to be implemented in the agent module):
    - Load the policy configuration and locate the member record by
      ``state["claim"].member_id``.
    - Execute each applicable rule in order:
        - Member eligibility and active policy check.
        - Initial waiting period (30 days from join date).
        - Condition-specific waiting periods (diabetes: 90 days, etc.).
        - Coverage category check (is ``claim_category`` covered?).
        - Sub-limit enforcement for the claim category.
        - Per-claim limit check (``coverage.per_claim_limit``).
        - Exclusion check (excluded conditions, procedures, items).
        - Pre-authorisation requirement check.
        - Network hospital status and applicable discount.
    - Produce one ``ValidationResult`` per rule and append to
      ``validations``.
    - Apply network discount and co-pay arithmetic; do not store the
      final approved amount here (that is the Decision Agent's
      responsibility).

    Parameters
    ----------
    state:
        Shared pipeline state after OCR extraction.

    Returns
    -------
    dict[str, object]
        Partial state update.  Expected keys when implemented:
        ``validations``, ``confidence_score``, ``trace_events``,
        ``errors``, ``warnings``, ``status``.
    """
    return {}


def fraud_detector(state: OverallState) -> dict[str, object]:
    """
    Evaluate the claim for suspicious patterns and produce a fraud assessment.

    Responsibilities (to be implemented in the agent module):
    - Check same-day claim count against ``fraud_thresholds.same_day_claims_limit``.
    - Check monthly claim count against ``fraud_thresholds.monthly_claims_limit``.
    - Check claimed amount against ``fraud_thresholds.high_value_claim_threshold``.
    - Inspect ``ExtractionResult`` warnings for document-alteration signals.
    - Compute an aggregate ``fraud_score`` from the detected signals.
    - Set ``requires_manual_review = True`` when
      ``fraud_score >= fraud_thresholds.fraud_score_manual_review_threshold``
      or when ``claimed_amount > fraud_thresholds.auto_manual_review_above``.
    - Populate ``fraud_result`` with the complete ``FraudResult``.

    Parameters
    ----------
    state:
        Shared pipeline state after policy validation.

    Returns
    -------
    dict[str, object]
        Partial state update.  Expected keys when implemented:
        ``fraud_result``, ``confidence_score``, ``trace_events``,
        ``errors``, ``warnings``, ``status``.
    """
    return {}


def decision_agent(state: OverallState) -> dict[str, object]:
    """
    Produce the final claim decision from the accumulated pipeline state.

    Responsibilities (to be implemented in the agent module):
    - Aggregate all ``ValidationResult`` objects to determine the overall
      eligibility outcome.
    - If ``fraud_result.requires_manual_review`` is ``True``, produce
      ``DecisionType.MANUAL_REVIEW`` regardless of policy outcome.
    - If any required validation failed, produce ``REJECTED`` with a
      specific reason derived from the failing ``ValidationResult``.
    - If some line items were excluded but others approved, produce
      ``PARTIAL`` with a ``DecisionBreakdown`` showing the itemised
      calculation.
    - Otherwise produce ``APPROVED`` with the full ``DecisionBreakdown``.
    - If the pipeline arrived here via the early-termination route (missing
      or unreadable documents), produce ``REJECTED`` or ``MANUAL_REVIEW``
      based on the recorded errors.
    - Set ``decision`` and update ``status`` to ``COMPLETED``,
      ``FAILED``, or ``MANUAL_REVIEW``.

    Parameters
    ----------
    state:
        Shared pipeline state.  This node is reachable from both
        ``document_verifier`` (early termination) and ``fraud_detector``
        (normal path).

    Returns
    -------
    dict[str, object]
        Partial state update.  Expected keys when implemented:
        ``decision``, ``status``, ``trace_events``, ``errors``.
    """
    return {}


def explainability_agent(state: OverallState) -> dict[str, object]:
    """
    Build a human-readable audit summary from the completed pipeline state.

    Responsibilities (to be implemented in the agent module):
    - Iterate over ``trace_events`` and produce a ``TraceSummary``
      (total / passed / failed / warning / critical event counts).
    - Generate one ``AgentExecutionSummary`` per agent by grouping events
      by ``agent_name`` and computing durations from ``STARTED`` /
      terminal event pairs.
    - Attach the summaries to the ``decision`` record or to a dedicated
      field so the operations dashboard and the API response can render
      the full explainability trace.
    - Emit a final ``TraceEvent`` marking the pipeline as complete.

    Parameters
    ----------
    state:
        Shared pipeline state after the Decision Agent has run.
        ``state["decision"]`` is guaranteed non-None at this point for
        the normal path; the early-termination path also reaches this
        node via the decision agent, so ``state["decision"]`` should be
        checked defensively.

    Returns
    -------
    dict[str, object]
        Partial state update.  Expected keys when implemented:
        ``trace_events``, ``status``.
    """
    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _build_graph() -> StateGraph:  # type: ignore[type-arg]
    """
    Construct and wire the ``StateGraph`` without compiling it.

    Keeping construction in a private factory function makes the graph
    independently testable: tests can call ``_build_graph()`` and inspect
    the topology before invoking ``compile()``.

    Returns
    -------
    StateGraph
        A fully wired but uncompiled ``StateGraph[OverallState]``.
    """
    workflow: StateGraph = StateGraph(OverallState)  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Node registration
    # ------------------------------------------------------------------
    workflow.add_node("document_classifier", document_classifier)
    workflow.add_node("document_verifier", document_verifier)
    workflow.add_node("ocr_extractor", ocr_extractor)
    workflow.add_node("policy_validator", policy_validator)
    workflow.add_node("fraud_detector", fraud_detector)
    workflow.add_node("decision_agent", decision_agent)
    workflow.add_node("explainability_agent", explainability_agent)

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    workflow.set_entry_point("document_classifier")

    # ------------------------------------------------------------------
    # Deterministic edges
    # ------------------------------------------------------------------
    # document_classifier → document_verifier (always)
    workflow.add_edge("document_classifier", "document_verifier")

    # ocr_extractor → policy_validator (always)
    workflow.add_edge("ocr_extractor", "policy_validator")

    # policy_validator → fraud_detector (always)
    workflow.add_edge("policy_validator", "fraud_detector")

    # decision_agent → explainability_agent (always)
    workflow.add_edge("decision_agent", "explainability_agent")

    # explainability_agent → END (always)
    workflow.add_edge("explainability_agent", END)

    # ------------------------------------------------------------------
    # Conditional edges
    # ------------------------------------------------------------------
    # document_verifier: branch on readability and error state
    workflow.add_conditional_edges(
        "document_verifier",
        route_after_verification,
        {
            "decision_agent": "decision_agent",
            "ocr_extractor": "ocr_extractor",
        },
    )

    # fraud_detector: unconditional continuation via named router
    # (routing function is named to allow future branching without
    # topology changes)
    workflow.add_conditional_edges(
        "fraud_detector",
        route_after_evaluation,
        {
            "decision_agent": "decision_agent",
        },
    )

    return workflow


# ---------------------------------------------------------------------------
# Compiled graph — public export
# ---------------------------------------------------------------------------

app_graph = _build_graph().compile()
"""
Compiled LangGraph runnable for the claim-processing pipeline.

Usage (synchronous)::

    from app.graph.claim_graph import app_graph

    initial_state: OverallState = { ... }
    final_state = app_graph.invoke(initial_state)

Usage (async streaming)::

    async for chunk in app_graph.astream(initial_state):
        # chunk is a dict of node-name → partial state update
        process(chunk)

The compiled graph is stateless and thread-safe; a single instance may be
shared across all FastAPI request handlers.
"""

__all__ = ["app_graph"]
