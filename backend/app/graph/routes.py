"""
Routing functions for the Plum Health Insurance Claims Processing graph.

This module contains exactly two pure routing functions used as edge
selectors in the LangGraph ``StateGraph``.  Both functions are stateless,
side-effect-free, and deterministic: given the same ``OverallState`` they
always return the same node name.

Design principles
-----------------
- Routing logic is kept intentionally minimal.  These functions inspect
  only the structural conditions that determine pipeline branching; all
  claim-level business decisions (APPROVED / PARTIAL / REJECTED /
  MANUAL_REVIEW) are delegated entirely to the Decision Agent node.
- No logging, no mutations, no external calls.
- Return values are literal strings that must match registered node names
  in ``claim_graph.py``; a mismatch raises at graph-compile time, not
  at runtime.

Route map
---------
::

    document_verifier
        ↓ route_after_verification
        ├── "decision_agent"   (unreadable document OR pipeline error)
        └── "ocr_extractor"    (all documents readable, no errors)

    fraud_detector
        ↓ route_after_evaluation
        └── "decision_agent"   (always — linear continuation)
"""

from __future__ import annotations

from app.graph.state import OverallState


def route_after_verification(state: OverallState) -> str:
    """
    Select the next node after the Document Verification Agent completes.

    The verification agent has two possible outcomes:

    1. **Early termination** — at least one uploaded document is flagged as
       unreadable (``is_readable == False``), or the verification agent
       recorded one or more ``ProcessingError`` objects.  In either case the
       pipeline cannot proceed to OCR extraction because it lacks a
       processable document set.  The claim is forwarded directly to the
       Decision Agent, which will produce a ``REJECTED`` or ``MANUAL_REVIEW``
       decision with an explanation derived from the verification failures.

    2. **Normal continuation** — all documents are readable and no errors
       were recorded.  The pipeline proceeds to OCR extraction.

    Parameters
    ----------
    state:
        The current ``OverallState`` after the document verification node
        has applied its updates.

    Returns
    -------
    str
        ``"decision_agent"`` when the pipeline must skip OCR due to
        unreadable documents or verification errors; ``"ocr_extractor"``
        when processing can continue normally.

    Notes
    -----
    The ``documents`` list on ``ClaimSubmission`` is guaranteed non-empty
    by the ``ClaimSubmission`` validator (``min_length=1``), so iterating
    over it here is safe without a length guard.
    """
    documents = state["claim"].documents
    for document in documents:
        if not document.is_readable:
            return "decision_agent"

    errors = state["errors"]
    if errors:
        return "decision_agent"

    return "ocr_extractor"


def route_after_evaluation(state: OverallState) -> str:
    """
    Select the next node after the Fraud Detection Agent completes.

    The graph is intentionally linear from the fraud detector onward.
    Regardless of what the fraud agent found, control always passes to
    the Decision Agent, which is the single node responsible for
    interpreting fraud signals and producing the final claim outcome
    (APPROVED / PARTIAL / REJECTED / MANUAL_REVIEW).

    This routing function exists as a named callable — rather than a direct
    edge — so that future branching logic (e.g. an express path for
    zero-risk claims) can be added here without touching the graph
    topology in ``claim_graph.py``.

    Parameters
    ----------
    state:
        The current ``OverallState`` after the fraud detection node has
        applied its updates.  Not inspected; present for signature
        compatibility with LangGraph's conditional-edge API.

    Returns
    -------
    str
        Always ``"decision_agent"``.
    """
    _ = state  # acknowledged; routing is unconditional at this stage
    return "decision_agent"