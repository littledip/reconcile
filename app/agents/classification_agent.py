"""Week 1 single-agent LangGraph prototype: the Classification Agent.

Goal this week is narrow on purpose: prove the LangGraph plumbing (typed
state, a node, a compiled graph, invocation) works end to end, using the
simplest possible real agent — one that reads a transaction and classifies
it (payment / refund / chargeback / duplicate_charge / fee) with reasoning.

Weeks 3-4 add the Reconciliation/Matching, Anomaly Detection, and
Reasoning/Explanation agents plus an orchestrator graph that routes between
them. This module is deliberately structured so that graph (a StateGraph
with multiple nodes and conditional edges) can be built on top of this one
without a rewrite: the state shape and the "agent = compiled graph +
typed state" pattern carry forward.

Two execution modes:
  - LLM mode (ANTHROPIC_API_KEY set): calls Claude via langchain-anthropic
    with structured output bound to ClassificationResult.
  - Heuristic mode (no API key): a simple rule-based classifier, so the
    graph wiring can be built, run, and tested in this environment without
    requiring a live API key. Swap to LLM mode automatically once a key is
    added to .env — no code changes needed.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import StateGraph, END

from app.config import settings
from app.models.transaction import ClassificationResult, Transaction, TransactionType


class ClassificationState(TypedDict):
    transaction: dict
    result: dict | None


SYSTEM_PROMPT = """You are a transaction classification agent for a payments \
reconciliation platform. Given a single transaction record, classify it into \
exactly one of: payment, refund, chargeback, duplicate_charge, fee, unknown. \
Base your classification on the amount sign, description, and metadata. \
Respond with the transaction_type, a confidence between 0 and 1, and a short \
one-sentence reasoning."""


def _heuristic_classify(txn: Transaction) -> ClassificationResult:
    """Rule-based fallback used when no ANTHROPIC_API_KEY is configured.

    Not meant to be a good classifier — just enough logic to prove the graph
    executes and returns a well-formed ClassificationResult, so the LangGraph
    wiring can be verified without a live model call.
    """
    desc = (txn.description or "").lower()
    amount = txn.amount

    if "dispute" in desc or "chargeback" in desc:
        label, confidence, reason = (
            TransactionType.CHARGEBACK,
            0.6,
            "Description references a dispute/chargeback.",
        )
    elif "fee" in desc:
        label, confidence, reason = (
            TransactionType.FEE,
            0.6,
            "Description references a platform/processing fee.",
        )
    elif "refund" in desc or "return" in desc or amount < 0:
        label, confidence, reason = (
            TransactionType.REFUND,
            0.55,
            "Negative amount or refund/return language.",
        )
    elif "retry" in desc:
        label, confidence, reason = (
            TransactionType.DUPLICATE_CHARGE,
            0.5,
            "Description indicates a retried charge for the same order.",
        )
    elif amount > 0:
        label, confidence, reason = (
            TransactionType.PAYMENT,
            0.55,
            "Positive amount with no other signal — default to payment.",
        )
    else:
        label, confidence, reason = (
            TransactionType.UNKNOWN,
            0.3,
            "No clear signal in description or amount.",
        )

    return ClassificationResult(
        transaction_id=txn.transaction_id,
        transaction_type=label,
        confidence=confidence,
        reasoning=f"[heuristic mode — no ANTHROPIC_API_KEY set] {reason}",
    )


def _llm_classify(txn: Transaction) -> ClassificationResult:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=0,
    )
    structured_llm = llm.with_structured_output(ClassificationResult)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=txn.model_dump_json()),
    ]
    result = structured_llm.invoke(messages)
    # Structured output won't know the transaction_id unless the model copies
    # it faithfully — enforce it directly rather than trusting the model.
    result.transaction_id = txn.transaction_id
    return result


def classify_node(state: ClassificationState) -> ClassificationState:
    txn = Transaction.model_validate(state["transaction"])

    if settings.anthropic_api_key:
        result = _llm_classify(txn)
    else:
        result = _heuristic_classify(txn)

    return {"transaction": state["transaction"], "result": result.model_dump(mode="json")}


def build_graph():
    graph = StateGraph(ClassificationState)
    graph.add_node("classify", classify_node)
    graph.set_entry_point("classify")
    graph.add_edge("classify", END)
    return graph.compile()


_compiled_graph = build_graph()


def classify_transaction(txn: Transaction) -> ClassificationResult:
    """Public entry point: run a single transaction through the compiled graph."""
    output = _compiled_graph.invoke({"transaction": txn.model_dump(mode="json"), "result": None})
    return ClassificationResult.model_validate(output["result"])
