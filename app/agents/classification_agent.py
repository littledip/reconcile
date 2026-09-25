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

Three execution modes, auto-selected in priority order by
_resolve_provider() below:
  - Local mode (Ollama reachable at settings.ollama_base_url): calls a local
    model — e.g. an Unsloth-tuned Qwen model imported into Ollama — via
    langchain-ollama, same structured-output binding as LLM mode. Preferred
    automatically whenever it's up: free, no API usage, and what the test
    suite's real-LLM smoke test (test_classification_agent_local_llm.py)
    exercises.
  - LLM mode (ANTHROPIC_API_KEY set, Ollama not reachable): calls Claude via
    langchain-anthropic with structured output bound to ClassificationResult.
  - Heuristic mode (neither available): a simple rule-based classifier, so
    the graph wiring can be built, run, and tested in this environment
    without requiring a live model at all.

settings.llm_provider is an explicit override ("local" / "anthropic" /
"heuristic") that skips auto-detection entirely when set. Tests pin this
directly (tests/conftest.py's autouse _force_heuristic_classification
fixture, and the local-LLM smoke test's own override) rather than relying on
whatever happens to be configured/running on the machine executing them.
"""
from __future__ import annotations

import socket
from typing import Optional, TypedDict
from urllib.parse import urlparse

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


_provider_cache: Optional[str] = None


def ollama_reachable(base_url: str, timeout: float = 0.3) -> bool:
    """Cheap liveness probe: can we open a TCP connection to Ollama's port?

    Deliberately not an HTTP request to /api/tags -- a plain connect is
    enough to decide "is something listening here", and the real
    correctness check (is it actually Ollama, does the configured model
    exist) happens naturally the first time _local_classify() makes a real
    call -- which the local-LLM smoke test exercises directly.
    """
    parsed = urlparse(base_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 11434
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _resolve_provider() -> str:
    """Priority: settings.llm_provider (an explicit override) always wins --
    this is what tests use to pin a specific mode regardless of what's
    actually configured/running on the machine executing them. Otherwise:
    prefer local Qwen via Ollama whenever it's reachable (free, no API
    usage), then real Anthropic if a key is configured, then the heuristic
    fallback.

    The auto-detected result (not the override) is cached per-process after
    the first resolution, so a batch of many transactions doesn't re-probe
    Ollama's port once per transaction -- mirrors memory_store._client's
    singleton-with-reset-hook shape.
    """
    global _provider_cache
    if settings.llm_provider:
        return settings.llm_provider
    if _provider_cache is not None:
        return _provider_cache
    if ollama_reachable(settings.ollama_base_url):
        _provider_cache = "local"
    elif settings.anthropic_api_key:
        _provider_cache = "anthropic"
    else:
        _provider_cache = "heuristic"
    return _provider_cache


def reset_provider_cache_for_tests() -> None:
    """Test-only: drop the cached auto-detected provider so the next call
    re-probes Ollama instead of reusing a resolution from an earlier test.
    Mirrors memory_store.reset_client_for_tests().
    """
    global _provider_cache
    _provider_cache = None


def _local_classify(txn: Transaction) -> ClassificationResult:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url)
    structured_llm = llm.with_structured_output(ClassificationResult)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=txn.model_dump_json()),
    ]
    result = structured_llm.invoke(messages)
    # Same enforcement as _llm_classify -- don't trust the model to copy the
    # transaction_id faithfully into structured output.
    result.transaction_id = txn.transaction_id
    return result


def _llm_classify(txn: Transaction) -> ClassificationResult:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatAnthropic(
        model=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
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

    provider = _resolve_provider()
    if provider == "local":
        result = _local_classify(txn)
    elif provider == "anthropic":
        result = _llm_classify(txn)
    else:
        result = _heuristic_classify(txn)

    return {"transaction": state["transaction"], "result": result.model_dump(mode="json")}


def build_graph():
    """Compile the single-node classification graph."""
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
