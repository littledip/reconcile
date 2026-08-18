"""Basic tests for the Week 1 LangGraph Classification Agent prototype.

Runs in heuristic mode (no ANTHROPIC_API_KEY needed) so these pass in any
environment, including this build sandbox. They verify the LangGraph wiring
end to end: state in -> node executes -> compiled graph -> well-formed
ClassificationResult out.
"""
from decimal import Decimal

from app.agents.classification_agent import classify_transaction
from app.models.transaction import Transaction, TransactionSource, TransactionType


def _txn(**overrides) -> Transaction:
    base = dict(
        transaction_id="txn_test_1",
        source=TransactionSource.PROCESSOR_LEDGER,
        merchant_id="merch_1",
        amount=Decimal("100.00"),
        currency="USD",
        timestamp="2026-08-10T00:00:00Z",
        description="Card payment - order #1",
    )
    base.update(overrides)
    return Transaction.model_validate(base)


def test_returns_well_formed_result():
    result = classify_transaction(_txn())
    assert result.transaction_id == "txn_test_1"
    assert isinstance(result.transaction_type, TransactionType)
    assert 0.0 <= result.confidence <= 1.0
    assert result.reasoning


def test_refund_language_classified_as_refund():
    result = classify_transaction(_txn(description="Refund issued for returned item", amount=Decimal("-20.00")))
    assert result.transaction_type == TransactionType.REFUND


def test_dispute_language_classified_as_chargeback():
    result = classify_transaction(_txn(description="Dispute filed by cardholder"))
    assert result.transaction_type == TransactionType.CHARGEBACK


def test_fee_language_classified_as_fee():
    result = classify_transaction(_txn(description="Monthly platform processing fee", amount=Decimal("2.50")))
    assert result.transaction_type == TransactionType.FEE
