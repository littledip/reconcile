"""Pydantic schema for the transaction records the agents operate on.

Deliberately small for Week 1 — just enough fields for the Classification
Agent to have something meaningful to reason about. Will grow in weeks 3-4
when the Reconciliation/Matching and Anomaly Detection agents need to compare
records across the two simulated systems (processor ledger vs. merchant
order system).
"""
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TransactionSource(str, Enum):
    PROCESSOR_LEDGER = "processor_ledger"
    MERCHANT_ORDER_SYSTEM = "merchant_order_system"


class TransactionType(str, Enum):
    """Labels the Classification Agent assigns."""

    PAYMENT = "payment"
    REFUND = "refund"
    CHARGEBACK = "chargeback"
    DUPLICATE_CHARGE = "duplicate_charge"
    FEE = "fee"
    UNKNOWN = "unknown"


class Transaction(BaseModel):
    transaction_id: str
    source: TransactionSource
    merchant_id: str
    amount: Decimal
    currency: str = "USD"
    timestamp: datetime
    description: Optional[str] = None
    raw_metadata: dict = Field(default_factory=dict)


class ClassificationResult(BaseModel):
    transaction_id: str
    transaction_type: TransactionType
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
