"""
ReVive :: data_generator.py
Generates synthetic failed-payment events for an Indian digital-payments
merchant (no real user data). Each transaction carries the raw signals a
payment gateway actually exposes so the downstream agents have something
real to reason over.
"""

from __future__ import annotations
import random
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta

random.seed(42)

# Reference vocab
FAILURE_REASONS = [
    "INSUFFICIENT_FUNDS",
    "BANK_SERVER_TIMEOUT",
    "OTP_EXPIRED",
    "CARD_DECLINED_RISK",
    "UPI_PIN_MISMATCH",
    "NETWORK_DROP",
    "LIMIT_EXCEEDED",
    "ISSUER_DOWN",
]

INSTRUMENTS = ["UPI", "CREDIT_CARD", "DEBIT_CARD", "NETBANKING", "WALLET"]

CUSTOMER_SEGMENTS = ["new", "returning", "loyal", "high_value"]

CITIES = ["Gurugram", "Mumbai", "Bengaluru", "Patna", "Delhi", "Pune", "Hyderabad"]


@dataclass
class FailedTransaction:
    txn_id: str
    customer_id: str
    amount: float
    instrument: str
    failure_reason: str
    retry_count: int
    hour_of_day: int
    day_of_week: int
    customer_segment: str
    customer_tenure_days: int
    city: str
    timestamp: str
    # "ground truth" used only by the simulator, never shown to the agents.
    _true_recoverability: float = field(default=0.0, repr=False)


def _true_recoverability(reason: str, amount: float, segment: str, retry_count: int) -> float:
    """Hidden probability a customer completes payment under an ideal strategy."""
    base = {
        "INSUFFICIENT_FUNDS": 0.35,
        "BANK_SERVER_TIMEOUT": 0.80,
        "OTP_EXPIRED": 0.75,
        "CARD_DECLINED_RISK": 0.25,
        "UPI_PIN_MISMATCH": 0.65,
        "NETWORK_DROP": 0.85,
        "LIMIT_EXCEEDED": 0.30,
        "ISSUER_DOWN": 0.70,
    }[reason]
    segment_boost = {"new": -0.05, "returning": 0.05, "loyal": 0.15, "high_value": 0.10}[segment]
    amount_penalty = -0.10 if amount > 15000 else 0.0
    retry_fatigue = -0.05 * retry_count
    return max(0.02, min(0.97, base + segment_boost + amount_penalty + retry_fatigue))


def generate_dataset(n: int = 500) -> list[FailedTransaction]:
    txns = []
    for _ in range(n):
        reason = random.choice(FAILURE_REASONS)
        segment = random.choices(CUSTOMER_SEGMENTS, weights=[0.3, 0.35, 0.2, 0.15])[0]
        amount = round(random.lognormvariate(7.5, 0.9), 2)
        retry_count = random.choices([0, 1, 2, 3], weights=[0.55, 0.25, 0.13, 0.07])[0]
        hour = random.randint(0, 23)
        dow = random.randint(0, 6)
        tenure = random.randint(0, 1800)
        ts = datetime(2026, 8, random.randint(1, 27), hour, random.randint(0, 59))

        txns.append(
            FailedTransaction(
                txn_id=f"txn_{uuid.uuid4().hex[:10]}",
                customer_id=f"cust_{uuid.uuid4().hex[:8]}",
                amount=amount,
                instrument=random.choice(INSTRUMENTS),
                failure_reason=reason,
                retry_count=retry_count,
                hour_of_day=hour,
                day_of_week=dow,
                customer_segment=segment,
                customer_tenure_days=tenure,
                city=random.choice(CITIES),
                timestamp=ts.isoformat(),
                _true_recoverability=_true_recoverability(reason, amount, segment, retry_count),
            )
        )
    return txns


def to_dicts(txns: list[FailedTransaction]) -> list[dict]:
    return [asdict(t) for t in txns]


if __name__ == "__main__":
    sample = generate_dataset(5)
    for t in sample:
        print(t)
