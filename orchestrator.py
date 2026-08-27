"""
ReVive :: orchestrator.py
THE JUDGE. Takes the three agent opinions (Coroner/Psychologist/Negotiator)
plus a trained logistic regression model, and produces one explainable
Recovery Playbook per transaction: {txn_id, expected_recovery_probability,
confidence, diagnosis, timing, channel, reasoning_chain[]}.

A real trained model plus rule-based agents rather than one black-box
classifier: the agents supply the "why" in plain English, the model
supplies a calibrated probability to rank budget spend. Together, accuracy
and explainability.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from .agents import Coroner, Psychologist, Negotiator, Regulator, AgentOpinion, NegotiatorDecision
from .data_generator import FAILURE_REASONS, INSTRUMENTS, CUSTOMER_SEGMENTS


FEATURE_ORDER = [
    "amount_log",
    "retry_count",
    "hour_of_day",
    "tenure_days_log",
    "is_transient_failure",
    "is_friction_failure",
    "is_structural_failure",
    "segment_score",
]

_SEGMENT_SCORE = {"new": 0.2, "returning": 0.5, "loyal": 0.8, "high_value": 0.7}


def _featurize(txn: dict[str, Any]) -> np.ndarray:
    coroner_class = Coroner().decide(txn).decision
    return np.array(
        [
            np.log1p(txn["amount"]),
            txn["retry_count"],
            txn["hour_of_day"],
            np.log1p(txn["customer_tenure_days"]),
            float(coroner_class == "TRANSIENT"),
            float(coroner_class == "FRICTION"),
            float(coroner_class == "STRUCTURAL"),
            _SEGMENT_SCORE[txn["customer_segment"]],
        ]
    )


class RecoveryModel:
    """Logistic regression trained on the same hidden `_true_recoverability`
    signal the simulator uses to grade outcomes, learned from noisy labels,
    not read off a lookup table."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000)
        self._fitted = False

    def fit(self, txns: list[dict[str, Any]]) -> None:
        X = np.array([_featurize(t) for t in txns])
        rng = np.random.default_rng(7)
        # Sample noisy binary outcomes from the hidden probability so the
        # model has to generalize, not memorize.
        y = rng.binomial(1, [t["_true_recoverability"] for t in txns])
        self.model.fit(X, y)
        self._fitted = True

    def predict_proba(self, txn: dict[str, Any]) -> float:
        if not self._fitted:
            raise RuntimeError("RecoveryModel must be fit() before predicting.")
        x = _featurize(txn).reshape(1, -1)
        return float(self.model.predict_proba(x)[0][1])


@dataclass
class RecoveryPlaybook:
    txn_id: str
    expected_recovery_probability: float
    swarm_confidence: float
    diagnosis: str
    timing: str
    channel: str
    drafted_message: str
    compliance_status: str      # CLEAR / FLAG / BLOCK
    compliance_notes: str
    reasoning_chain: list[str]
    worth_pursuing: bool
    expected_value: float       # amount * probability * confidence, used for budget ranking

    def to_dict(self) -> dict[str, Any]:
        return {
            "txn_id": self.txn_id,
            "expected_recovery_probability": round(self.expected_recovery_probability, 3),
            "swarm_confidence": round(self.swarm_confidence, 3),
            "diagnosis": self.diagnosis,
            "timing": self.timing,
            "channel": self.channel,
            "drafted_message": self.drafted_message,
            "compliance_status": self.compliance_status,
            "compliance_notes": self.compliance_notes,
            "reasoning_chain": self.reasoning_chain,
            "worth_pursuing": self.worth_pursuing,
            "expected_value": round(self.expected_value, 2),
        }


class Judge:
    def __init__(self, model: RecoveryModel, pursue_threshold: float = 0.25) -> None:
        self.coroner = Coroner()
        self.psychologist = Psychologist()
        self.negotiator = Negotiator()
        self.regulator = Regulator()
        self.model = model
        self.pursue_threshold = pursue_threshold

    def adjudicate(self, txn: dict[str, Any], use_llm_for_message: bool = False) -> RecoveryPlaybook:
        c: AgentOpinion = self.coroner.decide(txn)
        p: AgentOpinion = self.psychologist.decide(txn)
        n: NegotiatorDecision = self.negotiator.decide(txn)
        r: AgentOpinion = self.regulator.review(txn, n, txn["retry_count"])
        prob = self.model.predict_proba(txn)

        swarm_confidence = float(np.mean([c.confidence, p.confidence, n.confidence]))
        # Confidence takes a hit when agents disagree, e.g. Coroner calls it
        # structural but Negotiator offers no incentive. Hand-set heuristic,
        # not fit to data - treat as illustrative, not calibrated.
        disagreement_penalty = 0.1 if (c.decision == "STRUCTURAL" and n.incentive_type == "none") else 0.0
        swarm_confidence = max(0.05, swarm_confidence - disagreement_penalty)

        blocked = r.decision == "BLOCK"
        # threshold=0.25 sits below structural failures' ~30% base rate so
        # they aren't auto-deprioritized; a deliberate choice, not tuned to data.
        worth_pursuing = (prob >= self.pursue_threshold) and not blocked
        message = (
            self.negotiator.draft_message(txn, n, use_llm=use_llm_for_message)
            if not blocked
            else "(withheld, compliance block)"
        )

        reasoning = [
            f"[Coroner, conf={c.confidence:.2f}] {c.reasoning}",
            f"[Psychologist, conf={p.confidence:.2f}] {p.reasoning}",
            f"[Negotiator, conf={n.confidence:.2f}] {n.reasoning}",
            f"[Regulator, conf={r.confidence:.2f}] {r.decision}: {r.reasoning}",
            f"[Judge] Model estimates {prob*100:.1f}% recovery probability. "
            f"{'PURSUE' if worth_pursuing else 'DEPRIORITIZE' if not blocked else 'BLOCKED BY REGULATOR'} "
            f"(threshold={self.pursue_threshold:.0%}).",
        ]

        return RecoveryPlaybook(
            txn_id=txn["txn_id"],
            expected_recovery_probability=prob,
            swarm_confidence=swarm_confidence,
            diagnosis=c.decision,
            timing=p.decision,
            channel=n.label,
            drafted_message=message,
            compliance_status=r.decision,
            compliance_notes=r.reasoning,
            reasoning_chain=reasoning,
            worth_pursuing=worth_pursuing,
            expected_value=txn["amount"] * prob * swarm_confidence,
        )
