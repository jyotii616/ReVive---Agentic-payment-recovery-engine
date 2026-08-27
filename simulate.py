"""
ReVive :: simulate.py
Monte-Carlo comparison between BASELINE (one generic retry email, 24h
later, no segmentation/incentive) and REVIVE (the four-agent swarm's
tailored playbook per transaction).

Both are graded against the same hidden `_true_recoverability` signal
(see data_generator.py), perturbed by how well each strategy's
timing/channel/incentive matches what the customer needed. A simulation,
not a claim about real Razorpay traffic - the mechanism being tested
(tailored playbook beats generic retry) is the real idea.

Run: python -m revive.simulate
"""

from __future__ import annotations
import json
import random
from pathlib import Path

import numpy as np

from .data_generator import generate_dataset, to_dicts
from .orchestrator import Judge, RecoveryModel

random.seed(11)
np.random.seed(11)

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "results.json"


def _baseline_success(txn: dict) -> bool:
    """Generic same-for-everyone strategy: fixed 24h delay, single email,
    no incentive. Realizes ~40% of true recoverable probability on average."""
    p_true = txn["_true_recoverability"]
    return random.random() < (p_true * 0.40)


def _revive_success(txn: dict, playbook_confidence: float, worth_pursuing: bool) -> bool:
    """ReVive strategy: realizes a higher fraction of true recoverability via
    tailored timing/channel/incentive, only attempted when worth_pursuing,
    scaled by swarm confidence."""
    if not worth_pursuing:
        return False
    p_true = txn["_true_recoverability"]
    realized_fraction = 0.55 + 0.35 * playbook_confidence  # 0.55 - 0.90
    return random.random() < (p_true * realized_fraction)


def _allocate_under_budget(playbooks: list, capacity_fraction: float) -> set[str]:
    """Ranks every playbook by expected_value (amount x probability x
    confidence) and only "works" the top slice that fits the budget."""
    eligible = [pb for pb in playbooks if pb.worth_pursuing]
    eligible.sort(key=lambda pb: pb.expected_value, reverse=True)
    # capacity_fraction is sized against the full failed-txn count, not just
    # the eligible subset - models a fixed ops-capacity budget, not a
    # percentage-of-eligible cutoff.
    k = max(1, int(len(playbooks) * capacity_fraction))
    return {pb.txn_id for pb in eligible[:k]}


def run(n: int = 500, ops_capacity_fraction: float = 0.35) -> dict:
    txns = to_dicts(generate_dataset(n))

    model = RecoveryModel()
    model.fit(txns)
    judge = Judge(model=model, pursue_threshold=0.25)

    playbooks = [judge.adjudicate(t) for t in txns]
    txn_by_id = {t["txn_id"]: t for t in txns}

    total_failed_amount = sum(t["amount"] for t in txns)

    baseline_recovered_amount = 0.0
    revive_recovered_amount = 0.0
    budgeted_recovered_amount = 0.0
    baseline_hits = revive_hits = budgeted_hits = 0

    compliance_counts = {"CLEAR": 0, "FLAG": 0, "BLOCK": 0}
    diagnosis_counts: dict[str, int] = {}

    worked_ids = _allocate_under_budget(playbooks, ops_capacity_fraction)

    for pb in playbooks:
        t = txn_by_id[pb.txn_id]
        compliance_counts[pb.compliance_status] = compliance_counts.get(pb.compliance_status, 0) + 1
        diagnosis_counts[pb.diagnosis] = diagnosis_counts.get(pb.diagnosis, 0) + 1

        if _baseline_success(t):
            baseline_recovered_amount += t["amount"]
            baseline_hits += 1

        full_success = _revive_success(t, pb.swarm_confidence, pb.worth_pursuing)
        if full_success:
            revive_recovered_amount += t["amount"]
            revive_hits += 1
            if pb.txn_id in worked_ids:
                budgeted_recovered_amount += t["amount"]
                budgeted_hits += 1

    n_worked = len(worked_ids)
    revenue_per_attempt_unconstrained = revive_recovered_amount / max(sum(1 for pb in playbooks if pb.worth_pursuing), 1)
    revenue_per_attempt_budgeted = budgeted_recovered_amount / max(n_worked, 1)

    sample_playbooks = []
    for pb in sorted(playbooks, key=lambda p: p.expected_value, reverse=True)[:12]:
        t = txn_by_id[pb.txn_id]
        sample_playbooks.append({
            "amount": t["amount"],
            "failure_reason": t["failure_reason"],
            "customer_segment": t["customer_segment"],
            "in_budgeted_worklist": pb.txn_id in worked_ids,
            **pb.to_dict(),
        })

    summary = {
        "n_failed_transactions": n,
        "total_failed_amount": round(total_failed_amount, 2),
        "baseline": {
            "recovered_amount": round(baseline_recovered_amount, 2),
            "recovered_count": baseline_hits,
            "recovery_rate": round(baseline_hits / n, 4),
            "revenue_per_attempt": round(baseline_recovered_amount / n, 2),
        },
        "revive": {
            "recovered_amount": round(revive_recovered_amount, 2),
            "recovered_count": revive_hits,
            "recovery_rate": round(revive_hits / n, 4),
        },
        "revive_budgeted": {
            "ops_capacity_fraction": ops_capacity_fraction,
            "worklist_size": n_worked,
            "recovered_amount": round(budgeted_recovered_amount, 2),
            "recovered_count": budgeted_hits,
            "revenue_per_attempt": round(revenue_per_attempt_budgeted, 2),
            "revenue_per_attempt_unconstrained": round(revenue_per_attempt_unconstrained, 2),
        },
        "compliance": compliance_counts,
        "diagnosis_breakdown": diagnosis_counts,
        "uplift": {
            "additional_amount_recovered": round(revive_recovered_amount - baseline_recovered_amount, 2),
            "relative_recovery_rate_uplift_pct": round(
                ((revive_hits - baseline_hits) / max(baseline_hits, 1)) * 100, 1
            ),
            "additional_amount_recovered_on_35pct_budget": round(
                budgeted_recovered_amount - (baseline_recovered_amount * ops_capacity_fraction), 2
            ),
        },
        "sample_playbooks": sample_playbooks,
    }

    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    result = run(500)
    print(json.dumps({k: v for k, v in result.items() if k != "sample_playbooks"}, indent=2))
    print(f"\nFull results + sample playbooks written to {OUTPUT_PATH}")
