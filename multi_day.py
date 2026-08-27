"""
ReVive :: multi_day.py
Wraps the Negotiator's static incentive choice in a tiny epsilon-greedy
multi-armed bandit run across simulated days, so the repo demonstrates
learning-over-time rather than only a single snapshot.

Arms: "no_incentive", "small_cashback", "emi_conversion"
Reward: 1 if the customer completes payment, 0 otherwise, drawn from the
same hidden _true_recoverability signal as the rest of the sim.

Run: python -m revive.multi_day
"""

from __future__ import annotations
import json
import random
from pathlib import Path

from .data_generator import generate_dataset, to_dicts

random.seed(3)

ARMS = ["no_incentive", "small_cashback", "emi_conversion"]

# Hidden multipliers on true recoverability, what the bandit must discover.
_ARM_TRUE_LIFT = {"no_incentive": 1.0, "small_cashback": 1.20, "emi_conversion": 1.35}
_ARM_COST_PER_SEND = {"no_incentive": 0.0, "small_cashback": 15.0, "emi_conversion": 5.0}

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "learning_curve.json"


class EpsilonGreedyNegotiator:
    # epsilon=0.15: standard starting point for a 3-armed bandit, not tuned
    # against this specific reward distribution.
    def __init__(self, epsilon: float = 0.15) -> None:
        self.epsilon = epsilon
        self.counts = {a: 0 for a in ARMS}
        self.values = {a: 0.0 for a in ARMS}  # running average net reward per arm

    def choose(self) -> str:
        if random.random() < self.epsilon or all(c == 0 for c in self.counts.values()):
            return random.choice(ARMS)
        return max(self.values, key=self.values.get)

    def update(self, arm: str, reward: float) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


def run(n_days: int = 10, txns_per_day: int = 150) -> dict:
    bandit = EpsilonGreedyNegotiator(epsilon=0.15)
    daily_results = []

    for day in range(1, n_days + 1):
        txns = to_dicts(generate_dataset(txns_per_day))
        day_recovered = 0.0
        day_cost = 0.0
        arm_picks = {a: 0 for a in ARMS}

        for t in txns:
            arm = bandit.choose()
            arm_picks[arm] += 1
            p = min(0.97, t["_true_recoverability"] * _ARM_TRUE_LIFT[arm])
            success = random.random() < p
            reward = (t["amount"] if success else 0.0) - _ARM_COST_PER_SEND[arm]
            bandit.update(arm, reward)
            if success:
                day_recovered += t["amount"]
            day_cost += _ARM_COST_PER_SEND[arm]

        daily_results.append({
            "day": day,
            "recovered_amount": round(day_recovered, 2),
            "incentive_cost": round(day_cost, 2),
            "net_recovered": round(day_recovered - day_cost, 2),
            "arm_distribution": arm_picks,
            "learned_values": {a: round(v, 1) for a, v in bandit.values.items()},
        })

    summary = {
        "n_days": n_days,
        "txns_per_day": txns_per_day,
        "final_learned_best_arm": max(bandit.values, key=bandit.values.get),
        "daily_results": daily_results,
    }
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    result = run()
    print(f"Learned best incentive arm after {result['n_days']} days: {result['final_learned_best_arm']}")
    print(f"Day 1 net recovered: {result['daily_results'][0]['net_recovered']}")
    print(f"Day {result['n_days']} net recovered: {result['daily_results'][-1]['net_recovered']}")
    print(f"\nFull learning curve written to {OUTPUT_PATH}")
