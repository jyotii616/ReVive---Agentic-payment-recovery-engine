#!/usr/bin/env python3
"""
ReVive :: revive_cli.py
A plain terminal command to ask "what should I do about this failed
payment," instead of needing a Jupyter notebook.

Usage:
    python revive_cli.py --amount 4500 --reason INSUFFICIENT_FUNDS \
        --hour 23 --segment new --retries 0

    python revive_cli.py --demo     # runs 5 representative cases
"""

from __future__ import annotations
import argparse
import uuid

from src.revive.data_generator import generate_dataset, to_dicts, FAILURE_REASONS, CUSTOMER_SEGMENTS
from src.revive.orchestrator import Judge, RecoveryModel


def _print_playbook(pb_dict: dict) -> None:
    bar = "=" * 60
    print(bar)
    print(f"TXN {pb_dict['txn_id']}  |  recovery probability: {pb_dict['expected_recovery_probability']*100:.1f}%"
          f"  |  {'PURSUE' if pb_dict['worth_pursuing'] else 'DEPRIORITIZE'}")
    print(bar)
    print(f"Diagnosis : {pb_dict['diagnosis']}")
    print(f"Timing    : {pb_dict['timing']}")
    print(f"Channel   : {pb_dict['channel']}")
    print(f"Compliance: {pb_dict['compliance_status']}, {pb_dict['compliance_notes']}")
    print(f"\nDrafted message:\n  {pb_dict['drafted_message']}")
    print("\nReasoning chain:")
    for line in pb_dict["reasoning_chain"]:
        print(f"  - {line}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Get a ReVive recovery playbook for a failed payment.")
    parser.add_argument("--amount", type=float, help="Transaction amount in INR")
    parser.add_argument("--reason", choices=FAILURE_REASONS, help="Failure reason code")
    parser.add_argument("--hour", type=int, default=14, help="Hour of day (0-23) the failure occurred")
    parser.add_argument("--segment", choices=CUSTOMER_SEGMENTS, default="returning")
    parser.add_argument("--retries", type=int, default=0, help="Prior retry count for this customer")
    parser.add_argument("--demo", action="store_true", help="Run 5 representative demo cases instead")
    args = parser.parse_args()

    model = RecoveryModel()
    model.fit(to_dicts(generate_dataset(500)))
    judge = Judge(model=model)

    if args.demo or not (args.amount and args.reason):
        print("Running 5 demo cases (pass --amount and --reason for a single custom case)\n")
        cases = to_dicts(generate_dataset(5))
        for c in cases:
            _print_playbook(judge.adjudicate(c).to_dict())
        return

    txn = {
        "txn_id": f"txn_{uuid.uuid4().hex[:8]}",
        "customer_id": "cust_cli",
        "amount": args.amount,
        "instrument": "UPI",
        "failure_reason": args.reason,
        "retry_count": args.retries,
        "hour_of_day": args.hour,
        "day_of_week": 0,
        "customer_segment": args.segment,
        "customer_tenure_days": 200,
        "city": "Gurugram",
        "timestamp": "2026-08-27T00:00:00",
    }
    _print_playbook(judge.adjudicate(txn).to_dict())


if __name__ == "__main__":
    main()
