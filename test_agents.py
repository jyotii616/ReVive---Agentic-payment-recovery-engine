"""
Tests that check the swarm's decisions actually make sense, not just that
the code runs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.revive.agents import Coroner, Psychologist, Negotiator, Regulator, NegotiatorDecision
from src.revive.orchestrator import Judge, RecoveryModel
from src.revive.data_generator import generate_dataset, to_dicts


def _txn(**overrides):
    base = {
        "txn_id": "txn_test", "customer_id": "cust_test", "amount": 5000.0,
        "instrument": "UPI", "failure_reason": "BANK_SERVER_TIMEOUT", "retry_count": 0,
        "hour_of_day": 14, "day_of_week": 2, "customer_segment": "returning",
        "customer_tenure_days": 300, "city": "Gurugram", "timestamp": "2026-08-27T14:00:00",
        "_true_recoverability": 0.5,
    }
    base.update(overrides)
    return base


def test_coroner_classifies_transient_failures_correctly():
    result = Coroner().decide(_txn(failure_reason="NETWORK_DROP"))
    assert result.decision == "TRANSIENT"


def test_coroner_classifies_structural_failures_correctly():
    result = Coroner().decide(_txn(failure_reason="INSUFFICIENT_FUNDS"))
    assert result.decision == "STRUCTURAL"


def test_psychologist_backs_off_after_repeated_retries():
    fresh = Psychologist().decide(_txn(retry_count=0, hour_of_day=14))
    fatigued = Psychologist().decide(_txn(retry_count=3, hour_of_day=14))
    assert "delay" in fatigued.decision.lower() or "48h" in fatigued.decision
    assert fresh.decision != fatigued.decision


def test_negotiator_offers_emi_for_high_value_loyal_customers():
    result = Negotiator().decide(_txn(customer_segment="loyal", amount=20000))
    assert result.incentive_type == "emi"
    assert "EMI" in result.label


def test_negotiator_skips_incentive_for_small_funds_failures():
    result = Negotiator().decide(_txn(failure_reason="INSUFFICIENT_FUNDS", amount=1500))
    assert result.incentive_type == "none"
    assert "no incentive" in result.label.lower()


def test_regulator_flags_new_users_offered_cashback():
    decision = NegotiatorDecision("In-app nudge + small cashback (2%)", "cashback", 0.65, "test")
    result = Regulator().review(_txn(customer_segment="new"), decision, retry_count=0)
    assert result.decision == "FLAG"


def test_regulator_blocks_incentivised_risk_declines():
    decision = NegotiatorDecision("small cashback offer", "cashback", 0.65, "test")
    result = Regulator().review(_txn(failure_reason="CARD_DECLINED_RISK"), decision, retry_count=0)
    assert result.decision == "BLOCK"


def test_regulator_clears_ordinary_returning_customer():
    decision = NegotiatorDecision("WhatsApp, no incentive", "none", 0.7, "test")
    result = Regulator().review(_txn(customer_segment="returning"), decision, retry_count=0)
    assert result.decision == "CLEAR"


def test_regulator_is_keyed_off_structured_field_not_label_text():
    """Regression test: a label without 'cashback' but incentive_type=cashback
    must still flag for a new user, not rely on substring-matching `label`."""
    decision = NegotiatorDecision(label="Send them something nice", incentive_type="cashback", confidence=0.5, reasoning="test")
    result = Regulator().review(_txn(customer_segment="new"), decision, retry_count=0)
    assert result.decision == "FLAG"


def test_regulator_block_severity_survives_message_rewording():
    """Regression test: BLOCK vs FLAG must come from the structured severity
    field, not from scanning message text for a magic word like 'Escalate'."""
    risk_decline = Regulator().review(
        _txn(failure_reason="CARD_DECLINED_RISK"),
        NegotiatorDecision(label="cashback offer", incentive_type="cashback", confidence=0.5, reasoning="test", incentive_pct=0.02),
        retry_count=0,
    )
    assert risk_decline.decision == "BLOCK"
    assert "escalate" not in risk_decline.reasoning.lower()

    over_cap = Regulator().review(
        _txn(customer_segment="new"),
        NegotiatorDecision(label="cashback offer", incentive_type="cashback", confidence=0.5, reasoning="test", incentive_pct=0.10),
        retry_count=0,
    )
    assert over_cap.decision == "BLOCK"
    assert "escalate" not in over_cap.reasoning.lower()


def test_regulator_escalates_to_block_when_incentive_exceeds_cap():
    """A cashback offer within MAX_INCENTIVE_PCT_FOR_NEW_USERS (5%) should
    FLAG for review; one over the cap should BLOCK outright."""
    within_cap = NegotiatorDecision(label="cashback nudge", incentive_type="cashback", confidence=0.5, reasoning="test", incentive_pct=0.02)
    over_cap = NegotiatorDecision(label="cashback nudge", incentive_type="cashback", confidence=0.5, reasoning="test", incentive_pct=0.10)

    result_within = Regulator().review(_txn(customer_segment="new"), within_cap, retry_count=0)
    result_over = Regulator().review(_txn(customer_segment="new"), over_cap, retry_count=0)

    assert result_within.decision == "FLAG"
    assert result_over.decision == "BLOCK"


def test_regulator_handles_multiple_simultaneous_violations():
    """Over the contact cap AND a risk-flagged card being incentivized
    should BLOCK, with both violation reasons in the notes."""
    decision = NegotiatorDecision(label="cashback nudge", incentive_type="cashback", confidence=0.5, reasoning="test")
    result = Regulator().review(
        _txn(failure_reason="CARD_DECLINED_RISK", customer_segment="new"),
        decision,
        retry_count=5,
    )
    assert result.decision == "BLOCK"
    assert "contact" in result.reasoning.lower()
    assert "risk" in result.reasoning.lower()


def test_judge_produces_end_to_end_playbook_with_message_and_compliance():
    model = RecoveryModel()
    model.fit(to_dicts(generate_dataset(200)))
    judge = Judge(model=model)
    playbook = judge.adjudicate(_txn())
    d = playbook.to_dict()
    assert 0.0 <= d["expected_recovery_probability"] <= 1.0
    assert d["compliance_status"] in ("CLEAR", "FLAG", "BLOCK")
    assert d["drafted_message"]
    assert len(d["reasoning_chain"]) == 5


def test_judge_pursue_threshold_boundary_is_inclusive():
    """worth_pursuing uses >=, so landing exactly on the threshold still counts as PURSUE."""
    model = RecoveryModel()
    model.fit(to_dicts(generate_dataset(200)))
    judge = Judge(model=model, pursue_threshold=0.25)

    class _FixedModel:
        def predict_proba(self, txn):
            return 0.25

    judge.model = _FixedModel()
    playbook = judge.adjudicate(_txn())
    assert playbook.worth_pursuing is True


def test_judge_conflicting_signals_do_not_crash_and_produce_valid_playbook():
    """Conflicting agent signals (retry fatigue vs structural urgency vs a
    compliance flag) should still reconcile into one valid playbook."""
    model = RecoveryModel()
    model.fit(to_dicts(generate_dataset(200)))
    judge = Judge(model=model)
    txn = _txn(
        failure_reason="INSUFFICIENT_FUNDS",
        retry_count=3,
        customer_segment="new",
        amount=2000,
    )
    playbook = judge.adjudicate(txn)
    d = playbook.to_dict()
    assert d["diagnosis"] == "STRUCTURAL"
    assert d["compliance_status"] in ("CLEAR", "FLAG", "BLOCK")
    # If it's blocked, the message must be withheld; if not, it must exist.
    if d["compliance_status"] == "BLOCK":
        assert d["worth_pursuing"] is False
        assert "withheld" in d["drafted_message"].lower()
    else:
        assert d["drafted_message"]


def test_judge_raises_clearly_on_missing_required_field():
    """A missing required key should fail loudly with a KeyError, not
    silently produce a playbook with made-up values."""
    model = RecoveryModel()
    model.fit(to_dicts(generate_dataset(200)))
    judge = Judge(model=model)
    incomplete_txn = _txn()
    del incomplete_txn["failure_reason"]
    try:
        judge.adjudicate(incomplete_txn)
        assert False, "expected a KeyError for missing failure_reason"
    except KeyError:
        pass


def test_judge_withholds_message_when_regulator_blocks():
    model = RecoveryModel()
    model.fit(to_dicts(generate_dataset(200)))
    judge = Judge(model=model)
    playbook = judge.adjudicate(_txn(failure_reason="CARD_DECLINED_RISK", customer_segment="new"))
    # Negotiator would offer cashback to a new user here, which the
    # Regulator's CARD_DECLINED_RISK rule should block outright.
    if playbook.compliance_status == "BLOCK":
        assert playbook.worth_pursuing is False
        assert "withheld" in playbook.drafted_message.lower()


if __name__ == "__main__":
    import inspect
    current_module = sys.modules[__name__]
    test_fns = [f for name, f in inspect.getmembers(current_module, inspect.isfunction) if name.startswith("test_")]
    passed, failed = 0, 0
    for fn in test_fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
