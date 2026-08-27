"""
ReVive :: agents.py
Five single-responsibility agents: Coroner (why it failed), Psychologist
(when to retry), Negotiator (channel/incentive + message), Regulator
(compliance check), Judge (combines all into one Recovery Playbook).
Each exposes .decide(txn: dict) -> AgentOpinion (decision, confidence, reasoning).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class AgentOpinion:
    agent: str
    decision: str
    confidence: float          # 0..1
    reasoning: str


# Structured contract the Regulator audits. Never infer this from the
# Negotiator's human-readable `label` string — substring-matching on it
# is what let a copy change silently disable a compliance check before.
IncentiveType = Literal["none", "cashback", "emi"]


@dataclass
class NegotiatorDecision:
    """Structured output of Negotiator.decide(). `label` is the human-readable
    description; `incentive_type`/`incentive_pct` are what compliance keys off."""
    label: str
    incentive_type: IncentiveType
    confidence: float
    reasoning: str
    incentive_pct: float = 0.0  # e.g. 0.02 for a 2% cashback; 0.0 for none/EMI


# 1. THE CORONER, failure diagnosis
class Coroner:
    """Classifies why the payment failed: TRANSIENT (infra hiccup, retry now),
    FRICTION (user error, needs easier flow), or STRUCTURAL (money/limit/risk,
    needs incentive or delay)."""

    TRANSIENT = {"BANK_SERVER_TIMEOUT", "NETWORK_DROP", "ISSUER_DOWN"}
    FRICTION = {"OTP_EXPIRED", "UPI_PIN_MISMATCH"}
    STRUCTURAL = {"INSUFFICIENT_FUNDS", "CARD_DECLINED_RISK", "LIMIT_EXCEEDED"}

    def decide(self, txn: dict[str, Any]) -> AgentOpinion:
        reason = txn["failure_reason"]
        if reason in self.TRANSIENT:
            cls, conf = "TRANSIENT", 0.9
            why = f"{reason} is an infrastructure-side fault, not a customer decision, the payment almost certainly would have gone through."
        elif reason in self.FRICTION:
            cls, conf = "FRICTION", 0.75
            why = f"{reason} indicates the customer *intended* to pay but the flow broke mid-way, recoverable with an easier retry path."
        else:
            cls, conf = "STRUCTURAL", 0.6
            why = f"{reason} suggests a real constraint (funds/limit/risk), recovery needs incentive or timing, not just a retry link."
        return AgentOpinion("Coroner", cls, conf, why)


# 2. THE PSYCHOLOGIST, timing
class Psychologist:
    """Chooses when to re-approach based on payday cycles, hour-of-day, and
    retry fatigue."""

    def decide(self, txn: dict[str, Any]) -> AgentOpinion:
        hour = txn["hour_of_day"]
        reason = txn["failure_reason"]
        retry_count = txn["retry_count"]

        if reason == "INSUFFICIENT_FUNDS":
            window = "next likely salary window (1st or 7th of month, 09:00-11:00 IST)"
            conf = 0.7
            why = "Funds-based failures rarely resolve same-day; timing the nudge to a payday window roughly doubles response rates."
        elif retry_count >= 2:
            window = "delay 48h, single consolidated nudge"
            conf = 0.55
            why = f"Customer has already failed {retry_count} times, immediate re-approach reads as spam and suppresses response; back off."
        elif 0 <= hour <= 6:
            window = "next morning, 09:00-10:30 IST"
            conf = 0.8
            why = "Late-night failure, same-day retry has near-zero response; mornings show highest UPI completion rates."
        else:
            window = "within 15 minutes"
            conf = 0.85
            why = "Fresh, daytime, low-retry failure, intent is still 'hot'; immediate retry captures it before attention drifts."
        return AgentOpinion("Psychologist", window, conf, why)


# 3. THE NEGOTIATOR, channel + incentive
class Negotiator:
    """Chooses channel and whether an incentive is worth offering. Returns a
    NegotiatorDecision: a display label plus a structured `incentive_type`
    the Regulator/Judge key off — never inferred from the label string."""

    def decide(self, txn: dict[str, Any]) -> NegotiatorDecision:
        amount = txn["amount"]
        segment = txn["customer_segment"]
        reason = txn["failure_reason"]

        if segment in ("loyal", "high_value") and amount > 8000:
            channel = "WhatsApp (personal) + EMI-conversion offer"
            conf = 0.75
            why = f"High-value {segment} customer at ₹{amount:,.0f}, an EMI option removes the real objection instead of just retrying the same failed flow."
            incentive: IncentiveType = "emi"
        elif reason == "INSUFFICIENT_FUNDS" and amount < 3000:
            channel = "SMS, no incentive"
            conf = 0.6
            why = "Small-ticket funds failure, an incentive costs more margin than the extra recovery it buys; a plain reminder suffices."
            incentive = "none"
        elif segment == "new":
            channel = "In-app nudge + small cashback (2%)"
            conf = 0.65
            why = "First-time customer, a small goodwill cashback offsets the friction of a bad first impression and protects lifetime value."
            incentive = "cashback"
            return NegotiatorDecision(label=channel, incentive_type=incentive, confidence=conf, reasoning=why, incentive_pct=0.02)
        else:
            channel = "WhatsApp, no incentive"
            conf = 0.7
            why = "Returning customer, moderate ticket. A high-open-rate channel is enough here without eroding margin."
            incentive = "none"
        return NegotiatorDecision(label=channel, incentive_type=incentive, confidence=conf, reasoning=why)

    def _template_message(self, txn: dict[str, Any], incentive_type: IncentiveType, customer_name: str) -> str:
        """Deterministic fallback when no LLM is used; also what simulate.py/tests run."""
        amount = txn["amount"]
        reason = txn["failure_reason"]

        if incentive_type == "emi":
            return (
                f"Hi {customer_name}, your payment of ₹{amount:,.0f} didn't go through. "
                f"You can now split it into easy monthly instalments, no extra charges for the first conversion. "
                f"Tap to complete: {{recovery_link}}"
            )
        if incentive_type == "cashback":
            return (
                f"Hi {customer_name}, looks like your ₹{amount:,.0f} payment got stuck. "
                f"Finish it in the next hour and we'll add a small thank-you credit to your account. "
                f"{{recovery_link}}"
            )
        if reason in ("BANK_SERVER_TIMEOUT", "NETWORK_DROP", "ISSUER_DOWN"):
            return f"Hi {customer_name}, that last payment attempt dropped on our end, not yours. Retry here: {{recovery_link}}"
        if reason in ("OTP_EXPIRED", "UPI_PIN_MISMATCH"):
            return f"Hi {customer_name}, your OTP window closed before the payment finished. One more try: {{recovery_link}}"
        return f"Hi {customer_name}, your ₹{amount:,.0f} payment didn't complete. Pick up where you left off: {{recovery_link}}"

    def draft_message(
        self,
        txn: dict[str, Any],
        decision: NegotiatorDecision,
        customer_name: str = "there",
        use_llm: bool = False,
    ) -> str:
        """Turns the channel/incentive decision into send-ready message text.
        Defaults to the deterministic template (what backs simulate.py and
        the tests). Pass use_llm=True (+ ANTHROPIC_API_KEY) to have a model
        write it instead; falls back to the template on any failure."""
        template = self._template_message(txn, decision.incentive_type, customer_name)
        if not use_llm:
            return template

        try:
            import anthropic

            client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
            prompt = (
                "Write one short customer message (2-3 sentences, max ~40 words) for a failed "
                "payment recovery nudge. Do not invent facts beyond what's given.\n\n"
                f"Customer name: {customer_name}\n"
                f"Amount: ₹{txn['amount']:,.0f}\n"
                f"Failure reason (internal code, don't quote verbatim): {txn['failure_reason']}\n"
                f"Approach: {decision.label}\n"
                f"Incentive type: {decision.incentive_type}\n"
                "End the message with the literal placeholder {recovery_link} for the payment link.\n"
                "Tone: warm, brief, no guilt-tripping, no exclamation marks. "
                "Return only the message text, nothing else."
            )
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip()
            return text or template
        except Exception:
            # Any failure (network, missing key, rate limit) degrades to the template.
            return template


# 4. THE REGULATOR, compliance gate
# Violation severity must key off `severity`, never the wording of `message` -
# an earlier version checked `"Escalate" in violation_text`, which a message
# rename could silently break. See
# tests/test_agents.py::test_regulator_block_severity_survives_message_rewording.
@dataclass
class ComplianceViolation:
    message: str
    severity: Literal["flag", "block"]


class Regulator:
    """Audits the other three agents' output against RBI/TRAI-style rules on
    contact frequency, consent, and risk holds, and clears, flags, or blocks it."""

    # Conservative soft cap below typical complaint thresholds; a policy
    # choice, not derived from data here.
    MAX_CONTACTS_PER_WEEK = 3
    MAX_INCENTIVE_PCT_FOR_NEW_USERS = 0.05  # cap goodwill offers to first-time users

    def review(self, txn: dict[str, Any], decision: NegotiatorDecision, retry_count: int) -> AgentOpinion:
        """Reads `decision.incentive_type` (structured), never `decision.label`."""
        violations: list[ComplianceViolation] = []

        if retry_count >= self.MAX_CONTACTS_PER_WEEK:
            violations.append(ComplianceViolation(
                message=(
                    f"retry_count={retry_count} exceeds the {self.MAX_CONTACTS_PER_WEEK}-contact soft cap; "
                    f"further unsolicited nudges risk a DND/TRAI complaint."
                ),
                severity="flag",
            ))

        if txn["customer_segment"] == "new" and decision.incentive_type == "cashback":
            if decision.incentive_pct > self.MAX_INCENTIVE_PCT_FOR_NEW_USERS:
                violations.append(ComplianceViolation(
                    message=(
                        f"Proposed incentive of {decision.incentive_pct:.0%} for a first-time customer exceeds the "
                        f"{self.MAX_INCENTIVE_PCT_FOR_NEW_USERS:.0%} goodwill-offer cap. Needs manual approval "
                        f"rather than sending as-is."
                    ),
                    severity="block",
                ))
            else:
                violations.append(ComplianceViolation(
                    message=(
                        "First-time customer being offered an incentive before their identity/KYC has fully "
                        "matured; hold for manual review per goodwill-offer policy."
                    ),
                    severity="flag",
                ))

        if txn["failure_reason"] == "CARD_DECLINED_RISK" and decision.incentive_type != "none":
            violations.append(ComplianceViolation(
                message=(
                    "CARD_DECLINED_RISK indicates a possible fraud/risk hold, incentivising a retry on a "
                    "risk-flagged card bypasses the point of the block. Route to the risk team instead of "
                    "recovering."
                ),
                severity="block",
            ))

        if violations:
            decision_level = "BLOCK" if any(v.severity == "block" for v in violations) else "FLAG"
            return AgentOpinion(
                "Regulator",
                decision_level,
                0.9,
                " ".join(v.message for v in violations),
            )
        return AgentOpinion("Regulator", "CLEAR", 0.95, "No consent, contact-frequency, or risk-hold conflicts found.")
