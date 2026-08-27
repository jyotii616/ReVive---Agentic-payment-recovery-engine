# ReVive - Agentic Payment Recovery Engine

**Razorpay AI Builder Internship 2026, Track 3: AI Revenue Recovery**

Every failed payment gets treated the same way at most merchants today: one
generic retry link, sent on a fixed schedule. A bank timeout, an expired OTP,
and an insufficient-funds decline are three different problems. They get the
same email. That gap is what ReVive is built to close.

## What it actually does

ReVive is a swarm of five small agents that sit between "payment failed" and
"recovery attempt sent." Each one answers a narrow question, and each one has
to justify its answer in plain English:

| Agent | Question | 
|---|---|
| The Coroner | Why did this actually fail? Transient, friction, or structural |
| The Psychologist | When should we come back to this customer |
| The Negotiator | Which channel, is an incentive worth it, and what does the message say |
| The Regulator | Does this violate contact-frequency, consent, or risk-hold rules |
| The Judge | Combine everything into one scored, ranked playbook |

I kept these separate instead of training one end-to-end classifier on
purpose. A single model would score higher on a leaderboard metric, but a
merchant ops lead can't ask a black box why a specific customer got a
specific offer. They can ask the Coroner, and get an answer.

## Why this over a churn predictor

Most submissions in this space stop at "predict which payments will fail."
That's the easier half. ReVive assumes the failure already happened, the
harder and more commercially real question, and answers: given a failure,
what's the next action, who should get it, and can I defend that decision if
someone asks.

## What's new in this build

Four things that go past a single demo script:

1. **A compliance agent that can actually block a decision.** The Regulator
   checks every proposed playbook against a contact-frequency cap, a
   goodwill-offer policy for unverified new users, and a rule against
   incentivizing a risk-flagged card decline. In the 500-transaction run
   below, it flagged 152 playbooks and blocked 15 outright before they could
   ship. The Regulator audits a structured `incentive_type` field the
   Negotiator sets ("none" / "cashback" / "emi"), not the human-readable
   label string — an earlier version matched on substrings like
   `"cashback" in label`, which meant an EMI offer on a risk-flagged card
   slipped past the block it should have hit. That's fixed now; see
   `tests/test_agents.py::test_regulator_is_keyed_off_structured_field_not_label_text`
   for the regression test.
2. **Budget-constrained allocation.** A real recovery team can't chase every
   failure. `simulate.py` ranks every playbook by expected value (amount x
   probability x confidence) and only works the top slice that fits a
   35% ops-capacity budget. That worklist recovers ~₹3,219 per attempt,
   versus ~₹1,603 per attempt if every eligible case were worked
   unconstrained — working the highest expected-value cases first is
   roughly double the revenue per attempt of working all of them. (This
   is a ranked-vs-unranked comparison, not a comparison against a
   first-come-first-served queue — `simulate.py` doesn't model arrival
   order at all.)
3. **A message the customer would actually receive.** The Negotiator doesn't
   stop at "channel: WhatsApp." It drafts the send-ready copy by default
   from a deterministic template, so a reviewer can read the literal text
   instead of a label. It can optionally hand that same diagnosis +
   incentive decision to an actual Claude API call instead
   (`draft_message(..., use_llm=True)`, needs `ANTHROPIC_API_KEY`), with the
   template as an automatic fallback if the call fails. The batch
   simulation always uses the template, both for cost and so the results
   below stay exactly reproducible.
4. **A learning loop, not just a snapshot.** `multi_day.py` wraps the
   incentive decision in a small epsilon-greedy bandit and runs it across
   10 simulated days. The system starts by guessing between no-incentive,
   cashback, and EMI conversion, and converges on EMI conversion as the
   highest-value arm by day 3 or 4, without being told the answer in advance.

## Results (simulated, 500 failed transactions, roughly 15 lakh in stuck value)

| | Baseline (generic retry) | ReVive, unconstrained | ReVive, 35% ops budget |
|---|---|---|---|
| Recovery rate | ~21% | ~48% | n/a (fewer attempts, higher hit rate) |
| Revenue recovered | ~3.3L | ~7.3L | ~5.6L on 35% of the attempts |
| Revenue per attempt | ~667 | ~1,603 | ~3,219 |

Every number in this table is read directly from `results.json`'s output
(`baseline.revenue_per_attempt`, `revive_budgeted.revenue_per_attempt` and
`.revenue_per_attempt_unconstrained` — all three fields are computed in
`simulate.py`, not derived by hand), so re-running `python -m
src.revive.simulate` reproduces this table exactly. The "ReVive,
unconstrained" revenue-per-attempt figure is `revive_recovered_amount`
divided by the count of playbooks marked `worth_pursuing`, i.e. revenue per
case actually worked, not per case in the full 500-transaction batch.

**Read this table as an illustration of a mechanism, not a measured result.**
The gap between baseline and ReVive isn't something the simulation
discovered — `simulate.py` hard-codes baseline to realize ~40% of a
synthetic "true recoverability" signal and ReVive to realize 55-90% of it
(scaled by the swarm's own confidence), on the stated assumption that a
tailored playbook lands closer to a customer's actual recoverability than a
single generic retry does. The table shows what follows *if* that
assumption holds, not proof that it holds. What is real: the trained
`LogisticRegression`, the Regulator's rule checks, the budget-ranking logic,
and the bandit's convergence are all genuinely computed, not scripted to hit
a target number — only the two realization-fraction constants above are
authored assumptions rather than fitted or measured values. See
`src/revive/simulate.py` for exactly where those constants live.

These numbers come from a synthetic hidden "true recoverability" signal, not
real Razorpay traffic (see `src/revive/data_generator.py` for exactly how
it's constructed). Open `dashboard.html` for the full breakdown, including
which playbooks the Regulator flagged and why.

## Architecture

```
ReVive/
|-- src/revive/
|   |-- data_generator.py   synthetic failed-payment events, no real data
|   |-- agents.py           Coroner, Psychologist, Negotiator, Regulator
|   |-- orchestrator.py     Judge: trained LogisticRegression + reconciliation
|   |-- simulate.py         baseline vs ReVive, plus budget-constrained allocation
|   `-- multi_day.py        epsilon-greedy bandit, 10-day learning curve
|-- tests/test_agents.py    17 tests on the actual decision logic
|-- revive_cli.py           terminal tool: get a playbook for one transaction
|-- app.py                  FastAPI: /simulate, /playbook, serves dashboard.html
|-- dashboard.html          standalone interactive results dashboard
|-- results.json            last simulation output
|-- learning_curve.json     last multi_day.py output
`-- requirements.txt
```

## Running it

```bash
pip install -r requirements.txt

# Run the tests
python tests/test_agents.py

# Re-run the main simulation (regenerates results.json)
python -m src.revive.simulate

# Run the 10-day learning simulation
python -m src.revive.multi_day

# Get a playbook for one failed payment from the terminal
python revive_cli.py --amount 4500 --reason INSUFFICIENT_FUNDS --hour 23 --segment new
python revive_cli.py --demo

# Or launch the live API and dashboard
uvicorn app:app --reload
# http://127.0.0.1:8000/       dashboard
# http://127.0.0.1:8000/docs   try /playbook with your own transaction
```

`dashboard.html` also works standalone. Open the file directly and it still
renders, because the last simulation's results are embedded in it. No server
required.

## What's real and what's simulated, stated plainly

The agent logic, the trained `LogisticRegression` recovery model, the
Regulator's rule checks, the bandit, the FastAPI service, the CLI, and the
dashboard are all real and runnable. The transaction data and customer
response behavior are synthetic, generated to demonstrate the mechanism
end to end without needing access to real merchant data. Swapping
`data_generator.py` for a live webhook feed is the obvious next step, and
the rest of the pipeline does not need to change to support that.

**What the trained model's accuracy does and doesn't demonstrate.**
`RecoveryModel` is trained on binary outcomes sampled from
`_true_recoverability()` (see `data_generator.py`), using features
(diagnosis class, segment, amount, tenure, retry count) that are the same
inputs that formula is built from. So the model recovering that pattern is
evidence it can learn a known synthetic function from noisy binary draws of
itself — a real but narrow ML exercise — not evidence it would generalize
to whatever actually drives recoverability for real customers, which this
project has no data on. The "genuine prediction, not a lookup table" line
in `orchestrator.py`'s docstring is true in a narrow sense (the model isn't
literally reading `_true_recoverability` off a table, it has to infer it
from noisy 0/1 labels), but it shouldn't be read as a claim about
real-world generalization. Retraining on actual outcome data is the
first roadmap item below for exactly this reason.

## Roadmap

- Replace the synthetic response model with real outcomes from an A/B test
- Extend the Regulator's rule set to cover recurring-mandate consent, not
  just one-off retries
- Move the bandit from a single incentive decision to jointly optimizing
  channel, timing, and incentive together


AUTHOR 
JYOTI KUMARI
INDIRA GANDHI DELHI TECHNICAL UNIVERSITY FOR WOMEN
02601192025
