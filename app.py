"""
ReVive :: app.py
Thin FastAPI layer over the agent swarm, for plugging into a real dashboard
or webhook pipeline instead of only running as a batch script.

Run: uvicorn app:app --reload, then open http://127.0.0.1:8000/docs

Endpoints:
    GET  /                -> serves dashboard.html
    GET  /simulate?n=500  -> runs the baseline-vs-ReVive simulation
    POST /playbook        -> live Recovery Playbook for one transaction
"""

from __future__ import annotations
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.revive.data_generator import generate_dataset, to_dicts
from src.revive.orchestrator import Judge, RecoveryModel
from src.revive.simulate import run as run_simulation

app = FastAPI(title="ReVive, Agentic Payment Recovery Engine", version="1.0.0")

# Fit once at startup on a fresh synthetic dataset, reused across requests.
_model = RecoveryModel()
_model.fit(to_dicts(generate_dataset(500)))
_judge = Judge(model=_model)


class TransactionIn(BaseModel):
    txn_id: str = "txn_manual_0001"
    customer_id: str = "cust_demo"
    amount: float = Field(..., gt=0)
    instrument: str = "UPI"
    failure_reason: str
    retry_count: int = 0
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = 0
    customer_segment: str = "returning"
    customer_tenure_days: int = 180
    city: str = "Gurugram"
    timestamp: str = "2026-08-27T01:12:00"


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    html_path = Path(__file__).parent / "dashboard.html"
    return html_path.read_text()


@app.get("/simulate")
def simulate(n: int = 500) -> JSONResponse:
    return JSONResponse(run_simulation(n))


@app.post("/playbook")
def playbook(txn: TransactionIn) -> dict:
    playbook = _judge.adjudicate(txn.model_dump())
    return playbook.to_dict()
