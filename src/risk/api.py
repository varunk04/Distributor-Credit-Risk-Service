import csv
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from risk.train import CAT, NUM

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.joblib"
META_PATH = ARTIFACTS / "model_meta.json"
FEATURES_PATH = ARTIFACTS / "features.csv"
FEEDBACK_PATH = ARTIFACTS / "feedback.csv"

state = {"model": None, "threshold": None, "features": None, "error": None}


def load_state():
    state["model"] = None
    state["features"] = None
    state["threshold"] = None
    state["error"] = None
    try:
        state["model"] = joblib.load(MODEL_PATH)
        meta = json.loads(META_PATH.read_text())
        state["threshold"] = meta["held_out"]["threshold"]
        features = pd.read_csv(FEATURES_PATH)
        for col in ("zero_credit_limit", "no_history"):
            features[col] = features[col].astype(int)
        state["features"] = features.set_index("distributor_id")
        log.info("loaded model and %d distributors", len(features))
    except FileNotFoundError as exc:
        state["error"] = f"missing {Path(exc.filename).name}"
        log.warning("not ready: %s", state["error"])


@asynccontextmanager
async def lifespan(app):
    load_state()
    yield


app = FastAPI(title="Distributor credit risk", lifespan=lifespan)


@app.exception_handler(Exception)
async def unexpected(request, exc):
    log.exception("request failed")
    return JSONResponse(status_code=500, content={"detail": "Something went wrong"})


def require_ready():
    if state["model"] is None or state["features"] is None:
        raise HTTPException(status_code=503, detail=state["error"] or "model is not available")


def probability(row):
    frame = row[NUM + CAT].to_frame().T
    frame["zero_credit_limit"] = frame["zero_credit_limit"].astype(int)
    frame["no_history"] = frame["no_history"].astype(int)
    return float(state["model"].predict_proba(frame)[0, 1])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    require_ready()
    return {"status": "ready", "distributors": len(state["features"])}


@app.get("/distributors/{distributor_id}/risk")
def distributor_risk(distributor_id: str):
    require_ready()
    features = state["features"]
    if distributor_id not in features.index:
        raise HTTPException(status_code=404, detail=f"Unknown distributor {distributor_id}")

    row = features.loc[distributor_id]
    if int(row["no_history"]):
        return {
            "distributor_id": distributor_id,
            "probability": None,
            "high_risk": None,
            "threshold": state["threshold"],
            "no_history": True,
            "detail": "No orders or payments on file, so there is no score",
        }

    proba = probability(row)
    return {
        "distributor_id": distributor_id,
        "probability": round(proba, 3),
        "high_risk": proba >= state["threshold"],
        "threshold": state["threshold"],
        "no_history": False,
    }


@app.get("/portfolio/summary")
def portfolio():
    require_ready()
    features = state["features"]
    no_history = features[features["no_history"] == 1]
    scored = features[features["no_history"] == 0]
    proba = state["model"].predict_proba(scored[NUM + CAT])[:, 1]
    high = proba >= state["threshold"]

    by_region = []
    scored = scored.copy()
    scored["high_risk"] = high
    for region, group in scored.groupby("region"):
        by_region.append({
            "region": region,
            "distributors": int(len(group)),
            "high_risk": int(group["high_risk"].sum()),
        })

    return {
        "distributors": int(len(features)),
        "scored": int(len(scored)),
        "high_risk": int(high.sum()),
        "low_risk": int((~high).sum()),
        "no_history": int(len(no_history)),
        "threshold": state["threshold"],
        "by_region": by_region,
    }


class Feedback(BaseModel):
    distributor_id: str
    decision: Literal["approved", "rejected"]
    note: str | None = None


@app.post("/feedback", status_code=201)
def feedback(body: Feedback):
    require_ready()
    if body.distributor_id not in state["features"].index:
        raise HTTPException(status_code=404, detail=f"Unknown distributor {body.distributor_id}")

    ARTIFACTS.mkdir(exist_ok=True)
    new_file = not FEEDBACK_PATH.exists()
    with FEEDBACK_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["recorded_at", "distributor_id", "decision", "note"])
        if new_file:
            writer.writeheader()
        writer.writerow({
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "distributor_id": body.distributor_id,
            "decision": body.decision,
            "note": body.note or "",
        })
    return {"saved": True, "distributor_id": body.distributor_id, "decision": body.decision}
