from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import os
from functools import lru_cache
from time import perf_counter

import joblib
import pandas as pd

app = FastAPI(
    title="ACI606 Cloud Fleet Challenge",
    version="0.1.0-starter",
    description="Starter API for the ACI606 Cloud Fleet Challenge.",
)

Status = Literal["NORMAL", "WARNING", "CRITICAL"]


class Telemetry(BaseModel):
    robot_id: str = Field(min_length=1, max_length=20)
    timestamp: float
    x: float
    y: float
    speed: float = Field(ge=0)
    battery: float = Field(ge=0, le=100)
    motor_temp: float
    vibration: float = Field(ge=0)
    motor_current: float = Field(ge=0)
    obstacle_distance: float = Field(ge=0)


class PredictionRequest(Telemetry):
    pass


# Starter storage: intentionally volatile.
# Students must replace this with persistent storage for the final challenge.
LATEST: dict[str, Telemetry] = {}
HISTORY: dict[str, deque[Telemetry]] = defaultdict(lambda: deque(maxlen=100))
REQUEST_COUNT = 0


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "robots_in_memory": len(LATEST),
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry) -> dict:
    global REQUEST_COUNT
    REQUEST_COUNT += 1
    LATEST[sample.robot_id] = sample
    HISTORY[sample.robot_id].append(sample)
    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "note": "Starter mode: data is stored only in RAM.",
    }


@app.get("/robots")
def list_robots() -> list[dict]:
    return [sample.model_dump() for sample in LATEST.values()]


@app.get("/robots/{robot_id}/history")
def robot_history(robot_id: str) -> list[dict]:
    if robot_id not in HISTORY:
        raise HTTPException(status_code=404, detail="Robot not found")
    return [sample.model_dump() for sample in HISTORY[robot_id]]

MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        str(Path(__file__).resolve().parents[1] / "models" / "model.joblib"),
    )
)

PREDICTION_FEATURES = [
    "speed",
    "battery",
    "motor_temp",
    "vibration",
    "motor_current",
    "obstacle_distance",
]


@lru_cache(maxsize=1)
def load_prediction_model():
    return joblib.load(MODEL_PATH)

@app.post("/predict")
def predict(sample: PredictionRequest) -> dict:
    model = load_prediction_model()
    started = perf_counter()

    features = pd.DataFrame(
        [sample.model_dump()],
        columns=PREDICTION_FEATURES,
    )
    probabilities = model.predict_proba(features)[0]
    best_index = int(probabilities.argmax())

    return {
        "robot_id": sample.robot_id,
        "predicted_status": str(model.classes_[best_index]),
        "confidence": float(probabilities[best_index]),
        "model_version": "v1",
        "inference_ms": round((perf_counter() - started) * 1000, 2),
    }


@app.get("/metrics")
def metrics() -> dict:
    # TODO E4: extend with latency, errors, throughput and model metrics.
    return {
        "requests_received": REQUEST_COUNT,
        "robots_in_memory": len(LATEST),
        "persistence": "RAM_ONLY",
        "prediction_service": "NOT_IMPLEMENTED",
    }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")
