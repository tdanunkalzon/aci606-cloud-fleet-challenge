from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

import os
from functools import lru_cache
from threading import Lock
from time import perf_counter

import joblib
import pandas as pd

from . import db


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    load_prediction_model()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="ACI606 Cloud Fleet Challenge",
    version="0.1.0-starter",
    description="Starter API for the ACI606 Cloud Fleet Challenge.",
)

Status = Literal["NORMAL", "WARNING", "CRITICAL"]


class Telemetry(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

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


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, exc: RequestValidationError):
    # The original input can contain NaN or Infinity, which JSON cannot encode.
    detail = [
        {"loc": error["loc"], "msg": error["msg"], "type": error["type"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": detail})


STARTED = perf_counter()
METRICS_LOCK = Lock()
TELEMETRY_REQUESTS = 0
PREDICTION_REQUESTS = 0
ERRORS = 0
PREDICTION_TIME_MS = 0.0


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    global TELEMETRY_REQUESTS, PREDICTION_REQUESTS, ERRORS, PREDICTION_TIME_MS
    telemetry_request = request.method == "POST" and request.url.path == "/telemetry"
    prediction_request = request.method == "POST" and request.url.path == "/predict"
    started = perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        elapsed_ms = (perf_counter() - started) * 1000
        with METRICS_LOCK:
            TELEMETRY_REQUESTS += int(telemetry_request)
            PREDICTION_REQUESTS += int(prediction_request)
            ERRORS += int(status >= 400)
            if prediction_request:
                PREDICTION_TIME_MS += elapsed_ms


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "aci606-cloud-fleet",
        "version": app.version,
        "robots": db.robot_count(),
        "utc": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/telemetry")
def ingest_telemetry(sample: Telemetry) -> dict:
    db.store_sample(sample.model_dump())
    return {
        "accepted": True,
        "robot_id": sample.robot_id,
        "server_time": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/robots")
def list_robots() -> list[dict]:
    return db.latest_samples()


@app.get("/robots/{robot_id}/history")
def robot_history(robot_id: str) -> list[dict]:
    samples = db.history(robot_id)
    if not samples:
        raise HTTPException(status_code=404, detail="Robot not found")
    return samples

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
    with METRICS_LOCK:
        return {
            "requests_received": TELEMETRY_REQUESTS,
            "telemetry_requests": TELEMETRY_REQUESTS,
            "prediction_requests": PREDICTION_REQUESTS,
            "errors": ERRORS,
            "average_prediction_latency_ms": (
                round(PREDICTION_TIME_MS / PREDICTION_REQUESTS, 2)
                if PREDICTION_REQUESTS else 0.0
            ),
            "uptime_seconds": round(perf_counter() - STARTED, 2),
        }


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard() -> str:
    path = Path(__file__).resolve().parents[1] / "dashboard" / "index.html"
    return path.read_text(encoding="utf-8")
