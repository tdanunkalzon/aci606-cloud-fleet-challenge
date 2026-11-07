# ACI606 - Cloud Fleet Challenge

Fleet platform for **Exercise 4** of ACI606 Cloud Computing.

## Repository name

Use:

`aci606-cloud-fleet-challenge`

## The idea in one sentence

A virtual fleet sends telemetry to a remote platform. Your group must transform the starter into a **persistent Cloud service** that stores the data, trains and serves an ML model, exposes metrics, and presents the fleet in a dashboard.

## Components

- FastAPI for ingestion, current telemetry, complete per-robot history and model inference.
- PostgreSQL for hosted persistence and SQLite for local development.
- Dashboard and operational metrics at `/dashboard` and `/metrics`.
- Fleet simulator, reproducible model training and an HTTP smoke test.

## Quick start

### A. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In Codespaces you can normally use the existing Python environment instead of creating another one.

### B. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- `/docs`
- `/dashboard`

### C. Start the virtual fleet

In another terminal:

```bash
python simulator/fleet_simulator.py --api http://127.0.0.1:8000 --robots 5
```

The dashboard should begin to show robots.

### D. Generate training data

```bash
python simulator/generate_dataset.py --fleets 16 --samples 500 --robots 8 --seed 606
python ml/train_model.py
```

The training script splits by fleet, selects the classifier on separate
validation fleets and reports macro F1 and a confusion matrix on held-out fleets.
See [model evidence](evidencia_modelo.txt) for the results. Logistic regression
with standardized inputs was selected because it had the highest validation
macro F1 (0.9422 versus 0.9394 for Random Forest). The pipeline is tracked in
`models/model.joblib` and its scikit-learn version is pinned in `requirements.txt`.

## Deploy on Render with Neon PostgreSQL

1. Create a Neon PostgreSQL database and copy its connection string with TLS.
2. Push the repository to GitHub. In Render, create a Blueprint from this
   repository's `main` branch. The checked-in `render.yaml` installs the
   dependencies, starts Uvicorn using the assigned `PORT` and checks `/health`.
3. When Render prompts for `DATABASE_URL`, supply the Neon connection string
   with `sslmode=require`. Keep it in Render's environment settings. The API
   creates the telemetry table on startup and stores the full history in Neon.
   Local runs without `DATABASE_URL` use the ignored `fleet.db` SQLite file.
4. Record the assigned public HTTPS URL here after deployment. Check `/health`
   first, then run the smoke test and start the fleet against that URL.

```bash
python tests/smoke_test.py --api "$FLEET_API_URL" --final
python simulator/fleet_simulator.py --api "$FLEET_API_URL" --robots 5
```

5. Once one robot has more than 100 samples, stop the simulator and save its
   history response. Restart the web service from Render and compare the same
   history endpoint against the saved response. Review Render logs and
   `/metrics`, then open `/dashboard` from another device.

`/metrics` reports per-process counts of telemetry and prediction requests,
HTTP errors (4xx and 5xx), average elapsed time for prediction requests in
milliseconds, and uptime in seconds. These counters reset on restart.


## Final challenge

The instructor may evaluate your service using an unseen fleet scenario. The assessment can consider functionality, prediction quality, latency, robustness and the technical defence of the architecture.

## Important

The robot simulator is **not** the deliverable. It is a client/load generator.

The deliverable is the **Cloud platform** your group builds around it.
