# Longitudinal Care Analytics (Personal Project)

## About

FastAPI app for chronic-condition risk scoring, longitudinal patient
tracking, and care-gap detection on synthetic data: 5,000 synthetic
patients, balanced logistic risk model, physician dashboard, and
bring-your-own-data upload. Synthetic only, not a medical device.

A FastAPI app that demonstrates chronic-condition risk scoring,
longitudinal patient tracking, and care-gap detection, built to
cover the healthcare domain gap in the Counterpart Health Senior
Data Scientist role: early diagnosis support, longitudinal care
management, and data products for physicians.

Not affiliated with Counterpart Health or Clover Health. No real
patients, no EHR data, no clinical data. All 5,000 patients are
synthetic, generated with seed 42. This is NOT a medical device,
NOT a diagnosis, and NOT a clinical decision tool. Do not use it
for real patient care.

## What it does

- Risk model: logistic regression (class-balanced) on age, BMI,
  systolic BP, HbA1c, LDL, smoking, family history, and visits
  in the last year, predicting a synthetic chronic-condition
  label generated from a transparent logistic function
- Longitudinal tracking: 4 synthetic visits per patient, with
  risk, BP, and HbA1c drifting over time, viewable per patient
- Care gaps: HbA1c at or above 7.0, systolic BP at or above 140,
  and no visits in the last year, flagged per patient
- Physician-facing dashboard: model metrics, a risk-check form,
  highest-risk synthetic patients, and care gaps
- Bring your own data: `POST /api/upload` takes a CSV with the
  feature columns, plus an optional label column. With a label it
  retrains on YOUR data and returns metrics; without a label it
  returns predictions. Uploaded data is processed in memory and
  is not stored permanently. Synthetic sample results are never
  presented as a user's results.

## Verified results (local run, Nov 3, 2026)

- `verify.py`: PASS, including a risk-ordering check (high-risk
  example 0.9465, low-risk example 0.0298)
- Test set: 1,000 patients, prevalence 0.0348
- ROC-AUC 0.8137, accuracy 0.688, precision 0.0765,
  recall 0.7143, F1 0.1381
- Honest reading: the synthetic outcome is rare (3.5%), so
  precision is low even with good recall. This is a screening
  demo that flags patients for follow-up, not a diagnosis.
  An unweighted model was tried first and predicted no
  positives at all; the balanced logistic model above replaced
  it, and that change is recorded here, not hidden.
- Live server: `/api/health` ok, `/api/metrics` matches above,
  `/api/predict` for a high-risk profile returned 0.9465 high
  tier, `/api/patients` returned ranked patients with care gaps,
  frontend home page HTTP 200

## Run it

```
pip install -r requirements.txt
python verify.py
python train.py
uvicorn app:app --host 0.0.0.0 --port 8000
```

If `model.joblib` or `data/patients.db` is missing on first
start, the app regenerates them from `train.py` (synthetic,
seed 42), so a fresh clone works with no binary files committed.
`model.joblib` and `patients.db` are therefore NOT committed.

## API

- `GET /api/health`, `GET /api/metrics`
- `GET /api/patients?limit=20&risk_min=0.0`
- `GET /api/patients/{patient_id}` (with visits and care gaps)
- `POST /api/predict` (feature JSON, returns risk and tier)
- `POST /api/upload` (CSV, retrain or predict)

## Files

- `app.py`: FastAPI server, SQLite, BYOD upload
- `train.py`: synthetic generation, model training, SQLite build
- `verify.py`: end-to-end verification
- `frontend/index.html`: dashboard, no build step
- `metrics.json`: metrics from the last verified run
