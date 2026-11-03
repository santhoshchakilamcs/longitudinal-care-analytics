"""Longitudinal Care Analytics: FastAPI + HTML dashboard.

SYNTHETIC DATA ONLY. Not a medical device, not a diagnosis, not a
clinical decision tool. No real patients, no EHR data. If model.joblib
or the SQLite DB is missing on first start, the app regenerates
them from train.py (synthetic, seed 42), so a fresh clone works
with no binary files committed.
"""
import io
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from train import FEATURES, build_db, make_patients, train

ROOT = Path(__file__).parent
app = FastAPI(title="Longitudinal Care Analytics (synthetic demo)")


def ensure_model():
    model_path = ROOT / "model.joblib"
    db_path = ROOT / "data" / "patients.db"
    if not model_path.exists() or not db_path.exists():
        df = make_patients()
        model, metrics = train(df)
        joblib.dump(model, model_path)
        (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
        build_db(df)
    return joblib.load(model_path)


MODEL = ensure_model()


def metrics():
    p = ROOT / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else {}


def db():
    return sqlite3.connect(ROOT / "data" / "patients.db")


def care_gaps(row):
    gaps = []
    if row["hba1c"] >= 7.0:
        gaps.append("HbA1c at or above 7.0: follow-up lab due")
    if row["systolic_bp"] >= 140:
        gaps.append("Systolic BP at or above 140: blood pressure follow-up due")
    if row["visits_last_year"] == 0:
        gaps.append("No visits in the last year: care gap, schedule follow-up")
    return gaps


class PredictIn(BaseModel):
    age: float
    bmi: float
    systolic_bp: float
    hba1c: float
    ldl: float
    smoker: int = 0
    family_history: int = 0
    visits_last_year: int = 0


@app.get("/api/health")
def health():
    return {"status": "ok", "data": "synthetic only", "not_a_medical_device": True}


@app.get("/api/metrics")
def get_metrics():
    return metrics()


@app.get("/api/patients")
def patients(limit: int = 20, risk_min: float = 0.0):
    con = db()
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT * FROM patients WHERE risk_prob >= ? ORDER BY risk_prob DESC LIMIT ?",
        (risk_min, limit)).fetchall()
    con.close()
    out = []
    for r in rows:
        d = dict(r)
        d["care_gaps"] = care_gaps(d)
        out.append(d)
    return out


@app.get("/api/patients/{patient_id}")
def patient(patient_id: str):
    con = db()
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM patients WHERE patient_id = ?", (patient_id,)).fetchone()
    visits = con.execute("SELECT * FROM visits WHERE patient_id = ? ORDER BY visit", (patient_id,)).fetchall()
    con.close()
    if not row:
        return JSONResponse({"error": "patient not found"}, status_code=404)
    d = dict(row)
    d["care_gaps"] = care_gaps(d)
    d["visits"] = [dict(v) for v in visits]
    return d


@app.post("/api/predict")
def predict(inp: PredictIn):
    x = pd.DataFrame([[getattr(inp, f) for f in FEATURES]], columns=FEATURES)
    prob = float(MODEL.predict_proba(x)[0, 1])
    tier = "low" if prob < 0.3 else ("medium" if prob < 0.6 else "high")
    return {"risk_prob": round(prob, 4), "tier": tier, "note": "Synthetic demo score only, not a diagnosis"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Bring your own data: upload a CSV with the FEATURES columns
    plus a 'label' column to retrain on YOUR data and see metrics,
    or without a label to get predictions on your rows. Nothing is
    stored permanently; the uploaded file is processed in memory.
    """
    raw = await file.read()
    df = pd.read_csv(io.BytesIO(raw))
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        return JSONResponse({"error": f"missing columns: {missing}"}, status_code=400)
    if "label" in df.columns:
        model, m = train(df)
        return {"trained_on": "uploaded file", "rows": int(len(df)), "metrics": m}
    proba = MODEL.predict_proba(df[FEATURES])[:, 1]
    return {"predicted_on": "uploaded file", "rows": int(len(df)),
            "risk_probs": [round(float(x), 4) for x in proba[:20]]}


@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "frontend" / "index.html").read_text()
