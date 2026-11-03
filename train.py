"""Train the chronic-condition risk model on SYNTHETIC data only.

No real patients, no EHR data, no clinical data. Every patient is
generated here with seed 42 from plausible ranges, and the label is
a synthetic probability function, not a diagnosis. This model is a
portfolio demo, not a medical device, not a clinical decision tool,
and not a diagnosis. See README.md.
"""
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
FEATURES = ["age", "bmi", "systolic_bp", "hba1c", "ldl", "smoker", "family_history", "visits_last_year"]


def make_patients(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 85, n)
    bmi = np.round(rng.normal(29, 6, n), 1)
    systolic = rng.integers(100, 180, n)
    hba1c = np.round(rng.normal(5.8, 1.2, n), 1)
    ldl = rng.integers(70, 210, n)
    smoker = rng.integers(0, 2, n)
    family = rng.integers(0, 2, n)
    visits = rng.integers(0, 13, n)
    # Synthetic risk score: a transparent logistic function, NOT a
    # clinical rule. Coefficients are invented for the demo only.
    logit = (
        -6.0
        + 0.03 * (age - 30)
        + 0.05 * (bmi - 25)
        + 0.02 * (systolic - 120)
        + 0.6 * (hba1c - 5.5)
        + 0.008 * (ldl - 100)
        + 0.7 * smoker
        + 0.5 * family
        - 0.05 * visits
    )
    prob = 1 / (1 + np.exp(-logit))
    label = (rng.random(n) < prob).astype(int)
    df = pd.DataFrame({
        "patient_id": [f"SYN-{i:05d}" for i in range(n)],
        "age": age, "bmi": bmi, "systolic_bp": systolic, "hba1c": hba1c,
        "ldl": ldl, "smoker": smoker, "family_history": family,
        "visits_last_year": visits, "risk_prob": np.round(prob, 4),
        "label": label,
    })
    return df


def train(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df["label"], test_size=0.2, random_state=42, stratify=df["label"])
    # Logistic regression with class balancing: the synthetic label is
    # rare (about 3.5%), and an unweighted XGBoost predicted no
    # positives at all in testing. Logistic + balanced weights is
    # also the interpretable choice for a risk demo.
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "n_patients": int(len(df)),
        "n_test": int(len(y_test)),
        "prevalence": round(float(df["label"].mean()), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
    }
    return model, metrics


def build_db(df):
    db = ROOT / "data" / "patients.db"
    db.parent.mkdir(exist_ok=True)
    if db.exists():
        db.unlink()
    con = sqlite3.connect(db)
    df.to_sql("patients", con, index=False)
    # Longitudinal visits: 4 visits per patient, risk drifting with
    # HbA1c and BP, all synthetic.
    rng = np.random.default_rng(7)
    rows = []
    for _, p in df.iterrows():
        for v in range(4):
            rows.append({
                "patient_id": p["patient_id"],
                "visit": v + 1,
                "systolic_bp": int(p["systolic_bp"] + rng.integers(-8, 9)),
                "hba1c": round(float(p["hba1c"] + rng.normal(0, 0.2)), 1),
                "risk_prob": round(float(min(0.99, max(0.01, p["risk_prob"] + rng.normal(0, 0.03)))), 4),
            })
    pd.DataFrame(rows).to_sql("visits", con, index=False)
    con.close()
    return db


if __name__ == "__main__":
    df = make_patients()
    model, metrics = train(df)
    (ROOT / "data").mkdir(exist_ok=True)
    df.to_csv(ROOT / "data" / "patients.csv", index=False)
    import joblib
    joblib.dump(model, ROOT / "model.joblib")
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    build_db(df)
    print(json.dumps(metrics, indent=2))
