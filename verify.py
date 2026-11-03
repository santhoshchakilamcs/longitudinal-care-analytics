"""End-to-end verification: train, then check API logic offline.

Prints real metrics from the actual run. Do not copy metrics into
the README until this script has printed them.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train import FEATURES, make_patients, train  # noqa: E402

df = make_patients()
model, metrics = train(df)
print(json.dumps(metrics, indent=2))
assert metrics["roc_auc"] > 0.7, "ROC-AUC unexpectedly low for synthetic data"
assert metrics["n_patients"] == 5000
# Sanity: higher HbA1c patient must score higher than a low-risk one.
import pandas as pd
hi = pd.DataFrame([[65, 34, 160, 8.5, 190, 1, 1, 1]], columns=FEATURES)
lo = pd.DataFrame([[35, 22, 110, 5.0, 90, 0, 0, 6]], columns=FEATURES)
phi = float(model.predict_proba(hi)[0, 1])
plo = float(model.predict_proba(lo)[0, 1])
print(f"high-risk example: {phi:.4f}, low-risk example: {plo:.4f}")
assert phi > plo, "risk ordering is inverted"
print("verify: PASS")
