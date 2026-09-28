"""
Smoke tests for the marine-inference package.

Run from marine-inference/:
    python -m pytest tests/test_smoke.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import pytest

REPO_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG     = os.path.join(REPO_ROOT, "config",      "model_config.yaml")
CHECKPOINT = os.path.join(REPO_ROOT, "checkpoints", "bocknis_eck_3day.pth")
SAMPLE_CSV = os.path.join(REPO_ROOT, "examples",    "sample_station_input.csv")

from src.inference import load_model, predict_from_raw, predict
from src.preprocessing import build_input, load_scaler_stats
from src.postprocessing import inverse_transform_oxygen

PRED_LEN = 432


@pytest.fixture(scope="module")
def bundle():
    return load_model(CONFIG, CHECKPOINT, device="cpu")


def test_raw_input_columns():
    """Confirm sample CSV has only the raw columns partners supply."""
    df = pd.read_csv(SAMPLE_CSV)
    assert list(df.columns) == ["date", "Sal", "Temp", "Oxygen"], \
        f"Sample CSV has unexpected columns: {list(df.columns)}"


def test_output_shape(bundle):
    result = predict_from_raw(bundle, SAMPLE_CSV)
    assert result.shape == (PRED_LEN, 3), f"Expected ({PRED_LEN}, 3), got {result.shape}"


def test_output_columns(bundle):
    result = predict_from_raw(bundle, SAMPLE_CSV)
    assert list(result.columns) == ["timestamp", "station_id", "oxygen_mg_per_l"]


def test_output_timestamps(bundle):
    result = predict_from_raw(bundle, SAMPLE_CSV)
    df_in = pd.read_csv(SAMPLE_CSV, parse_dates=["date"])
    expected_start = df_in["date"].max() + pd.Timedelta("10min")
    assert result["timestamp"].iloc[0] == expected_start
    deltas = result["timestamp"].diff().dropna()
    assert (deltas == pd.Timedelta("10min")).all()


def test_oxygen_range(bundle):
    result = predict_from_raw(bundle, SAMPLE_CSV)
    lo, hi = result["oxygen_mg_per_l"].min(), result["oxygen_mg_per_l"].max()
    assert 0.0 <= lo, f"Oxygen below zero: {lo}"
    assert hi <= 16.0, f"Oxygen unrealistically high: {hi}"
    print(f"\nOxygen range: [{lo:.4f}, {hi:.4f}] mg/L")


def test_parity_with_training_repo(bundle):
    """
    Re-implement the training-repo inference path and compare.

    SageFormer's graph_constructor adds torch.rand_like(adj)*0.01 noise per
    forward pass — both paths are seeded with torch.manual_seed(0) so the
    residual difference comes only from JSON stat rounding (~1e-6 mg/L).
    """
    from sklearn.preprocessing import StandardScaler

    CSV_PATH = "/data22/nas/time-series-experiments/Insyst/final_training_data/Bocknis_Eck_2_NoWind.csv"
    if not os.path.exists(CSV_PATH):
        pytest.skip("Original training CSV not available — skipping parity check")

    df_raw = pd.read_csv(CSV_PATH)
    target = "Oxygen"
    cols = [c for c in df_raw.columns if c not in ("date", target)]
    df_raw = df_raw[["date"] + cols + [target]]

    n = len(df_raw)
    num_train = int(n * 0.7)
    df_data = df_raw[df_raw.columns[1:]]

    scaler_orig = StandardScaler()
    scaler_orig.fit(df_data.iloc[:num_train].values)

    # Use the same rows as sample_station_input.csv
    df_sample_raw = pd.read_csv(SAMPLE_CSV, parse_dates=["date"])
    from src.raw_to_features import prepare_features
    stats = bundle["stats"]
    df_feat = prepare_features(df_sample_raw, stats)

    FEAT = ["hour_sin", "hour_cos", "day_sin", "day_cos", "Sal", "Temp", "Oxygen"]
    raw = df_feat[FEAT].values.astype(np.float32)
    norm_sklearn = scaler_orig.transform(raw).astype(np.float32)
    x_enc_orig = torch.from_numpy(norm_sklearn[np.newaxis, :, :]).float()

    model = bundle["model"]
    model.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        out_orig = model(x_enc_orig)

    pred_orig_norm = out_orig[0, :, -1].cpu().numpy()
    pred_orig = pred_orig_norm * float(scaler_orig.scale_[-1]) + float(scaler_orig.mean_[-1])

    result = predict_from_raw(bundle, SAMPLE_CSV)
    pred_new = result["oxygen_mg_per_l"].values

    max_diff = np.abs(pred_orig - pred_new).max()
    print(f"\nMax absolute difference vs training repo: {max_diff:.2e} mg/L")
    assert max_diff < 5e-3, (
        f"Parity check failed: max diff {max_diff:.2e} exceeds 5e-3 mg/L."
    )


if __name__ == "__main__":
    b = load_model(CONFIG, CHECKPOINT, device="cpu")
    test_raw_input_columns()
    test_output_shape(b)
    test_output_columns(b)
    test_output_timestamps(b)
    test_oxygen_range(b)
    test_parity_with_training_repo(b)
    print("\nAll tests passed.")
