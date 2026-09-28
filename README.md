# marine-inference

Inference package for 3-day dissolved-oxygen forecasting at **Bocknis Eck 2** (Baltic Sea), using a trained SageFormer model. Inference only — no training required.

Supply raw sensor readings and receive a 72-hour oxygen forecast. All feature engineering and normalisation happen inside the package.


---

## Scope

- **Included:** a self-contained pipeline from a raw sensor CSV to an oxygen forecast, with fixed weights and scaler statistics for Bocknis Eck 2 (7-day input → 3-day output).
- **Not included:** training or fine-tuning code, and support for other stations. The package is specific to Bocknis Eck 2.

---

## Input

The package takes exactly what the station instruments record and derives everything else internally.

| Field | Requirement |
|---|---|
| Format | CSV with a `date` column and three sensor columns |
| Rows | At least 1008 (7 days at 10-minute resolution); the most recent 1008 are used |
| Resolution | 10 minutes, sorted ascending |
| Gaps | Gaps up to 1 hour are interpolated automatically; longer gaps are rejected with a clear error |

Required columns:

```
date, Sal, Temp, Oxygen
```

- `date` — timestamp in UTC (e.g. `2024-06-01 17:30:00`)
- `Sal` — salinity (PSU)
- `Temp` — water temperature (°C)
- `Oxygen` — dissolved oxygen (mg/L); recent history, used as model input

Time-of-day and seasonal features (`hour_sin/cos`, `day_sin/cos`) are computed from `date` inside the package.

---

## Output

A CSV with 432 rows — one per 10-minute step across the 3-day horizon:

| Column | Description |
|---|---|
| `timestamp` | UTC timestamp of the predicted step |
| `station_id` | Station label (`Bocknis_Eck_2`) |
| `oxygen_mg_per_l` | Predicted dissolved oxygen (mg/L) |

The forecast begins 10 minutes after the last input timestamp.

---

## Installation

Requires Python 3.10.

```bash
pip install -r requirements.txt
```

---

## Usage

Command line:

```bash
python examples/run_inference.py \
    --input  examples/sample_station_input.csv \
    --output predictions.csv
```

Python:

```python
from src.inference import load_model, predict_from_raw

bundle   = load_model("config/model_config.yaml", "checkpoints/bocknis_eck_3day.pth")
forecast = predict_from_raw(bundle, "path/to/raw_station.csv")
print(forecast.head())
```

A runnable example input (`examples/sample_station_input.csv`) is included, so the commands above work out of the box.

---

## Tests

```bash
python -m pytest tests/test_smoke.py -v
```

The suite runs against the bundled example and checks output shape, timestamp alignment, and physical value ranges. It is self-contained and requires no additional data.

---

## Model

SageFormer, 39.4 M parameters, ~160 MB checkpoint. Runs on CPU or GPU.

Performance on the Bocknis Eck 2 test split (30 % held-out, 10-minute resolution):

| Metric | Value |
|---|---|
| MAE | 0.668 mg/L |
| RMSE | 0.904 mg/L |

---

## Attribution

The SageFormer model architecture in `sageformer/model.py` is derived from two
upstream open-source projects, both under the MIT License.

**SageFormer** — the model architecture and graph-learning components:

> **SageFormer: Series-Aware Framework for Long-Term Multivariate Time-Series Forecasting**  
> Zhenwei Zhang, Linghang Meng, Yuantao Gu  
> *IEEE Internet of Things Journal*, 2024. DOI: 10.1109/JIOT.2024.3362640  
> Source: https://github.com/zhangzw16/SageFormer  
> Copyright (c) 2024 ZHANG ZHENWEI — MIT License (see `sageformer/LICENSE`)

**Time-Series-Library** — transformer primitives (`FullAttention`, `AttentionLayer`,
`EncoderLayer`, `PositionalEmbedding`, `PatchEmbedding`, `Flatten_Head`) that
SageFormer incorporates from this library:

> Source: https://github.com/thuml/Time-Series-Library  
> Copyright (c) 2021 THUML @ Tsinghua University — MIT License
> (see `sageformer/LICENSE-Time-Series-Library`)

The remainder of this package (preprocessing, postprocessing, feature engineering,
configuration, and examples) is original work released under the MIT License
(see `LICENSE`).

---

## Layout

```
marine-inference/
├── config/model_config.yaml           model hyperparameters and data spec
├── checkpoints/bocknis_eck_3day.pth    trained weights (Bocknis Eck 2, 3-day horizon)
├── artifacts/scaler_stats.json         normalisation statistics from the training split
├── sageformer/model.py                 model architecture (inference-only)
├── src/
│   ├── raw_to_features.py              feature engineering from raw sensor input
│   ├── preprocessing.py                validation and normalisation
│   ├── postprocessing.py               inverse transform and output assembly
│   └── inference.py                    load_model / predict_from_raw
├── examples/
│   ├── sample_station_input.csv        real 7-day example input
│   └── run_inference.py                command-line entry point
└── tests/test_smoke.py                 shape, alignment, and range checks
```