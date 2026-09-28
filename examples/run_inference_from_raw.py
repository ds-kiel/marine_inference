#!/usr/bin/env python3
"""
Run 3-day oxygen forecast from RAW sensor observations.

Partners supply only what their instruments record.  All feature engineering
(cyclical time encodings, wind decomposition, gap filling, normalisation)
is handled internally.

NoWind model:
    python run_inference_from_raw.py \\
        --station  examples/sample_raw_station.csv \\
        --output   predictions.csv

WithWind model:
    python run_inference_from_raw.py \\
        --station  examples/sample_raw_station.csv \\
        --wind     examples/sample_raw_wind.csv \\
        --checkpoint checkpoints/bocknis_eck_3day.pth \\
        --output   predictions.csv

Station CSV must have columns: date, Sal, Temp, Oxygen
Wind CSV must have columns   : date, t, ff1, gust, dd  (dd = wind direction, degrees)
Both must cover the most recent 7 days at 10-minute resolution (wind may be hourly).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference import load_model, predict_from_raw

DEFAULT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "config", "model_config.yaml")
DEFAULT_CKPT   = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "bocknis_eck_3day.pth")


def main():
    parser = argparse.ArgumentParser(
        description="SageFormer Bocknis Eck 2 oxygen inference from raw sensor data"
    )
    parser.add_argument("--station",    required=True,
                        help="Raw station CSV: date, Sal, Temp, Oxygen")
    parser.add_argument("--wind",       default=None,
                        help="Raw wind CSV: date, t, ff1, gust, dd  (withwind model only)")
    parser.add_argument("--output",     required=True,
                        help="Path for output predictions CSV")
    parser.add_argument("--config",     default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--station-id", default="Bocknis_Eck_2")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    bundle = load_model(args.config, args.checkpoint, device=args.device)
    result = predict_from_raw(
        bundle,
        station_input=args.station,
        wind_input=args.wind,
        station_id=args.station_id,
    )

    result.to_csv(args.output, index=False)
    print(f"Predictions written to: {args.output}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
