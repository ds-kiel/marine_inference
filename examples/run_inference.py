#!/usr/bin/env python3
"""
Run 3-day dissolved oxygen forecast from raw sensor observations.

Supply only what your instruments record — the package handles all
feature engineering (cyclical time encodings, gap filling, normalisation).

Usage:
    python run_inference.py --input raw_station.csv --output predictions.csv

Input CSV must have columns: date, Sal, Temp, Oxygen
At least 7 days of 10-minute observations (1008+ rows).
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
        description="SageFormer Bocknis Eck 2 — 3-day oxygen forecast"
    )
    parser.add_argument("--input",      required=True,
                        help="Raw station CSV: date, Sal, Temp, Oxygen")
    parser.add_argument("--output",     required=True,
                        help="Output predictions CSV")
    parser.add_argument("--config",     default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--station-id", default="Bocknis_Eck_2")
    parser.add_argument("--device",     default="cpu")
    args = parser.parse_args()

    bundle = load_model(args.config, args.checkpoint, device=args.device)
    result = predict_from_raw(bundle, args.input, station_id=args.station_id)

    result.to_csv(args.output, index=False)
    print(f"Predictions written to: {args.output}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
