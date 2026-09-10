"""
Load the trained model bundle and produce the two required prediction files:

  1. outputs/validation_predictions.csv
     -> load_id, predicted_rate for every row in data/validation.csv,
        written into the structure of data/validation_predictions_template.csv.

  2. outputs/december_chart_inputs_filled.csv
     -> data/december_chart_inputs.csv with predicted_rate filled in,
        ready to pass to score.py --december-predictions.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from . import config, data_prep


def predict_rates(model, weight_medians, category_levels, raw_df: pd.DataFrame) -> np.ndarray:
    X = data_prep.prepare_features(raw_df, weight_medians, category_levels)
    pred_log = model.predict(X)
    pred = np.expm1(pred_log)
    pred = np.clip(pred, a_min=1.0, a_max=None)  # a predicted rate must be positive
    return pred


def main() -> None:
    bundle = joblib.load(config.MODEL_PATH)
    model = bundle["model"]
    weight_medians = bundle["weight_medians"]
    category_levels = bundle["category_levels"]
    print(f"Loaded model (trained on data up to {bundle['cutoff_date']} in the "
          f"time-based holdout check).")

    # --- 1. validation.csv -> validation_predictions.csv -----------------
    validation_raw = data_prep.load_raw(config.VALIDATION_PATH)
    template = data_prep.load_raw(config.VALIDATION_TEMPLATE_PATH)
    assert (template[config.ID_COL].values == validation_raw[config.ID_COL].values).all(), (
        "validation.csv and validation_predictions_template.csv load_id order must match"
    )

    val_preds = predict_rates(model, weight_medians, category_levels, validation_raw)
    output = template.copy()
    output["predicted_rate"] = np.round(val_preds, 2)
    output.to_csv(config.VALIDATION_PREDICTIONS_PATH, index=False)
    print(f"Wrote {len(output):,} predictions -> {config.VALIDATION_PREDICTIONS_PATH}")

    # --- 2. december_chart_inputs.csv -> filled predicted_rate -----------
    december_original = data_prep.load_raw(config.DECEMBER_INPUT_PATH)  # original 7 columns, kept for output
    december_enriched = data_prep.fill_missing_market_columns(
        december_original,
        city_coords=bundle["city_coords"],
        market_index_default=bundle["market_index_default"],
        quote_signal_default=bundle["quote_signal_default"],
    )
    dec_preds = predict_rates(model, weight_medians, category_levels, december_enriched)

    # score.py requires the ORIGINAL 7 columns, in the original order --
    # the enriched lat/lon/market_index/quote_signal columns above were
    # only needed to feed the model and must not appear in the output file.
    december_filled = december_original.copy()
    december_filled["predicted_rate"] = np.round(dec_preds, 2)
    december_filled.to_csv(config.DECEMBER_PREDICTIONS_PATH, index=False)
    print(f"Wrote {len(december_filled):,} December predictions -> {config.DECEMBER_PREDICTIONS_PATH}")
    print(december_filled[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
