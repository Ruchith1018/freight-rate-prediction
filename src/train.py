"""
Train and validate the freight rate model.

Modeling decisions (see report/report.md and freight_rate_prediction.ipynb
for the full write-up):

1. Split strategy: TIME-BASED, not random.
   train_test.csv covers 2025-01-01 through 2025-10-31. validation.csv and
   the December chart inputs are entirely in the FUTURE relative to that
   (Nov-Dec 2025). A random split would let the model "see" rate patterns
   from late in the year while training on an interleaved random sample,
   which overstates how well it will generalize to genuinely unseen future
   dates. We instead hold out the last ~15% of days (by date) as a
   validation set, so the holdout mimics the real task: predict rates for
   dates after the training window ends.

2. Target transform: log1p(posted_rate).
   posted_rate is strongly right-skewed (median ~$2,031, max ~$25,533) and
   its spread scales with distance. Modeling log(1 + rate) and inverting
   with expm1 at prediction time stabilizes variance and keeps the model
   from being dominated by a handful of very long, very expensive loads.

3. Model: HistGradientBoostingRegressor (scikit-learn).
   Gradient-boosted trees handle the mix of numeric (distance, weight,
   market_index, quote_signal, lat/lon) and categorical (equipment,
   pickup, delivery) features natively, requires no scaling, and copes
   gracefully with the missing values left after imputation choices above
   turn out to be incomplete, and with categorical levels unseen at
   predict time (see data_prep.set_categorical_dtypes).

4. Hyperparameters: chosen by a small randomized search (16 trials) over
   learning_rate / max_leaf_nodes / min_samples_leaf / l2_regularization,
   each trial scored on the SAME time-based holdout as above (a random
   K-fold here would undercut the point of #1). The winning config
   (TUNED_PARAMS below) gave holdout MAE $126.55 vs $127.42 for untuned
   defaults -- a modest gain, expected since distance already explains
   most of the variance. See freight_rate_prediction.ipynb Section 6 for
   the full trial table and search code.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

from . import config, data_prep

# Winning config from the Section 6 hyperparameter search in
# freight_rate_prediction.ipynb (16-trial randomized search over a fixed
# grid, scored on the time-based holdout, seed=config.RANDOM_STATE).
TUNED_PARAMS = {
    "learning_rate": 0.08,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 25,
    "l2_regularization": 1.0,
}


def time_based_split(df: pd.DataFrame, holdout_fraction: float = 0.15):
    df = df.copy()
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    df = df.sort_values(config.DATE_COL)
    cutoff_index = int(len(df) * (1 - holdout_fraction))
    cutoff_date = df.iloc[cutoff_index][config.DATE_COL]
    train_part = df[df[config.DATE_COL] < cutoff_date].reset_index(drop=True)
    holdout_part = df[df[config.DATE_COL] >= cutoff_date].reset_index(drop=True)
    return train_part, holdout_part, cutoff_date


def build_model(params: dict | None = None) -> HistGradientBoostingRegressor:
    params = params or {}
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=600,
        early_stopping=True,
        n_iter_no_change=25,
        validation_fraction=0.1,
        categorical_features=[
            i for i, c in enumerate(config.FEATURE_COLS) if c in config.CATEGORICAL_COLS
        ],
        random_state=config.RANDOM_STATE,
        **params,
    )


def tune_hyperparameters(
    X_train: pd.DataFrame,
    y_train_log: np.ndarray,
    X_holdout: pd.DataFrame,
    y_holdout_log: np.ndarray,
    n_trials: int = 16,
) -> tuple[dict, pd.DataFrame]:
    """
    Small randomized search over HistGradientBoostingRegressor's most
    impactful knobs, each trial scored on the time-based holdout (not a
    random K-fold -- see module docstring for why that matters here).
    Returns (best_params, results_df) sorted by holdout MAE ascending.
    """
    param_space = {
        "learning_rate": [0.03, 0.05, 0.08, 0.1],
        "max_leaf_nodes": [15, 31, 63, 127],
        "min_samples_leaf": [10, 20, 25, 40, 60],
        "l2_regularization": [0.0, 0.1, 0.3, 0.5, 1.0],
    }
    rng = np.random.RandomState(config.RANDOM_STATE)
    y_holdout = np.expm1(y_holdout_log)

    trials = []
    for _ in range(n_trials):
        params = {k: rng.choice(v) for k, v in param_space.items()}
        params = {k: (float(v) if isinstance(v, np.floating) else int(v)) for k, v in params.items()}

        candidate = build_model(params)
        candidate.fit(X_train, y_train_log)
        pred = np.clip(np.expm1(candidate.predict(X_holdout)), 1.0, None)
        trials.append({
            **params,
            "mae": mean_absolute_error(y_holdout, pred),
            "mape": mean_absolute_percentage_error(y_holdout, pred) * 100,
            "r2": r2_score(y_holdout, pred),
        })

    results_df = pd.DataFrame(trials).sort_values("mae").reset_index(drop=True)
    best_row = results_df.iloc[0]
    best_params = {
        "learning_rate": float(best_row["learning_rate"]),
        "max_leaf_nodes": int(best_row["max_leaf_nodes"]),
        "min_samples_leaf": int(best_row["min_samples_leaf"]),
        "l2_regularization": float(best_row["l2_regularization"]),
    }
    return best_params, results_df


def evaluate(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, a_min=1.0, a_max=None)  # rates must be positive
    return {
        "mae_usd": float(mean_absolute_error(y_true, y_pred)),
        "mape_pct": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "r2": float(r2_score(y_true, y_pred)),
        "n_holdout_rows": int(len(y_true)),
    }


def main() -> None:
    raw = data_prep.load_raw(config.TRAIN_TEST_PATH)
    print(f"Loaded {len(raw):,} training rows from {config.TRAIN_TEST_PATH.name}")

    train_raw, holdout_raw, cutoff_date = time_based_split(raw, holdout_fraction=0.15)
    print(f"Time-based split -> train: {len(train_raw):,} rows (< {cutoff_date.date()}), "
          f"holdout: {len(holdout_raw):,} rows (>= {cutoff_date.date()})")

    # Learn imputation stats and categorical levels from the TRAIN slice only.
    weight_medians = data_prep.compute_weight_medians(train_raw)
    category_levels = data_prep.get_category_levels(train_raw)

    X_train = data_prep.prepare_features(train_raw, weight_medians, category_levels)
    X_holdout = data_prep.prepare_features(holdout_raw, weight_medians, category_levels)

    y_train_log = np.log1p(train_raw[config.TARGET_COL].values)
    y_holdout_log = np.log1p(holdout_raw[config.TARGET_COL].values)

    model = build_model()
    model.fit(X_train, y_train_log)

    holdout_pred_log = model.predict(X_holdout)
    metrics = evaluate(y_holdout_log, holdout_pred_log)
    print("\nBaseline holdout (time-based, future-dates) performance:")
    for k, v in metrics.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Sanity-check baseline: naive "distance * training mean rate-per-mile" model,
    # so the report can show the gradient-boosted model beats a simple heuristic.
    train_rpm = (train_raw[config.TARGET_COL] / train_raw["distance"]).mean()
    baseline_pred = holdout_raw["distance"].values * train_rpm
    baseline_mae = mean_absolute_error(holdout_raw[config.TARGET_COL].values, baseline_pred)
    baseline_mape = mean_absolute_percentage_error(holdout_raw[config.TARGET_COL].values, baseline_pred) * 100
    metrics["baseline_distance_only_mae_usd"] = float(baseline_mae)
    metrics["baseline_distance_only_mape_pct"] = float(baseline_mape)
    print(f"\nHeuristic baseline (distance x avg $/mile) MAE: ${baseline_mae:,.2f}  MAPE: {baseline_mape:.2f}%")
    print(f"Model improvement over heuristic baseline: "
          f"{(1 - metrics['mae_usd']/baseline_mae) * 100:.1f}% lower MAE")

    print(f"\nRunning {16}-trial hyperparameter search on the same time-based holdout...")
    tuned_params, search_results = tune_hyperparameters(X_train, y_train_log, X_holdout, y_holdout_log, n_trials=16)
    print("Best hyperparameters found:", tuned_params)
    print(f"Tuned holdout MAE: ${search_results.iloc[0]['mae']:,.2f} "
          f"(vs ${metrics['mae_usd']:,.2f} untuned)")

    # Refit on ALL of train_test.csv (train + holdout) using the tuned
    # hyperparameters before predicting the real validation/December sets,
    # so the shipped model uses every labeled row available AND the best
    # config found. Imputation stats & category levels are recomputed on
    # the full training file for the same reason.
    full_weight_medians = data_prep.compute_weight_medians(raw)
    full_category_levels = data_prep.get_category_levels(raw)
    X_full = data_prep.prepare_features(raw, full_weight_medians, full_category_levels)
    y_full_log = np.log1p(raw[config.TARGET_COL].values)

    final_model = build_model(tuned_params)
    final_model.fit(X_full, y_full_log)

    # Needed to score the December file, which lacks lat/lon and
    # market_index/quote_signal columns entirely -- see data_prep.fill_missing_market_columns.
    city_coords = data_prep.build_city_coords(raw)
    market_index_default = float(raw["market_index"].mean())
    quote_signal_default = float(raw["quote_signal"].mean())

    bundle = {
        "model": final_model,
        "weight_medians": full_weight_medians,
        "category_levels": full_category_levels,
        "city_coords": city_coords,
        "market_index_default": market_index_default,
        "quote_signal_default": quote_signal_default,
        "holdout_metrics": metrics,
        "cutoff_date": str(cutoff_date.date()),
    }
    joblib.dump(bundle, config.MODEL_PATH)
    print(f"\nSaved final model (trained on all {len(raw):,} rows) -> {config.MODEL_PATH}")

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved holdout metrics -> {config.METRICS_PATH}")


if __name__ == "__main__":
    main()