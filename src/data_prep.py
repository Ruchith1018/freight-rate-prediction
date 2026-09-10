"""
Data cleaning and feature engineering shared by training and prediction.

Design decision: every cleaning step here is *deterministic and stateless*
except for one thing (median weight-by-equipment for imputation), which is
learned on the training data only and passed in explicitly to avoid leaking
validation/December statistics into training. See train.py for how the
medians are computed and threaded through.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_raw(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean_weight(df: pd.DataFrame, fallback_medians: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Data-quality issue #1: `weight` contains a small number of negative
    values (~0.6% of rows). These are physically impossible for a freight
    load and are treated as a sign-entry error, not as legitimately
    negative data -> we take the absolute value rather than dropping the
    rows (dropping would throw away otherwise-valid distance/rate
    information, and the assessment requires predictions for every row
    regardless).

    Data-quality issue #2: `weight` has missing values (~0.6% of rows).
    We impute with the *training-set* median weight for that row's
    equipment type (weight distributions differ meaningfully by equipment,
    see EDA), falling back to the global training median if an equipment
    type is unseen.
    """
    df = df.copy()
    df["weight"] = df["weight"].astype(float).abs()

    if fallback_medians is not None:
        global_median = fallback_medians["__global__"]
        for equipment, median_value in fallback_medians.items():
            if equipment == "__global__":
                continue
            mask = (df["equipment"] == equipment) & (df["weight"].isna())
            df.loc[mask, "weight"] = median_value
        df["weight"] = df["weight"].fillna(global_median)
    return df


def compute_weight_medians(train_df: pd.DataFrame) -> dict[str, float]:
    """Compute per-equipment (and global) median weight from TRAINING data only."""
    cleaned = train_df.copy()
    cleaned["weight"] = cleaned["weight"].abs()
    medians = cleaned.groupby("equipment")["weight"].median().to_dict()
    medians["__global__"] = cleaned["weight"].median()
    return medians


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Turn the raw calendar date into features a tree model can use:
      - month, day_of_week, is_weekend: capture weekly / monthly patterns
      - day_of_year encoded as sin/cos: captures broader seasonality
        (e.g. Q4 freight surge) as a smooth, cyclical signal instead of
        a raw integer that would wrongly imply Dec 31 and Jan 1 are far
        apart.
    """
    df = df.copy()
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL])
    day_of_year = df[config.DATE_COL].dt.dayofyear
    df["month"] = df[config.DATE_COL].dt.month
    df["day_of_week"] = df[config.DATE_COL].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    return df


def set_categorical_dtypes(df: pd.DataFrame, category_levels: dict[str, list[str]]) -> pd.DataFrame:
    """
    Cast categorical columns to pandas `category` dtype with a FIXED set of
    levels learned from the training data. This is what lets
    HistGradientBoostingRegressor treat them as native categoricals, and it
    also means a city seen in validation/December but not in training
    (see EDA notes: 8 validation pickup/delivery cities are unseen in
    training) is encoded as NaN/"unknown" rather than crashing or silently
    being mis-mapped.
    """
    df = df.copy()
    for col in config.CATEGORICAL_COLS:
        df[col] = pd.Categorical(df[col], categories=category_levels[col])
    return df


def get_category_levels(train_df: pd.DataFrame) -> dict[str, list[str]]:
    return {col: sorted(train_df[col].dropna().unique().tolist()) for col in config.CATEGORICAL_COLS}


def build_city_coords(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """
    Map city name -> (lat, lon), learned from training data. Every city
    appears with one fixed lat/lon pair throughout train_test.csv (verified
    in EDA), whether it shows up as a pickup or a delivery, so a single
    lookup table built from both columns covers every city we've seen.

    Needed because december_chart_inputs.csv only gives city NAMES
    (Lexington / Fort Wayne), not lat/lon columns -- unlike train_test.csv
    and validation.csv, which include both.
    """
    pickup_coords = train_df[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
    )
    delivery_coords = train_df[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
    )
    all_coords = pd.concat([pickup_coords, delivery_coords], ignore_index=True).drop_duplicates("city")
    return {row.city: (row.lat, row.lon) for row in all_coords.itertuples()}


def fill_missing_market_columns(
    df: pd.DataFrame,
    city_coords: dict[str, tuple[float, float]],
    market_index_default: float,
    quote_signal_default: float,
) -> pd.DataFrame:
    """
    december_chart_inputs.csv omits pickup_lat/pickup_lon/delivery_lat/
    delivery_lon and market_index/quote_signal entirely (unlike
    train_test.csv / validation.csv). Reconstruct what we can:

      - lat/lon: looked up deterministically from `city_coords` (exact,
        not an estimate -- every city has one fixed coordinate pair).
      - market_index / quote_signal: these two columns barely correlate
        with posted_rate in training data (|r| < 0.1, see EDA), and
        December is entirely outside the training date range with no
        clear seasonal trend in either signal (checked month-by-month:
        both fluctuate without a consistent direction). Rather than
        extrapolate a trend we can't support with evidence, we fill both
        with their TRAINING-SET global mean as a neutral placeholder, so
        the December seasonal curve is driven by the features that
        actually matter here: distance, weight, equipment, and date.
    """
    df = df.copy()
    if "pickup_lat" not in df.columns:
        df["pickup_lat"] = df["pickup"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[0])
        df["pickup_lon"] = df["pickup"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[1])
    if "delivery_lat" not in df.columns:
        df["delivery_lat"] = df["delivery"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[0])
        df["delivery_lon"] = df["delivery"].map(lambda c: city_coords.get(c, (np.nan, np.nan))[1])
    if "market_index" not in df.columns:
        df["market_index"] = market_index_default
    if "quote_signal" not in df.columns:
        df["quote_signal"] = quote_signal_default
    return df


def prepare_features(
    df: pd.DataFrame,
    weight_medians: dict[str, float],
    category_levels: dict[str, list[str]],
) -> pd.DataFrame:
    """Full pipeline: clean weight -> add date features -> set categorical dtypes -> select feature columns."""
    df = clean_weight(df, fallback_medians=weight_medians)
    df = add_date_features(df)
    df = set_categorical_dtypes(df, category_levels)
    return df[config.FEATURE_COLS]
