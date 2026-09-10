"""
Standalone exploratory data analysis script. Not required to produce the
submission files, but documents the key findings referenced in the report
and saves supporting plots to outputs/eda_plots/.

Run: python -m src.eda
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, data_prep


def main() -> None:
    df = data_prep.load_raw(config.TRAIN_TEST_PATH)
    val = data_prep.load_raw(config.VALIDATION_PATH)

    print("=" * 70)
    print("SHAPE & MISSING VALUES")
    print("=" * 70)
    print(f"train_test.csv: {df.shape}")
    print(df.isna().sum()[df.isna().sum() > 0])

    print("\n" + "=" * 70)
    print("DATA QUALITY ISSUES FOUND")
    print("=" * 70)
    n_negative_weight = (df["weight"] < 0).sum()
    n_missing_weight = df["weight"].isna().sum()
    n_missing_market = df["market_index"].isna().sum()
    print(f"- weight: {n_negative_weight} negative values, {n_missing_weight} missing "
          f"({(n_negative_weight + n_missing_weight) / len(df) * 100:.2f}% of rows affected)")
    print(f"- market_index: {n_missing_market} missing ({n_missing_market / len(df) * 100:.2f}%)")

    train_cities = set(df["pickup"]) | set(df["delivery"])
    val_cities = set(val["pickup"]) | set(val["delivery"])
    unseen = val_cities - train_cities
    print(f"- validation.csv contains {len(unseen)} pickup/delivery cities never seen in "
          f"train_test.csv: {sorted(unseen)}")
    train_lanes = set(zip(df["pickup"], df["delivery"]))
    val_lanes = set(zip(val["pickup"], val["delivery"]))
    unseen_lanes = val_lanes - train_lanes
    print(f"- {len(unseen_lanes)} of {len(val_lanes)} validation pickup->delivery lanes "
          f"never appear in training data")

    print("\n" + "=" * 70)
    print("KEY RELATIONSHIPS")
    print("=" * 70)
    print(f"corr(distance, posted_rate)     = {df['distance'].corr(df['posted_rate']):.3f}")
    print(f"corr(weight, posted_rate)       = {df['weight'].corr(df['posted_rate']):.3f}")
    print(f"corr(market_index, posted_rate) = {df['market_index'].corr(df['posted_rate']):.3f}")
    print(f"corr(quote_signal, posted_rate) = {df['quote_signal'].corr(df['posted_rate']):.3f}")

    df["rate_per_mile"] = df["posted_rate"] / df["distance"]
    print("\nMean $/mile by equipment:")
    print(df.groupby("equipment")["rate_per_mile"].mean().round(3))

    # --- Plots -------------------------------------------------------------
    config.EDA_PLOT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(df["distance"], df["posted_rate"], s=4, alpha=0.25, color="#064A56")
    ax.set_xlabel("distance (miles)")
    ax.set_ylabel("posted_rate ($)")
    ax.set_title("posted_rate vs distance")
    fig.tight_layout()
    fig.savefig(config.EDA_PLOT_DIR / "rate_vs_distance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    df.boxplot(column="rate_per_mile", by="equipment", ax=ax)
    ax.set_ylim(0, df["rate_per_mile"].quantile(0.99))
    ax.set_title("$/mile by equipment type")
    plt.suptitle("")
    fig.tight_layout()
    fig.savefig(config.EDA_PLOT_DIR / "rate_per_mile_by_equipment.png", dpi=150)
    plt.close(fig)

    monthly = df.copy()
    monthly[config.DATE_COL] = pd.to_datetime(monthly[config.DATE_COL])
    monthly = monthly.groupby(monthly[config.DATE_COL].dt.to_period("M"))["rate_per_mile"].mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    monthly.plot(kind="bar", ax=ax, color="#064A56")
    ax.set_title("Mean $/mile by month (2025 training data)")
    ax.set_ylabel("$/mile")
    fig.tight_layout()
    fig.savefig(config.EDA_PLOT_DIR / "monthly_rate_per_mile.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved 3 EDA plots to {config.EDA_PLOT_DIR}")


if __name__ == "__main__":
    main()
