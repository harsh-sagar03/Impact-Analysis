from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
IMPACT_PATH = ROOT / "data" / "naye_pankh_sample_impact_data.csv"
CONTEXT_PATH = ROOT / "data" / "india_state_context.csv"
OUTPUT_PATH = ROOT / "data" / "modeling_dataset.csv"


def load_impact_data() -> pd.DataFrame:
    df = pd.read_csv(IMPACT_PATH, parse_dates=["date"])
    df = df.sort_values(["city", "program", "date"]).reset_index(drop=True)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["city", "program"], sort=False)
    for lag in (1, 2, 3):
        df[f"donation_lag_{lag}"] = grouped["donation_amount"].shift(lag)
        df[f"beneficiaries_lag_{lag}"] = grouped["beneficiaries_reached"].shift(lag)

    for lag in (2, 3):
        df[f"donation_lag_{lag}"] = df[f"donation_lag_{lag}"].fillna(df[f"donation_lag_{lag - 1}"])
        df[f"beneficiaries_lag_{lag}"] = df[f"beneficiaries_lag_{lag}"].fillna(
            df[f"beneficiaries_lag_{lag - 1}"]
        )

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df["donation_per_beneficiary"] = df["donation_amount"] / df["beneficiaries_reached"]
    df["expense_ratio"] = df["expense_amount"] / df["donation_amount"].replace(0, pd.NA)
    df["volunteer_productivity"] = df["beneficiaries_reached"] / df["volunteer_hours"]
    df["donation_surplus"] = df["donation_amount"] - df["expense_amount"]
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12).round(4)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12).round(4)
    median_cost = df["cost_per_beneficiary"].median()
    df["high_efficiency"] = (df["cost_per_beneficiary"] <= median_cost).astype(int)
    return df


def merge_context(df: pd.DataFrame) -> pd.DataFrame:
    context = pd.read_csv(CONTEXT_PATH)
    return df.merge(context, on="state", how="left")


def prepare_modeling_dataset() -> pd.DataFrame:
    df = load_impact_data()
    df = add_lag_features(df)
    df = add_derived_features(df)
    df = merge_context(df)
    df = df.dropna(subset=["donation_lag_1", "beneficiaries_lag_1"]).reset_index(drop=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    return df


def main() -> None:
    df = prepare_modeling_dataset()
    print(f"Prepared modeling dataset with {len(df)} rows at {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
