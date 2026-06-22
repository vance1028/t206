from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .sample_data import PRODUCT_CATEGORIES, STAGES
from .trajectory import build_trajectory, get_stage_timeline


def _arrhenius_rate(temp_c: float, sensitivity: float, ref_temp_c: float = 2.0) -> float:
    temp_k = temp_c + 273.15
    ref_k = ref_temp_c + 273.15
    base = sensitivity * 1000.0
    ratio = np.exp(base / 8.314 * (1.0 / ref_k - 1.0 / temp_k))
    return float(ratio)


def calculate_effective_accumulated_temp(
    batch_id: str,
    batches_df: pd.DataFrame,
    cleaned_temps_df: pd.DataFrame,
    ref_temp_c: float = 2.0,
) -> Tuple[float, pd.DataFrame]:
    batch_row = batches_df[batches_df["batch_id"] == batch_id]
    if batch_row.empty:
        return 0.0, pd.DataFrame()

    category_key = batch_row.iloc[0]["category"]
    category = PRODUCT_CATEGORIES.get(category_key)
    if category is None:
        return 0.0, pd.DataFrame()

    trajectory = build_trajectory(batch_id, batches_df, cleaned_temps_df)
    if trajectory.empty or len(trajectory) < 2:
        return 0.0, trajectory

    times = trajectory["timestamp"].values
    temps = trajectory["temperature"].values
    sensitivity = category.temp_sensitivity

    eat = 0.0
    eat_curve = np.zeros(len(times))
    for i in range(1, len(times)):
        dt_h = (pd.Timestamp(times[i]) - pd.Timestamp(times[i - 1])).total_seconds() / 3600.0
        avg_t = (temps[i] + temps[i - 1]) / 2.0
        rate = _arrhenius_rate(avg_t, sensitivity, ref_temp_c)
        effective_hours = dt_h * rate
        eat += effective_hours
        eat_curve[i] = eat

    result = trajectory.copy()
    result["effective_accumulated_hours"] = eat_curve
    result["instantaneous_rate"] = np.array([
        _arrhenius_rate(t, sensitivity, ref_temp_c) for t in temps
    ])

    return round(eat, 4), result


def estimate_shelf_life(
    batches_df: pd.DataFrame,
    cleaned_temps_df: pd.DataFrame,
    warning_ratio: float = 0.7,
    critical_ratio: float = 0.9,
    ref_temp_c: float = 2.0,
) -> pd.DataFrame:
    rows = []
    for _, batch_row in batches_df.iterrows():
        batch_id = batch_row["batch_id"]
        category_key = batch_row["category"]
        category = PRODUCT_CATEGORIES.get(category_key)
        if category is None:
            continue

        eat, _ = calculate_effective_accumulated_temp(batch_id, batches_df, cleaned_temps_df, ref_temp_c)
        base_life = category.base_shelf_life_hours
        remaining = max(0.0, base_life - eat)
        ratio = eat / base_life if base_life > 0 else 1.0

        if ratio >= critical_ratio:
            risk_level = "critical"
        elif ratio >= warning_ratio:
            risk_level = "warning"
        else:
            risk_level = "normal"

        timeline = get_stage_timeline(batch_id, cleaned_temps_df)
        total_elapsed = timeline["duration_hours"].sum() if not timeline.empty else 0.0

        rows.append({
            "batch_id": batch_id,
            "category": category_key,
            "category_cn": category.name_cn,
            "line_id": batch_row["line_id"],
            "base_shelf_life_hours": base_life,
            "effective_accumulated_hours": round(eat, 2),
            "remaining_hours": round(remaining, 2),
            "used_ratio": round(ratio, 4),
            "risk_level": risk_level,
            "actual_elapsed_hours": round(total_elapsed, 2),
            "loss_rate": batch_row["loss_rate"],
            "harvest_time": batch_row["harvest_time"],
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("used_ratio", ascending=False).reset_index(drop=True)
    return df


def identify_high_risk_batches(
    shelf_life_df: pd.DataFrame,
    min_risk_level: str = "warning",
) -> pd.DataFrame:
    if shelf_life_df.empty:
        return shelf_life_df

    level_order = {"normal": 0, "warning": 1, "critical": 2}
    threshold = level_order.get(min_risk_level, 1)
    mask = shelf_life_df["risk_level"].map(level_order).fillna(0) >= threshold
    return shelf_life_df[mask].reset_index(drop=True)
