from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from .sample_data import STAGES, STAGE_NAMES_CN


def build_trajectory(
    batch_id: str,
    batches_df: pd.DataFrame,
    cleaned_temps_df: pd.DataFrame,
) -> pd.DataFrame:
    sub = cleaned_temps_df[cleaned_temps_df["batch_id"] == batch_id].copy()
    if sub.empty:
        return sub

    sub = sub.sort_values("timestamp").reset_index(drop=True)
    sub["elapsed_hours"] = (sub["timestamp"] - sub["timestamp"].iloc[0]).dt.total_seconds() / 3600.0

    stage_order = {s: i for i, s in enumerate(STAGES)}
    sub["stage_order"] = sub["stage"].map(stage_order)
    return sub


def get_stage_timeline(
    batch_id: str,
    cleaned_temps_df: pd.DataFrame,
) -> pd.DataFrame:
    sub = cleaned_temps_df[cleaned_temps_df["batch_id"] == batch_id].copy()
    if sub.empty:
        return pd.DataFrame(columns=["stage", "stage_cn", "start_time", "end_time", "duration_hours"])

    timeline = sub.groupby(["stage", "stage_cn"])["timestamp"].agg(["min", "max"]).reset_index()
    timeline.columns = ["stage", "stage_cn", "start_time", "end_time"]
    timeline["duration_hours"] = (timeline["end_time"] - timeline["start_time"]).dt.total_seconds() / 3600.0

    stage_order = {s: i for i, s in enumerate(STAGES)}
    timeline["stage_order"] = timeline["stage"].map(stage_order)
    timeline = timeline.sort_values("stage_order").drop(columns=["stage_order"]).reset_index(drop=True)
    return timeline
