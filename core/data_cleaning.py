from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from .sample_data import PRODUCT_CATEGORIES, STAGES, STAGE_NAMES_CN


@dataclass
class DataIssue:
    issue_type: str
    batch_id: str
    stage: str | None = None
    description: str = ""
    severity: str = "warning"


def _detect_temp_anomalies(
    temps_df: pd.DataFrame,
    batch_id: str,
    category: str,
    z_thresh: float = 3.5,
    phys_min: float = -25.0,
    phys_max: float = 40.0,
) -> Tuple[pd.DataFrame, List[DataIssue]]:
    issues: List[DataIssue] = []
    sub = temps_df[temps_df["batch_id"] == batch_id].copy()

    if sub.empty:
        return sub, issues

    mask_null = sub["temperature"].isna()
    null_count = int(mask_null.sum())
    if null_count > 0:
        issues.append(DataIssue(
            issue_type="missing_temp",
            batch_id=batch_id,
            description=f"温度记录缺失 {null_count} 条",
            severity="warning",
        ))
        sub.loc[mask_null, "temperature"] = sub["temperature"].interpolate(limit_direction="both")
        if sub["temperature"].isna().any():
            cat = PRODUCT_CATEGORIES.get(category)
            fill = (cat.temp_min + cat.temp_max) / 2.0 if cat else 2.0
            sub["temperature"] = sub["temperature"].fillna(fill)

    mask_phys = (sub["temperature"] < phys_min) | (sub["temperature"] > phys_max)
    phys_count = int(mask_phys.sum())
    if phys_count > 0:
        issues.append(DataIssue(
            issue_type="physical_anomaly",
            batch_id=batch_id,
            description=f"物理范围外温度 {phys_count} 条 (<{phys_min}°C 或 >{phys_max}°C)",
            severity="error",
        ))
        sub.loc[mask_phys, "temperature"] = np.nan
        sub["temperature"] = sub["temperature"].interpolate(limit_direction="both")
        if sub["temperature"].isna().any():
            cat = PRODUCT_CATEGORIES.get(category)
            fill = (cat.temp_min + cat.temp_max) / 2.0 if cat else 2.0
            sub["temperature"] = sub["temperature"].fillna(fill)

    temps = sub["temperature"].values
    if len(temps) >= 5:
        mean_t = np.mean(temps)
        std_t = np.std(temps)
        if std_t > 0:
            z_scores = np.abs((temps - mean_t) / std_t)
            mask_spike = z_scores > z_thresh
            spike_count = int(mask_spike.sum())
            if spike_count > 0 and spike_count < len(temps) * 0.2:
                issues.append(DataIssue(
                    issue_type="statistical_spike",
                    batch_id=batch_id,
                    description=f"统计异常尖峰 {spike_count} 条 (Z>{z_thresh})",
                    severity="warning",
                ))
                sub.loc[mask_spike, "temperature"] = np.nan
                sub["temperature"] = sub["temperature"].interpolate(limit_direction="both")

    return sub.sort_values("timestamp").reset_index(drop=True), issues


def validate_batch(
    batch_id: str,
    batches_df: pd.DataFrame,
    temps_df: pd.DataFrame,
) -> List[DataIssue]:
    issues: List[DataIssue] = []
    batch_row = batches_df[batches_df["batch_id"] == batch_id]

    if batch_row.empty:
        issues.append(DataIssue(
            issue_type="missing_batch",
            batch_id=batch_id,
            description="批次基本信息缺失",
            severity="error",
        ))
        return issues

    category = batch_row.iloc[0]["category"]
    sub_temps = temps_df[temps_df["batch_id"] == batch_id].copy()
    present_stages = set(sub_temps["stage"].unique()) if not sub_temps.empty else set()

    for stage in STAGES:
        if stage not in present_stages:
            issues.append(DataIssue(
                issue_type="missing_stage",
                batch_id=batch_id,
                stage=stage,
                description=f"缺失环节记录: {STAGE_NAMES_CN.get(stage, stage)}",
                severity="warning",
            ))

    if not sub_temps.empty:
        stage_times = sub_temps.groupby("stage")["timestamp"].agg(["min", "max"])
        prev_max = None
        prev_stage = None
        for stage in STAGES:
            if stage in stage_times.index:
                cur_min = stage_times.loc[stage, "min"]
                if prev_max is not None and cur_min < prev_max:
                    issues.append(DataIssue(
                        issue_type="time_overlap",
                        batch_id=batch_id,
                        stage=stage,
                        description=f"环节时间倒序: {STAGE_NAMES_CN.get(prev_stage, prev_stage)} -> {STAGE_NAMES_CN.get(stage, stage)}",
                        severity="error",
                    ))
                prev_max = stage_times.loc[stage, "max"]
                prev_stage = stage

    return issues


def clean_data(
    batches_df: pd.DataFrame,
    temps_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[DataIssue]]:
    all_issues: List[DataIssue] = []
    cleaned_frames: List[pd.DataFrame] = []

    for _, batch_row in batches_df.iterrows():
        batch_id = batch_row["batch_id"]
        category = batch_row["category"]

        issues = validate_batch(batch_id, batches_df, temps_df)
        all_issues.extend(issues)

        cleaned, temp_issues = _detect_temp_anomalies(temps_df, batch_id, category)
        all_issues.extend(temp_issues)
        cleaned_frames.append(cleaned)

    cleaned_temps = pd.concat(cleaned_frames, ignore_index=True) if cleaned_frames else temps_df.copy()
    cleaned_temps = cleaned_temps.sort_values(["batch_id", "timestamp"]).reset_index(drop=True)

    return batches_df.copy(), cleaned_temps, all_issues
