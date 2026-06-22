from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .sample_data import PRODUCT_CATEGORIES, STAGES, STAGE_NAMES_CN
from .trajectory import build_trajectory, get_stage_timeline


@dataclass
class ChainBreak:
    batch_id: str
    break_type: str
    stage: str
    stage_cn: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_hours: float
    max_temp: Optional[float] = None
    min_temp: Optional[float] = None
    avg_temp: Optional[float] = None
    temp_threshold_min: Optional[float] = None
    temp_threshold_max: Optional[float] = None
    safe_hours: Optional[float] = None
    severity_score: float = 0.0


def _make_chain_break(
    batch_id: str,
    break_type: str,
    stage: str,
    stage_cn: str,
    start_t: pd.Timestamp,
    end_t: pd.Timestamp,
    seg_temps: np.ndarray,
    temp_min: float,
    temp_max: float,
) -> ChainBreak | None:
    dur = (end_t - start_t).total_seconds() / 3600.0
    if dur <= 0:
        return None
    avg_temp = float(np.mean(seg_temps))
    if break_type == "temp_over_high":
        exceed = float(np.mean(seg_temps - temp_max))
        severity = (dur / 2.0) * (exceed / 3.0 + 1.0)
    else:
        exceed = float(np.mean(temp_min - seg_temps))
        severity = (dur / 2.0) * (exceed / 3.0 + 1.0)
    return ChainBreak(
        batch_id=batch_id,
        break_type=break_type,
        stage=stage,
        stage_cn=stage_cn,
        start_time=start_t,
        end_time=end_t,
        duration_hours=round(dur, 3),
        max_temp=round(float(np.max(seg_temps)), 2),
        min_temp=round(float(np.min(seg_temps)), 2),
        avg_temp=round(avg_temp, 2),
        temp_threshold_min=round(temp_min, 2),
        temp_threshold_max=round(temp_max, 2),
        severity_score=round(severity, 3),
    )


def _detect_temp_breaks(
    trajectory: pd.DataFrame,
    temp_min: float,
    temp_max: float,
    min_break_duration_hours: float = 0.25,
) -> List[ChainBreak]:
    breaks: List[ChainBreak] = []
    if trajectory.empty:
        return breaks

    temps = trajectory["temperature"].values
    times = trajectory["timestamp"].values
    stages = trajectory["stage"].values
    stages_cn = trajectory["stage_cn"].values
    batch_id = trajectory["batch_id"].iloc[0]

    out_of_range = (temps < temp_min) | (temps > temp_max)
    above_max = temps > temp_max
    below_min = temps < temp_min

    in_break = False
    break_start_idx = 0
    break_type = ""
    break_stage = ""
    break_stage_cn = ""

    for i in range(len(out_of_range)):
        if out_of_range[i] and not in_break:
            in_break = True
            break_start_idx = i
            break_type = "temp_over_high" if above_max[i] else "temp_over_low"
            break_stage = stages[i]
            break_stage_cn = stages_cn[i]

        elif in_break and stages[i] != break_stage and out_of_range[i]:
            start_t = pd.Timestamp(times[break_start_idx])
            end_t = pd.Timestamp(times[i - 1])
            seg_temps = temps[break_start_idx:i]
            br = _make_chain_break(
                batch_id, break_type, break_stage, break_stage_cn,
                start_t, end_t, seg_temps, temp_min, temp_max,
            )
            if br and br.duration_hours >= min_break_duration_hours:
                breaks.append(br)

            break_start_idx = i
            break_type = "temp_over_high" if above_max[i] else "temp_over_low"
            break_stage = stages[i]
            break_stage_cn = stages_cn[i]

        elif not out_of_range[i] and in_break:
            in_break = False
            start_t = pd.Timestamp(times[break_start_idx])
            end_t = pd.Timestamp(times[i - 1])
            seg_temps = temps[break_start_idx:i]
            br = _make_chain_break(
                batch_id, break_type, break_stage, break_stage_cn,
                start_t, end_t, seg_temps, temp_min, temp_max,
            )
            if br and br.duration_hours >= min_break_duration_hours:
                breaks.append(br)

    if in_break:
        start_t = pd.Timestamp(times[break_start_idx])
        end_t = pd.Timestamp(times[-1])
        seg_temps = temps[break_start_idx:]
        br = _make_chain_break(
            batch_id, break_type, break_stage, break_stage_cn,
            start_t, end_t, seg_temps, temp_min, temp_max,
        )
        if br and br.duration_hours >= min_break_duration_hours:
            breaks.append(br)

    return breaks


def _detect_delay_breaks(
    batch_id: str,
    timeline: pd.DataFrame,
    safe_hours: Dict[str, float],
    delay_ratio_threshold: float = 1.3,
) -> List[ChainBreak]:
    breaks: List[ChainBreak] = []
    if timeline.empty:
        return breaks

    for _, row in timeline.iterrows():
        stage = row["stage"]
        safe = safe_hours.get(stage)
        if safe is None:
            continue
        actual = row["duration_hours"]
        if actual > safe * delay_ratio_threshold:
            excess = actual - safe
            severity = (excess / safe) * 2.0
            breaks.append(ChainBreak(
                batch_id=batch_id,
                break_type="stage_delay",
                stage=stage,
                stage_cn=row["stage_cn"],
                start_time=row["start_time"],
                end_time=row["end_time"],
                duration_hours=round(actual, 3),
                safe_hours=round(safe, 3),
                severity_score=round(severity, 3),
            ))

    return breaks


def detect_chain_breaks(
    batches_df: pd.DataFrame,
    cleaned_temps_df: pd.DataFrame,
    temp_min_override: Optional[Dict[str, float]] = None,
    temp_max_override: Optional[Dict[str, float]] = None,
    safe_hours_override: Optional[Dict[str, Dict[str, float]]] = None,
    min_break_duration_hours: float = 0.25,
    delay_ratio_threshold: float = 1.3,
) -> List[ChainBreak]:
    all_breaks: List[ChainBreak] = []

    for _, batch_row in batches_df.iterrows():
        batch_id = batch_row["batch_id"]
        category_key = batch_row["category"]
        category = PRODUCT_CATEGORIES.get(category_key)
        if category is None:
            continue

        t_min = temp_min_override.get(category_key, category.temp_min) if temp_min_override else category.temp_min
        t_max = temp_max_override.get(category_key, category.temp_max) if temp_max_override else category.temp_max
        safe_h = safe_hours_override.get(category_key, category.safe_hours) if safe_hours_override else category.safe_hours

        trajectory = build_trajectory(batch_id, batches_df, cleaned_temps_df)
        timeline = get_stage_timeline(batch_id, cleaned_temps_df)

        temp_breaks = _detect_temp_breaks(trajectory, t_min, t_max, min_break_duration_hours)
        delay_breaks = _detect_delay_breaks(batch_id, timeline, safe_h, delay_ratio_threshold)

        all_breaks.extend(temp_breaks)
        all_breaks.extend(delay_breaks)

    return all_breaks


def aggregate_breaks(breaks: List[ChainBreak]) -> pd.DataFrame:
    if not breaks:
        return pd.DataFrame(columns=[
            "batch_id", "break_type", "stage", "stage_cn", "start_time", "end_time",
            "duration_hours", "max_temp", "min_temp", "avg_temp",
            "temp_threshold_min", "temp_threshold_max", "safe_hours", "severity_score",
        ])

    rows = []
    for b in breaks:
        rows.append({
            "batch_id": b.batch_id,
            "break_type": b.break_type,
            "stage": b.stage,
            "stage_cn": b.stage_cn,
            "start_time": b.start_time,
            "end_time": b.end_time,
            "duration_hours": b.duration_hours,
            "max_temp": b.max_temp,
            "min_temp": b.min_temp,
            "avg_temp": b.avg_temp,
            "temp_threshold_min": b.temp_threshold_min,
            "temp_threshold_max": b.temp_threshold_max,
            "safe_hours": b.safe_hours,
            "severity_score": b.severity_score,
        })
    return pd.DataFrame(rows)
