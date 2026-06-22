from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from .sample_data import PRODUCT_CATEGORIES, STAGES, STAGE_NAMES_CN
from .chain_break import ChainBreak, aggregate_breaks
from .trajectory import get_stage_timeline


def _build_batch_metrics(
    batches_df: pd.DataFrame,
    cleaned_temps_df: pd.DataFrame,
    breaks: List[ChainBreak],
) -> pd.DataFrame:
    breaks_df = aggregate_breaks(breaks)

    metrics = batches_df[["batch_id", "category", "category_cn", "line_id", "origin", "dest", "loss_rate", "weight_kg"]].copy()

    if breaks_df.empty:
        metrics["break_count"] = 0
        metrics["temp_break_count"] = 0
        metrics["delay_break_count"] = 0
        metrics["total_overtemp_hours"] = 0.0
        metrics["total_delay_hours"] = 0.0
        metrics["total_severity"] = 0.0
        for s in STAGES:
            metrics[f"{s}_break_count"] = 0
            metrics[f"{s}_overtemp_hours"] = 0.0
        return metrics

    batch_stats = breaks_df.groupby("batch_id").agg(
        break_count=("break_type", "count"),
        total_overtemp_hours=("duration_hours", lambda x: x[breaks_df.loc[x.index, "break_type"].isin(["temp_over_high", "temp_over_low"])].sum()),
        total_delay_hours=("duration_hours", lambda x: x[breaks_df.loc[x.index, "break_type"] == "stage_delay"].sum()),
        total_severity=("severity_score", "sum"),
    ).reset_index()

    temp_break_counts = breaks_df[breaks_df["break_type"].isin(["temp_over_high", "temp_over_low"])].groupby("batch_id").size().reset_index(name="temp_break_count")
    delay_break_counts = breaks_df[breaks_df["break_type"] == "stage_delay"].groupby("batch_id").size().reset_index(name="delay_break_count")

    stage_counts = breaks_df.groupby(["batch_id", "stage"]).size().unstack(fill_value=0).reset_index()
    stage_overtemp = breaks_df[breaks_df["break_type"].isin(["temp_over_high", "temp_over_low"])].groupby(["batch_id", "stage"])["duration_hours"].sum().unstack(fill_value=0).reset_index()

    metrics = metrics.merge(batch_stats, on="batch_id", how="left")
    metrics = metrics.merge(temp_break_counts, on="batch_id", how="left")
    metrics = metrics.merge(delay_break_counts, on="batch_id", how="left")

    for s in STAGES:
        col = f"{s}_break_count"
        if s in stage_counts.columns:
            metrics = metrics.merge(stage_counts[["batch_id", s]].rename(columns={s: col}), on="batch_id", how="left")
        else:
            metrics[col] = 0
        metrics[col] = metrics[col].fillna(0).astype(int)

        col2 = f"{s}_overtemp_hours"
        if s in stage_overtemp.columns:
            metrics = metrics.merge(stage_overtemp[["batch_id", s]].rename(columns={s: col2}), on="batch_id", how="left")
        else:
            metrics[col2] = 0.0
        metrics[col2] = metrics[col2].fillna(0.0)

    for col in ["break_count", "temp_break_count", "delay_break_count"]:
        if col in metrics.columns:
            metrics[col] = metrics[col].fillna(0).astype(int)
    for col in ["total_overtemp_hours", "total_delay_hours", "total_severity"]:
        if col in metrics.columns:
            metrics[col] = metrics[col].fillna(0.0)

    stage_durations_list = []
    for bid in metrics["batch_id"]:
        tl = get_stage_timeline(bid, cleaned_temps_df)
        row = {"batch_id": bid}
        for _, r in tl.iterrows():
            row[f"{r['stage']}_duration_hours"] = r["duration_hours"]
        stage_durations_list.append(row)
    stage_dur_df = pd.DataFrame(stage_durations_list)
    metrics = metrics.merge(stage_dur_df, on="batch_id", how="left")
    for s in STAGES:
        col = f"{s}_duration_hours"
        if col not in metrics.columns:
            metrics[col] = 0.0
        metrics[col] = metrics[col].fillna(0.0)

    return metrics


def attribute_loss(
    batches_df: pd.DataFrame,
    cleaned_temps_df: pd.DataFrame,
    breaks: List[ChainBreak],
) -> pd.DataFrame:
    return _build_batch_metrics(batches_df, cleaned_temps_df, breaks)


def get_stage_loss_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in STAGES:
        col_cnt = f"{s}_break_count"
        col_hours = f"{s}_overtemp_hours"
        col_dur = f"{s}_duration_hours"

        has_break = metrics_df[col_cnt] > 0
        no_break = metrics_df[col_cnt] == 0

        rows.append({
            "stage": s,
            "stage_cn": STAGE_NAMES_CN.get(s, s),
            "batch_count": int(len(metrics_df)),
            "batches_with_break": int(has_break.sum()),
            "break_rate": float(has_break.mean()) if len(metrics_df) > 0 else 0.0,
            "avg_loss_with_break": float(metrics_df.loc[has_break, "loss_rate"].mean()) if has_break.any() else 0.0,
            "avg_loss_no_break": float(metrics_df.loc[no_break, "loss_rate"].mean()) if no_break.any() else 0.0,
            "loss_delta": float(
                metrics_df.loc[has_break, "loss_rate"].mean() - metrics_df.loc[no_break, "loss_rate"].mean()
            ) if (has_break.any() and no_break.any()) else 0.0,
            "total_overtemp_hours": float(metrics_df[col_hours].sum()) if col_hours in metrics_df.columns else 0.0,
            "avg_duration_hours": float(metrics_df[col_dur].mean()) if col_dur in metrics_df.columns else 0.0,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("loss_delta", ascending=False).reset_index(drop=True)
    return df


def _corr(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 3:
        return 0.0
    mask = ~(x.isna() | y.isna())
    if mask.sum() < 3:
        return 0.0
    xm = x[mask]
    ym = y[mask]
    if xm.std() == 0 or ym.std() == 0:
        return 0.0
    return float(xm.corr(ym))


def get_factor_importance(metrics_df: pd.DataFrame) -> pd.DataFrame:
    factor_cols = [
        ("break_count", "断链总次数"),
        ("temp_break_count", "温度断链次数"),
        ("delay_break_count", "滞留断链次数"),
        ("total_overtemp_hours", "累计超温时长(小时)"),
        ("total_delay_hours", "累计滞留超时(小时)"),
        ("total_severity", "断链严重度总分"),
    ]
    for s in STAGES:
        factor_cols.append((f"{s}_break_count", f"{STAGE_NAMES_CN[s]}断链次数"))
        factor_cols.append((f"{s}_overtemp_hours", f"{STAGE_NAMES_CN[s]}超温时长"))

    rows = []
    for col, name in factor_cols:
        if col not in metrics_df.columns:
            continue
        corr = _corr(metrics_df[col], metrics_df["loss_rate"])

        med = metrics_df[col].median() if len(metrics_df) > 0 else 0
        high = metrics_df[col] > med
        if high.any() and (~high).any():
            delta = float(metrics_df.loc[high, "loss_rate"].mean() - metrics_df.loc[~high, "loss_rate"].mean())
        else:
            delta = 0.0

        rows.append({
            "factor": col,
            "factor_name": name,
            "correlation": round(corr, 4),
            "abs_correlation": round(abs(corr), 4),
            "high_group_loss_delta": round(delta, 4),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("abs_correlation", ascending=False).reset_index(drop=True)
    return df
