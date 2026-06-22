from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from core.sample_data import PRODUCT_CATEGORIES, STAGES, generate_sample_data
from core.data_cleaning import clean_data, validate_batch, DataIssue
from core.chain_break import detect_chain_breaks, aggregate_breaks
from core.loss_attribution import attribute_loss, get_stage_loss_summary, get_factor_importance
from core.shelf_life import estimate_shelf_life, identify_high_risk_batches


class TestDataCleaning:
    def test_harvest_time_before_batch_harvest(self, make_batch):
        harvest = datetime(2026, 6, 1, 8, 0, 0)
        batches_df, temps_df = make_batch("T_HARVEST_BAD", harvest_time=harvest)

        harvest_sub = temps_df[temps_df["stage"] == "harvest"].copy()
        harvest_sub["timestamp"] = harvest - timedelta(hours=2)
        temps_df.loc[temps_df["stage"] == "harvest", "timestamp"] = harvest_sub["timestamp"].values

        issues = validate_batch("T_HARVEST_BAD", batches_df, temps_df)
        issue_types = [i.issue_type for i in issues]
        assert "harvest_time_anomaly" in issue_types, "应检测到采收记录早于批次采收时间"

    def test_stage_before_harvest(self, make_batch):
        harvest = datetime(2026, 6, 1, 8, 0, 0)
        batches_df, temps_df = make_batch("T_STAGE_BEFORE", harvest_time=harvest)

        transport = temps_df[temps_df["stage"] == "transport"].copy()
        transport["timestamp"] = harvest - timedelta(hours=5)
        temps_df.loc[temps_df["stage"] == "transport", "timestamp"] = transport["timestamp"].values

        issues = validate_batch("T_STAGE_BEFORE", batches_df, temps_df)
        issue_types = [i.issue_type for i in issues]
        assert "stage_before_harvest" in issue_types, "应检测到运输记录早于采收时间"

    def test_stage_start_before_harvest_end(self, make_batch):
        harvest = datetime(2026, 6, 1, 8, 0, 0)
        batches_df, temps_df = make_batch("T_OVERLAP", harvest_time=harvest)

        harvest_end = harvest + timedelta(hours=2)
        precool_sub = temps_df[temps_df["stage"] == "precool"].copy()
        precool_times = precool_sub["timestamp"].values
        precool_times = [harvest_end - timedelta(minutes=30), harvest_end + timedelta(minutes=30)]
        temps_df.loc[temps_df["stage"] == "precool", "timestamp"] = pd.to_datetime(precool_times)

        issues = validate_batch("T_OVERLAP", batches_df, temps_df)
        issue_types = [i.issue_type for i in issues]
        assert "stage_before_harvest_end" in issue_types or "time_overlap" in issue_types

    def test_single_stage_time_inversion(self, make_batch):
        batches_df, temps_df = make_batch("T_INV")
        sub = temps_df[temps_df["stage"] == "transport"].copy()
        times = sub["timestamp"].dt.to_pydatetime().tolist()
        times[0], times[-1] = times[-1], times[0]
        temps_df.loc[temps_df["stage"] == "transport", "timestamp"] = pd.to_datetime(times)

        issues = validate_batch("T_INV", batches_df, temps_df)
        issue_types = [i.issue_type for i in issues]
        assert "stage_time_inversion" in issue_types, "应检测到单环节内时间倒序"

    def test_stage_overlap_detection(self, make_batch):
        batches_df, temps_df = make_batch("T_OVERLAP2")
        sub = temps_df[temps_df["stage"] == "precool"].copy()
        times = sub["timestamp"].dt.to_pydatetime().tolist()
        times = [t + timedelta(hours=3) for t in times]
        temps_df.loc[temps_df["stage"] == "precool", "timestamp"] = pd.to_datetime(times)

        issues = validate_batch("T_OVERLAP2", batches_df, temps_df)
        issue_types = [i.issue_type for i in issues]
        assert "time_overlap" in issue_types or "stage_before_harvest_end" in issue_types

    def test_missing_stage_detection(self, make_batch):
        batches_df, temps_df = make_batch("T_MISS_STAGE")
        temps_df = temps_df[temps_df["stage"] != "transport"].copy()

        issues = validate_batch("T_MISS_STAGE", batches_df, temps_df)
        issue_types = [i.issue_type for i in issues]
        assert "missing_stage" in issue_types, "应检测到缺失运输环节"
        missing = [i for i in issues if i.issue_type == "missing_stage" and i.stage == "transport"]
        assert len(missing) > 0

    def test_clean_data_handles_missing_temp(self, make_batch):
        batches_df, temps_df = make_batch("T_NULL")
        temps_df.loc[0, "temperature"] = None
        temps_df.loc[1, "temperature"] = 999.0
        temps_df.loc[2, "temperature"] = -50.0

        _, cleaned, issues = clean_data(batches_df, temps_df)
        assert not cleaned["temperature"].isna().any(), "清洗后不应有缺失温度"
        assert (cleaned["temperature"] > -25).all() and (cleaned["temperature"] < 40).all()

        issue_types = [i.issue_type for i in issues]
        assert "missing_temp" in issue_types
        assert "physical_anomaly" in issue_types


class TestChainBreakDetection:
    def test_temp_over_high_break(self, make_batch):
        cat = PRODUCT_CATEGORIES["leafy_green"]
        t_max = cat.temp_max
        temps = {s: [(cat.temp_min + t_max) / 2.0] * 4 for s in STAGES}
        temps["transport"] = [2.0, t_max + 5, t_max + 5, 2.0]
        durations = {s: 2.0 for s in STAGES}

        batches_df, temps_df = make_batch("T_HIGH", temps_by_stage=temps, durations_by_stage=durations)
        _, cleaned, _ = clean_data(batches_df, temps_df)
        breaks = detect_chain_breaks(batches_df, cleaned, min_break_duration_hours=0.1)

        transport_breaks = [b for b in breaks if b.stage == "transport" and b.break_type == "temp_over_high"]
        assert len(transport_breaks) >= 1
        br = transport_breaks[0]
        assert br.duration_hours > 0.5
        assert br.max_temp > t_max
        assert br.temp_threshold_max == t_max

    def test_normal_temp_no_break(self, make_batch):
        cat = PRODUCT_CATEGORIES["leafy_green"]
        mid = (cat.temp_min + cat.temp_max) / 2.0
        temps = {s: [mid] * 4 for s in STAGES}
        durations = {s: 1.0 for s in STAGES}

        batches_df, temps_df = make_batch("T_OK2", temps_by_stage=temps, durations_by_stage=durations)
        _, cleaned, _ = clean_data(batches_df, temps_df)
        breaks = detect_chain_breaks(batches_df, cleaned, min_break_duration_hours=0.1)
        temp_breaks = [b for b in breaks if b.break_type in ("temp_over_high", "temp_over_low")]
        assert len(temp_breaks) == 0

    def test_delay_break_detection(self, make_batch):
        cat = PRODUCT_CATEGORIES["leafy_green"]
        safe_h = cat.safe_hours["distribution"]
        mid = (cat.temp_min + cat.temp_max) / 2.0
        temps = {s: [mid] * 4 for s in STAGES}
        durations = {s: 1.0 for s in STAGES}
        durations["distribution"] = safe_h * 2.0

        batches_df, temps_df = make_batch("T_DELAY2", temps_by_stage=temps, durations_by_stage=durations)
        _, cleaned, _ = clean_data(batches_df, temps_df)
        breaks = detect_chain_breaks(batches_df, cleaned, delay_ratio_threshold=1.3)

        delay_breaks = [b for b in breaks if b.break_type == "stage_delay" and b.stage == "distribution"]
        assert len(delay_breaks) >= 1
        br = delay_breaks[0]
        assert br.safe_hours == safe_h
        assert br.duration_hours >= safe_h * 1.3

    def test_cross_stage_continuous_break_split(self, make_cross_break_batch):
        batches_df, temps_df = make_cross_break_batch("T_CROSS2", break_stages=["precool", "transport"])
        _, cleaned, _ = clean_data(batches_df, temps_df)
        breaks = detect_chain_breaks(batches_df, cleaned, min_break_duration_hours=0.1)

        precool_breaks = [b for b in breaks if b.stage == "precool" and b.break_type == "temp_over_high"]
        transport_breaks = [b for b in breaks if b.stage == "transport" and b.break_type == "temp_over_high"]

        assert len(precool_breaks) >= 1, "跨环节超温应拆分为预冷环节的断链"
        assert len(transport_breaks) >= 1, "跨环节超温应拆分为运输环节的断链"
        assert len(precool_breaks) + len(transport_breaks) >= 2

        for b in precool_breaks:
            assert b.stage == "precool"
            assert b.duration_hours > 0
        for b in transport_breaks:
            assert b.stage == "transport"
            assert b.duration_hours > 0

    def test_three_stage_continuous_break_split(self, make_batch):
        cat = PRODUCT_CATEGORIES["leafy_green"]
        t_max = cat.temp_max
        high = t_max + 5.0
        mid = (cat.temp_min + t_max) / 2.0

        temps = {}
        durations = {}
        for s in STAGES:
            if s in ["precool", "transport", "warehouse"]:
                temps[s] = [high] * 5
            else:
                temps[s] = [mid] * 2
            durations[s] = 2.0

        batches_df, temps_df = make_batch("T_CROSS3", temps_by_stage=temps, durations_by_stage=durations)
        _, cleaned, _ = clean_data(batches_df, temps_df)
        breaks = detect_chain_breaks(batches_df, cleaned, min_break_duration_hours=0.1)

        for s in ["precool", "transport", "warehouse"]:
            stage_breaks = [b for b in breaks if b.stage == s and b.break_type == "temp_over_high"]
            assert len(stage_breaks) >= 1, f"跨3环节超温应在{s}环节检测到独立断链"

        breaks_df = aggregate_breaks(breaks)
        stage_groups = breaks_df.groupby("stage")["duration_hours"].sum()
        for s in ["precool", "transport", "warehouse"]:
            assert s in stage_groups.index
            assert stage_groups[s] > 1.5


class TestLossAttribution:
    def test_group_matches_detail(self):
        batches_df, temps_df = generate_sample_data(n_batches=30, seed=99)
        _, cleaned, _ = clean_data(batches_df, temps_df)
        breaks = detect_chain_breaks(batches_df, cleaned)
        metrics = attribute_loss(batches_df, cleaned, breaks)
        summary = get_stage_loss_summary(metrics)

        for _, row in summary.iterrows():
            s = row["stage"]
            col_cnt = f"{s}_break_count"
            detail_cnt = int((metrics[col_cnt] > 0).sum())
            assert detail_cnt == int(row["batches_with_break"]), f"环节{s}明细与汇总不一致"

            col_hours = f"{s}_overtemp_hours"
            detail_hours = float(metrics[col_hours].sum())
            assert abs(detail_hours - float(row["total_overtemp_hours"])) < 1e-6

    def test_factor_importance_not_empty(self):
        batches_df, temps_df = generate_sample_data(n_batches=40, seed=88)
        _, cleaned, _ = clean_data(batches_df, temps_df)
        breaks = detect_chain_breaks(batches_df, cleaned)
        metrics = attribute_loss(batches_df, cleaned, breaks)
        factors = get_factor_importance(metrics)

        assert not factors.empty
        assert "factor" in factors.columns
        assert "correlation" in factors.columns
        assert "abs_correlation" in factors.columns
        assert factors.iloc[0]["abs_correlation"] >= factors.iloc[-1]["abs_correlation"]


class TestShelfLife:
    def test_shelf_life_monotonic_with_temp(self, make_batch):
        results = []
        for test_temp in [0.0, 2.0, 5.0, 10.0, 15.0]:
            temps = {s: [test_temp] * 4 for s in STAGES}
            durations = {s: 4.0 for s in STAGES}
            batches_df, temps_df = make_batch(f"T_{int(test_temp*10)}", temps_by_stage=temps, durations_by_stage=durations)
            _, cleaned, _ = clean_data(batches_df, temps_df)
            sl = estimate_shelf_life(batches_df, cleaned)
            eat = sl.iloc[0]["effective_accumulated_hours"]
            remaining = sl.iloc[0]["remaining_hours"]
            results.append((test_temp, eat, remaining))

        for i in range(1, len(results)):
            assert results[i][1] >= results[i - 1][1], f"温度升高EAT应单调不减"
            assert results[i][2] <= results[i - 1][2], f"温度升高剩余货架期应单调不增"

    def test_high_risk_batch_identification(self, make_batch):
        high_temp = 18.0
        temps = {s: [high_temp] * 20 for s in STAGES}
        durations = {s: 24.0 for s in STAGES}
        batches_df, temps_df = make_batch("T_HIGHRISK", temps_by_stage=temps, durations_by_stage=durations)
        _, cleaned, _ = clean_data(batches_df, temps_df)
        sl = estimate_shelf_life(batches_df, cleaned)
        high_risk = identify_high_risk_batches(sl, min_risk_level="warning")

        assert len(high_risk) == 1
        assert high_risk.iloc[0]["risk_level"] in ["warning", "critical"]
        assert high_risk.iloc[0]["used_ratio"] >= 0.3

    def test_normal_temp_low_risk(self, make_batch):
        cat = PRODUCT_CATEGORIES["leafy_green"]
        mid = (cat.temp_min + cat.temp_max) / 2.0
        temps = {s: [mid] * 3 for s in STAGES}
        durations = {s: 0.5 for s in STAGES}
        batches_df, temps_df = make_batch("T_NORMAL", temps_by_stage=temps, durations_by_stage=durations)
        _, cleaned, _ = clean_data(batches_df, temps_df)
        sl = estimate_shelf_life(batches_df, cleaned)
        assert sl.iloc[0]["risk_level"] == "normal"
        assert sl.iloc[0]["used_ratio"] < 0.7


class TestDeterministic:
    def test_deterministic_output(self):
        batches_df, temps_df = generate_sample_data(n_batches=20, seed=123)
        _, cleaned1, _ = clean_data(batches_df, temps_df)
        breaks1 = detect_chain_breaks(batches_df, cleaned1)
        metrics1 = attribute_loss(batches_df, cleaned1, breaks1)
        sl1 = estimate_shelf_life(batches_df, cleaned1)

        _, cleaned2, _ = clean_data(batches_df, temps_df)
        breaks2 = detect_chain_breaks(batches_df, cleaned2)
        metrics2 = attribute_loss(batches_df, cleaned2, breaks2)
        sl2 = estimate_shelf_life(batches_df, cleaned2)

        assert len(breaks1) == len(breaks2)
        pd.testing.assert_frame_equal(metrics1, metrics2)
        pd.testing.assert_frame_equal(
            sl1.drop(columns=["harvest_time"]),
            sl2.drop(columns=["harvest_time"]),
        )


class TestBoundaryCases:
    def test_empty_temperature_data(self, make_batch):
        batches_df, _ = make_batch("T_EMPTY")
        empty_temps = pd.DataFrame(columns=["batch_id", "stage", "stage_cn", "timestamp", "temperature"])
        _, cleaned, issues = clean_data(batches_df, empty_temps)
        assert cleaned.empty
        issue_types = [i.issue_type for i in issues]
        assert "missing_stage" in issue_types

    def test_all_temps_normal_no_breaks(self, make_batch):
        cat = PRODUCT_CATEGORIES["leafy_green"]
        mid = (cat.temp_min + cat.temp_max) / 2.0
        temps = {s: [mid] * 3 for s in STAGES}
        durations = {s: min(cat.safe_hours[s] * 0.5, 2.0) for s in STAGES}

        batches_df, temps_df = make_batch("T_PERFECT", temps_by_stage=temps, durations_by_stage=durations)
        _, cleaned, _ = clean_data(batches_df, temps_df)
        breaks = detect_chain_breaks(batches_df, cleaned, delay_ratio_threshold=2.0)
        assert len(breaks) == 0, "完美数据应无断链"

    def test_min_break_duration_filter(self, make_batch):
        cat = PRODUCT_CATEGORIES["leafy_green"]
        mid = (cat.temp_min + cat.temp_max) / 2.0
        high = cat.temp_max + 3.0

        temps = {s: [mid] * 6 for s in STAGES}
        temps["transport"] = [mid, high, high, high, high, mid]
        durations = {s: 1.0 for s in STAGES}
        durations["transport"] = 0.5

        batches_df, temps_df = make_batch("T_SHORT", temps_by_stage=temps, durations_by_stage=durations)
        _, cleaned, _ = clean_data(batches_df, temps_df)

        breaks_long = detect_chain_breaks(batches_df, cleaned, min_break_duration_hours=0.01)
        breaks_short = detect_chain_breaks(batches_df, cleaned, min_break_duration_hours=1.0)

        assert len(breaks_long) > len(breaks_short), "最小持续时长阈值应过滤短断链"
        assert len(breaks_long) >= 1

    def test_below_min_temp_break(self, make_batch):
        cat = PRODUCT_CATEGORIES["berry"]
        t_min = cat.temp_min
        mid = (t_min + cat.temp_max) / 2.0
        low = t_min - 3.0

        temps = {s: [mid] * 4 for s in STAGES}
        temps["warehouse"] = [mid, low, low, mid]
        durations = {s: 2.0 for s in STAGES}

        batches_df, temps_df = make_batch("T_LOW", category="berry", temps_by_stage=temps, durations_by_stage=durations)
        _, cleaned, _ = clean_data(batches_df, temps_df)
        breaks = detect_chain_breaks(batches_df, cleaned, min_break_duration_hours=0.1)

        low_breaks = [b for b in breaks if b.break_type == "temp_over_low" and b.stage == "warehouse"]
        assert len(low_breaks) >= 1
        br = low_breaks[0]
        assert br.min_temp < t_min
        assert br.temp_threshold_min == t_min
