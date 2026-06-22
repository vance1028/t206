from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sample_data import PRODUCT_CATEGORIES, STAGES, STAGE_NAMES_CN, generate_sample_data
from core.data_cleaning import clean_data, validate_batch, DataIssue
from core.trajectory import build_trajectory, get_stage_timeline
from core.chain_break import detect_chain_breaks, aggregate_breaks, ChainBreak
from core.loss_attribution import attribute_loss, get_stage_loss_summary, get_factor_importance
from core.shelf_life import (
    calculate_effective_accumulated_temp,
    estimate_shelf_life,
    identify_high_risk_batches,
)


def _make_simple_batch(
    batch_id: str = "T001",
    category: str = "leafy_green",
    temps_by_stage: dict | None = None,
    durations_by_stage: dict | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if temps_by_stage is None:
        temps_by_stage = {s: [2.0, 2.0] for s in STAGES}
    if durations_by_stage is None:
        durations_by_stage = {s: 2.0 for s in STAGES}

    cat = PRODUCT_CATEGORIES[category]
    batches_df = pd.DataFrame([{
        "batch_id": batch_id,
        "category": category,
        "category_cn": cat.name_cn,
        "line_id": "测试->测试",
        "origin": "测试",
        "dest": "测试",
        "harvest_time": datetime(2026, 6, 1, 8, 0, 0),
        "loss_rate": 0.02,
        "weight_kg": 1000.0,
    }])

    temp_rows = []
    current = datetime(2026, 6, 1, 8, 0, 0)
    for s in STAGES:
        dur = durations_by_stage[s]
        temps = temps_by_stage[s]
        n = len(temps)
        for i, t_val in enumerate(temps):
            t = current + timedelta(hours=dur * (i / max(1, n - 1)))
            temp_rows.append({
                "batch_id": batch_id,
                "stage": s,
                "stage_cn": STAGE_NAMES_CN[s],
                "timestamp": t,
                "temperature": t_val,
            })
        current += timedelta(hours=dur)

    temps_df = pd.DataFrame(temp_rows)
    temps_df["timestamp"] = pd.to_datetime(temps_df["timestamp"])
    return batches_df, temps_df


def test_temp_break_detection_basic() -> bool:
    cat = PRODUCT_CATEGORIES["leafy_green"]
    t_min, t_max = cat.temp_min, cat.temp_max

    temps = {s: [(t_min + t_max) / 2.0] * 4 for s in STAGES}
    temps["transport"] = [2.0, 10.0, 10.0, 2.0]
    durations = {s: 1.0 for s in STAGES}
    durations["transport"] = 2.0

    batches_df, temps_df = _make_simple_batch("T_BREAK", "leafy_green", temps, durations)
    _, cleaned, _ = clean_data(batches_df, temps_df)
    breaks = detect_chain_breaks(batches_df, cleaned, min_break_duration_hours=0.1)

    transport_breaks = [b for b in breaks if b.stage == "transport" and b.break_type == "temp_over_high"]
    assert len(transport_breaks) >= 1, f"应在运输环节检测到高温断链, 实际 {len(transport_breaks)} 个"
    br = transport_breaks[0]
    assert br.duration_hours > 0.5, f"断链时长应>0.5h, 实际 {br.duration_hours}"
    assert br.max_temp > t_max, f"最高温度应超过上限 {t_max}, 实际 {br.max_temp}"
    assert br.temp_threshold_max == t_max
    print("  [PASS] 超温判定与持续时长正确")
    return True


def test_temp_break_no_false_positive() -> bool:
    cat = PRODUCT_CATEGORIES["leafy_green"]
    mid = (cat.temp_min + cat.temp_max) / 2.0
    temps = {s: [mid] * 4 for s in STAGES}
    durations = {s: 1.0 for s in STAGES}

    batches_df, temps_df = _make_simple_batch("T_OK", "leafy_green", temps, durations)
    _, cleaned, _ = clean_data(batches_df, temps_df)
    breaks = detect_chain_breaks(batches_df, cleaned, min_break_duration_hours=0.1)
    temp_breaks = [b for b in breaks if b.break_type in ("temp_over_high", "temp_over_low")]
    assert len(temp_breaks) == 0, f"正常温度不应检测到断链, 实际 {len(temp_breaks)} 个"
    print("  [PASS] 正常温度无误报")
    return True


def test_delay_break_detection() -> bool:
    cat = PRODUCT_CATEGORIES["leafy_green"]
    safe_h = cat.safe_hours["distribution"]
    mid = (cat.temp_min + cat.temp_max) / 2.0
    temps = {s: [mid] * 4 for s in STAGES}
    durations = {s: 1.0 for s in STAGES}
    durations["distribution"] = safe_h * 2.0

    batches_df, temps_df = _make_simple_batch("T_DELAY", "leafy_green", temps, durations)
    _, cleaned, _ = clean_data(batches_df, temps_df)
    breaks = detect_chain_breaks(batches_df, cleaned, delay_ratio_threshold=1.3)

    delay_breaks = [b for b in breaks if b.break_type == "stage_delay" and b.stage == "distribution"]
    assert len(delay_breaks) >= 1, "应检测到分拨环节滞留断链"
    br = delay_breaks[0]
    assert br.safe_hours == safe_h
    assert br.duration_hours >= safe_h * 1.3
    print("  [PASS] 滞留超时断链识别正确")
    return True


def test_loss_attribution_group_matches_detail() -> bool:
    batches_df, temps_df = generate_sample_data(n_batches=30, seed=99)
    _, cleaned, _ = clean_data(batches_df, temps_df)
    breaks = detect_chain_breaks(batches_df, cleaned)
    metrics = attribute_loss(batches_df, cleaned, breaks)
    summary = get_stage_loss_summary(metrics)

    for _, row in summary.iterrows():
        s = row["stage"]
        col_cnt = f"{s}_break_count"
        detail_with_break = (metrics[col_cnt] > 0).sum()
        assert int(detail_with_break) == int(row["batches_with_break"]), (
            f"环节 {s} 明细有断链批次={detail_with_break} vs 汇总={row['batches_with_break']}"
        )

        detail_total = metrics[f"{s}_overtemp_hours"].sum() if f"{s}_overtemp_hours" in metrics.columns else 0.0
        assert abs(float(detail_total) - float(row["total_overtemp_hours"])) < 1e-6, (
            f"环节 {s} 明细累计超温={detail_total} vs 汇总={row['total_overtemp_hours']}"
        )
    print("  [PASS] 损耗归因分组统计与明细一致")
    return True


def test_shelf_life_monotonic_with_temp() -> bool:
    results = []
    for test_temp in [0.0, 2.0, 5.0, 10.0, 15.0]:
        temps = {s: [test_temp] * 4 for s in STAGES}
        durations = {s: 4.0 for s in STAGES}
        batches_df, temps_df = _make_simple_batch(f"T_{int(test_temp*10)}", "leafy_green", temps, durations)
        _, cleaned, _ = clean_data(batches_df, temps_df)
        sl = estimate_shelf_life(batches_df, cleaned)
        eat = sl.iloc[0]["effective_accumulated_hours"]
        remaining = sl.iloc[0]["remaining_hours"]
        results.append((test_temp, eat, remaining))

    for i in range(1, len(results)):
        prev_t, prev_eat, prev_rem = results[i - 1]
        cur_t, cur_eat, cur_rem = results[i]
        assert cur_eat >= prev_eat, f"温度升高时EAT应单调不减: {prev_t}->{cur_t}, EAT {prev_eat}->{cur_eat}"
        assert cur_rem <= prev_rem, f"温度升高时剩余货架期应单调不增: {prev_t}->{cur_t}, 剩余 {prev_rem}->{cur_rem}"

    strict_increase = any(results[i][1] > results[i - 1][1] for i in range(1, len(results)))
    assert strict_increase, "至少应有一组温度升高导致EAT严格增加"
    print("  [PASS] 货架期随温度暴露单调缩短")
    return True


def test_deterministic_output() -> bool:
    batches_df, temps_df = generate_sample_data(n_batches=20, seed=123)
    _, cleaned1, _ = clean_data(batches_df, temps_df)
    breaks1 = detect_chain_breaks(batches_df, cleaned1)
    metrics1 = attribute_loss(batches_df, cleaned1, breaks1)
    sl1 = estimate_shelf_life(batches_df, cleaned1)

    _, cleaned2, _ = clean_data(batches_df, temps_df)
    breaks2 = detect_chain_breaks(batches_df, cleaned2)
    metrics2 = attribute_loss(batches_df, cleaned2, breaks2)
    sl2 = estimate_shelf_life(batches_df, cleaned2)

    assert len(breaks1) == len(breaks2), f"断链数量应一致: {len(breaks1)} vs {len(breaks2)}"
    pd.testing.assert_frame_equal(metrics1, metrics2)
    pd.testing.assert_frame_equal(sl1.drop(columns=["harvest_time"]), sl2.drop(columns=["harvest_time"]))
    print("  [PASS] 同一输入结果一致（确定性）")
    return True


def test_data_cleaning_handles_issues() -> bool:
    batches_df, temps_df = _make_simple_batch("T_DIRTY", "leafy_green")
    temps_df.loc[0, "temperature"] = None
    temps_df.loc[1, "temperature"] = 999.0
    temps_df.loc[2, "temperature"] = -50.0

    _, cleaned, issues = clean_data(batches_df, temps_df)

    assert not cleaned["temperature"].isna().any(), "清洗后不应有缺失温度"
    assert (cleaned["temperature"] > -25).all() and (cleaned["temperature"] < 40).all(), "清洗后温度应在物理范围内"

    issue_types = [i.issue_type for i in issues]
    assert "missing_temp" in issue_types, "应检测到缺失温度"
    assert "physical_anomaly" in issue_types, "应检测到物理异常"
    print("  [PASS] 脏数据识别与清洗正确")
    return True


def run_all_tests():
    print("=" * 60)
    print("生鲜冷链断链分析平台 - 内核逻辑验证")
    print("=" * 60)

    tests = [
        test_temp_break_detection_basic,
        test_temp_break_no_false_positive,
        test_delay_break_detection,
        test_loss_attribution_group_matches_detail,
        test_shelf_life_monotonic_with_temp,
        test_deterministic_output,
        test_data_cleaning_handles_issues,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  [FAIL] {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"  [ERROR] {t.__name__}: {type(e).__name__}: {e}")

    print("=" * 60)
    print(f"结果: {passed} 通过 / {failed} 失败 / 共 {len(tests)} 项")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    ok = run_all_tests()
    sys.exit(0 if ok else 1)
