from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.sample_data import PRODUCT_CATEGORIES, STAGES, STAGE_NAMES_CN


def _make_simple_batch(
    batch_id: str = "T001",
    category: str = "leafy_green",
    temps_by_stage: dict | None = None,
    durations_by_stage: dict | None = None,
    harvest_time: datetime | None = None,
    loss_rate: float = 0.02,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if temps_by_stage is None:
        temps_by_stage = {s: [2.0, 2.0] for s in STAGES}
    if durations_by_stage is None:
        durations_by_stage = {s: 2.0 for s in STAGES}
    if harvest_time is None:
        harvest_time = datetime(2026, 6, 1, 8, 0, 0)

    cat = PRODUCT_CATEGORIES[category]
    batches_df = pd.DataFrame([{
        "batch_id": batch_id,
        "category": category,
        "category_cn": cat.name_cn,
        "line_id": "测试->测试",
        "origin": "测试",
        "dest": "测试",
        "harvest_time": harvest_time,
        "loss_rate": loss_rate,
        "weight_kg": 1000.0,
    }])

    temp_rows = []
    current = harvest_time
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


def _make_cross_stage_break_batch(
    batch_id: str = "T_CROSS",
    category: str = "leafy_green",
    break_stages: List[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if break_stages is None:
        break_stages = ["precool", "transport"]

    cat = PRODUCT_CATEGORIES[category]
    t_max = cat.temp_max
    mid = (cat.temp_min + cat.temp_max) / 2.0
    high_temp = t_max + 5.0

    temps = {}
    durations = {}
    for s in STAGES:
        n_points = 5
        if s in break_stages:
            temps[s] = [high_temp] * n_points
        else:
            temps[s] = [mid] * n_points
        durations[s] = 2.0

    return _make_simple_batch(batch_id, category, temps, durations)


@pytest.fixture
def sample_batch():
    return _make_simple_batch()


@pytest.fixture
def make_batch():
    return _make_simple_batch


@pytest.fixture
def make_cross_break_batch():
    return _make_cross_stage_break_batch
