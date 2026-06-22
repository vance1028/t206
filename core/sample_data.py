from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

STAGES = ["harvest", "precool", "transport", "warehouse", "distribution"]
STAGE_NAMES_CN = {
    "harvest": "采收",
    "precool": "预冷入库",
    "transport": "干线运输",
    "warehouse": "冷库暂存",
    "distribution": "末端分拨",
}


@dataclass
class ProductCategory:
    name: str
    name_cn: str
    temp_min: float
    temp_max: float
    safe_hours: Dict[str, float]
    base_shelf_life_hours: float
    temp_sensitivity: float


PRODUCT_CATEGORIES: Dict[str, ProductCategory] = {
    "leafy_green": ProductCategory(
        name="leafy_green",
        name_cn="叶菜类",
        temp_min=0.0,
        temp_max=4.0,
        safe_hours={
            "harvest": 2.0,
            "precool": 4.0,
            "transport": 72.0,
            "warehouse": 168.0,
            "distribution": 6.0,
        },
        base_shelf_life_hours=168.0,
        temp_sensitivity=0.08,
    ),
    "berry": ProductCategory(
        name="berry",
        name_cn="浆果类",
        temp_min=-0.5,
        temp_max=2.0,
        safe_hours={
            "harvest": 1.5,
            "precool": 3.0,
            "transport": 48.0,
            "warehouse": 120.0,
            "distribution": 4.0,
        },
        base_shelf_life_hours=120.0,
        temp_sensitivity=0.12,
    ),
    "root": ProductCategory(
        name="root",
        name_cn="根茎类",
        temp_min=0.0,
        temp_max=8.0,
        safe_hours={
            "harvest": 4.0,
            "precool": 8.0,
            "transport": 120.0,
            "warehouse": 720.0,
            "distribution": 12.0,
        },
        base_shelf_life_hours=720.0,
        temp_sensitivity=0.03,
    ),
    "meat": ProductCategory(
        name="meat",
        name_cn="冷鲜肉",
        temp_min=-2.0,
        temp_max=4.0,
        safe_hours={
            "harvest": 1.0,
            "precool": 3.0,
            "transport": 36.0,
            "warehouse": 72.0,
            "distribution": 4.0,
        },
        base_shelf_life_hours=96.0,
        temp_sensitivity=0.15,
    ),
    "seafood": ProductCategory(
        name="seafood",
        name_cn="水产类",
        temp_min=-1.0,
        temp_max=2.0,
        safe_hours={
            "harvest": 1.0,
            "precool": 2.0,
            "transport": 24.0,
            "warehouse": 48.0,
            "distribution": 3.0,
        },
        base_shelf_life_hours=72.0,
        temp_sensitivity=0.18,
    ),
}

LINES = [
    {"origin": "云南昆明", "dest": "上海", "transit_base_hours": 30},
    {"origin": "山东寿光", "dest": "北京", "transit_base_hours": 8},
    {"origin": "广东湛江", "dest": "深圳", "transit_base_hours": 6},
    {"origin": "海南海口", "dest": "广州", "transit_base_hours": 18},
    {"origin": "辽宁大连", "dest": "沈阳", "transit_base_hours": 5},
]


def _rand_normal(mean: float, std: float, min_val: float | None = None, max_val: float | None = None) -> float:
    val = np.random.normal(mean, std)
    if min_val is not None:
        val = max(val, min_val)
    if max_val is not None:
        val = min(val, max_val)
    return float(val)


def _generate_temps_for_stage(
    category: ProductCategory,
    stage: str,
    hours: float,
    has_break: bool = False,
    break_start_frac: float = 0.3,
    break_frac: float = 0.3,
) -> List[Tuple[datetime, float]]:
    points: List[Tuple[datetime, float]] = []
    n_points = max(2, int(hours / 0.5))
    base_temp = (category.temp_min + category.temp_max) / 2.0

    for i in range(n_points):
        frac = i / (n_points - 1) if n_points > 1 else 0.0
        in_break = has_break and frac >= break_start_frac and frac < (break_start_frac + break_frac)

        if in_break:
            direction = 1 if random.random() < 0.7 else -1
            if direction > 0:
                temp = _rand_normal(category.temp_max + 3.5, 1.5, category.temp_max + 1.5, category.temp_max + 10.0)
            else:
                temp = _rand_normal(category.temp_min - 2.0, 0.8, category.temp_min - 5.0, category.temp_min - 0.5)
        else:
            temp = _rand_normal(base_temp, 0.8, category.temp_min - 0.3, category.temp_max + 0.3)

        points.append((frac, temp))

    return points


def generate_sample_data(n_batches: int = 120, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    random.seed(seed)
    np.random.seed(seed)

    batch_records: List[Dict] = []
    temp_records: List[Dict] = []

    cat_keys = list(PRODUCT_CATEGORIES.keys())

    for batch_idx in range(n_batches):
        batch_id = f"B{2026000 + batch_idx:07d}"
        category_key = random.choice(cat_keys)
        category = PRODUCT_CATEGORIES[category_key]
        line = random.choice(LINES)
        line_id = f"{line['origin']}->{line['dest']}"

        start_time = datetime(2026, 5, 1, 6, 0, 0) + timedelta(days=random.randint(0, 20))
        start_time += timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))

        batch_total_break_hours = 0.0
        batch_break_count = 0
        stage_durations: Dict[str, float] = {}
        stage_breaks: Dict[str, bool] = {}

        for stage in STAGES:
            safe_h = category.safe_hours[stage]
            if stage == "transport":
                base_h = line["transit_base_hours"]
            else:
                base_h = safe_h * 0.6

            has_delay = random.random() < 0.25
            actual_hours = base_h * (1.5 + random.random() * 1.5) if has_delay else base_h * (0.8 + random.random() * 0.6)
            if stage == "warehouse":
                actual_hours *= (0.5 + random.random() * 1.5)
            stage_durations[stage] = actual_hours

            has_break = random.random() < 0.20
            stage_breaks[stage] = has_break
            if has_break:
                batch_break_count += 1
                batch_total_break_hours += actual_hours * (0.2 + random.random() * 0.4)

        loss_rate = 0.015 + batch_total_break_hours * category.temp_sensitivity * 0.3
        loss_rate += batch_break_count * 0.01
        for s in STAGES:
            if stage_durations[s] > category.safe_hours[s] * 1.3:
                loss_rate += 0.015
        loss_rate += random.gauss(0, 0.01)
        loss_rate = max(0.005, min(0.45, loss_rate))

        batch_records.append({
            "batch_id": batch_id,
            "category": category_key,
            "category_cn": category.name_cn,
            "line_id": line_id,
            "origin": line["origin"],
            "dest": line["dest"],
            "harvest_time": start_time,
            "loss_rate": round(loss_rate, 4),
            "weight_kg": round(random.uniform(500, 3000), 1),
        })

        current_time = start_time
        for stage in STAGES:
            dur = stage_durations[stage]
            temp_points = _generate_temps_for_stage(
                category, stage, dur,
                has_break=stage_breaks[stage],
            )
            for frac, temp in temp_points:
                t = current_time + timedelta(hours=dur * frac)
                temp_records.append({
                    "batch_id": batch_id,
                    "stage": stage,
                    "stage_cn": STAGE_NAMES_CN[stage],
                    "timestamp": t,
                    "temperature": round(temp, 2),
                })
            current_time += timedelta(hours=dur)

    for i in range(int(len(temp_records) * 0.02)):
        idx = random.randint(0, len(temp_records) - 1)
        temp_records[idx]["temperature"] = round(random.uniform(-20, 40), 2)

    for i in range(int(len(temp_records) * 0.015)):
        idx = random.randint(0, len(temp_records) - 1)
        temp_records[idx]["temperature"] = None

    batches_df = pd.DataFrame(batch_records)
    temps_df = pd.DataFrame(temp_records)
    temps_df["timestamp"] = pd.to_datetime(temps_df["timestamp"])

    return batches_df, temps_df
