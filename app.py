from __future__ import annotations

import sys
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.sample_data import (
    PRODUCT_CATEGORIES,
    STAGES,
    STAGE_NAMES_CN,
    LINES,
    generate_sample_data,
)
from core.data_cleaning import clean_data
from core.trajectory import build_trajectory, get_stage_timeline
from core.chain_break import detect_chain_breaks, aggregate_breaks
from core.loss_attribution import attribute_loss, get_stage_loss_summary, get_factor_importance
from core.shelf_life import (
    calculate_effective_accumulated_temp,
    estimate_shelf_life,
    identify_high_risk_batches,
)


BREAK_TYPE_CN = {
    "temp_over_high": "温度偏高",
    "temp_over_low": "温度偏低",
    "stage_delay": "滞留超时",
}

RISK_COLOR = {"normal": "#22c55e", "warning": "#f59e0b", "critical": "#ef4444"}
RISK_CN = {"normal": "正常", "warning": "临期预警", "critical": "高危"}

STAGE_COLORS = px.colors.qualitative.Set2[: len(STAGES)]


@st.cache_data(show_spinner=False)
def load_data(n_batches: int, seed: int):
    return generate_sample_data(n_batches=n_batches, seed=seed)


@st.cache_data(show_spinner=False)
def run_pipeline(
    _batches_df: pd.DataFrame,
    _temps_df: pd.DataFrame,
    temp_min_override: dict | None,
    temp_max_override: dict | None,
    safe_hours_override: dict | None,
    min_break_duration_hours: float,
    delay_ratio_threshold: float,
):
    cleaned_batches, cleaned_temps, issues = clean_data(_batches_df, _temps_df)
    breaks = detect_chain_breaks(
        cleaned_batches,
        cleaned_temps,
        temp_min_override=temp_min_override,
        temp_max_override=temp_max_override,
        safe_hours_override=safe_hours_override,
        min_break_duration_hours=min_break_duration_hours,
        delay_ratio_threshold=delay_ratio_threshold,
    )
    breaks_df = aggregate_breaks(breaks)
    metrics = attribute_loss(cleaned_batches, cleaned_temps, breaks)
    stage_summary = get_stage_loss_summary(metrics)
    factor_importance = get_factor_importance(metrics)
    shelf_life = estimate_shelf_life(cleaned_batches, cleaned_temps)
    high_risk = identify_high_risk_batches(shelf_life)
    return cleaned_batches, cleaned_temps, issues, breaks, breaks_df, metrics, stage_summary, factor_importance, shelf_life, high_risk


def _filter_data(
    batches_df,
    temps_df,
    breaks_df,
    metrics_df,
    shelf_life_df,
    sel_categories,
    sel_lines,
    sel_risk,
    date_range,
):
    mask_cat = batches_df["category_cn"].isin(sel_categories)
    mask_line = batches_df["line_id"].isin(sel_lines)
    harvest_ts = pd.to_datetime(batches_df["harvest_time"])
    mask_date = (harvest_ts >= pd.Timestamp(date_range[0])) & (harvest_ts <= pd.Timestamp(date_range[1]) + timedelta(days=1))
    batch_mask = mask_cat & mask_line & mask_date

    if sel_risk and "全部" not in sel_risk:
        sl = shelf_life_df[["batch_id", "risk_level"]].copy()
        sl_mask = sl["risk_level"].isin(sel_risk)
        sl_bids = set(sl.loc[sl_mask, "batch_id"])
        batch_mask = batch_mask & batches_df["batch_id"].isin(sl_bids)

    sel_batches = batches_df[batch_mask].reset_index(drop=True)
    bids = set(sel_batches["batch_id"])
    sel_temps = temps_df[temps_df["batch_id"].isin(bids)].reset_index(drop=True)
    sel_breaks = breaks_df[breaks_df["batch_id"].isin(bids)].reset_index(drop=True) if not breaks_df.empty else breaks_df
    sel_metrics = metrics_df[metrics_df["batch_id"].isin(bids)].reset_index(drop=True)
    sel_sl = shelf_life_df[shelf_life_df["batch_id"].isin(bids)].reset_index(drop=True)
    return sel_batches, sel_temps, sel_breaks, sel_metrics, sel_sl


def build_sidebar():
    st.sidebar.header("⚙️ 分析参数")

    with st.sidebar.expander("📦 数据生成", expanded=False):
        n_batches = st.slider("模拟批次数量", 30, 300, 120, 10)
        seed = st.number_input("随机种子", 0, 9999, 42, 1)

    with st.sidebar.expander("🎚️ 温度阈值调整", expanded=True):
        temp_overrides_min = {}
        temp_overrides_max = {}
        safe_overrides = {}
        for key, cat in PRODUCT_CATEGORIES.items():
            st.markdown(f"**{cat.name_cn}** ({cat.temp_min}~{cat.temp_max}°C)")
            c1, c2 = st.columns(2)
            with c1:
                tmin = st.number_input(
                    f"下限(°C)-{cat.name_cn}",
                    float(cat.temp_min - 3),
                    float(cat.temp_max),
                    float(cat.temp_min),
                    0.5,
                    key=f"tmin_{key}",
                )
            with c2:
                tmax = st.number_input(
                    f"上限(°C)-{cat.name_cn}",
                    float(cat.temp_min),
                    float(cat.temp_max + 5),
                    float(cat.temp_max),
                    0.5,
                    key=f"tmax_{key}",
                )
            temp_overrides_min[key] = float(tmin)
            temp_overrides_max[key] = float(tmax)

    with st.sidebar.expander("⏱️ 断链判定阈值", expanded=False):
        min_break_dur = st.slider("最小断链持续(小时)", 0.05, 2.0, 0.25, 0.05)
        delay_ratio = st.slider("滞留超时倍数阈值", 1.1, 2.0, 1.3, 0.05)

    st.sidebar.divider()
    st.sidebar.header("🔍 筛选条件")

    all_cats = [c.name_cn for c in PRODUCT_CATEGORIES.values()]
    sel_categories = st.sidebar.multiselect("生鲜品类", all_cats, default=all_cats)

    all_lines = [f"{l['origin']}->{l['dest']}" for l in LINES]
    sel_lines = st.sidebar.multiselect("运输线路", all_lines, default=all_lines)

    sel_risk = st.sidebar.multiselect("货架期风险等级", ["normal", "warning", "critical"], default=["normal", "warning", "critical"], format_func=lambda x: RISK_CN.get(x, x))

    return (
        n_batches,
        seed,
        temp_overrides_min,
        temp_overrides_max,
        None,
        min_break_dur,
        delay_ratio,
        sel_categories,
        sel_lines,
        sel_risk,
    )


def plot_stage_break_frequency(breaks_df: pd.DataFrame, stage_summary: pd.DataFrame):
    st.subheader("各环节断链频次与超温时长分布")

    if breaks_df.empty:
        st.info("未检测到断链事件")
        return

    col1, col2 = st.columns(2)

    with col1:
        stage_counts = breaks_df.groupby(["stage_cn", "break_type"]).size().reset_index(name="count")
        stage_counts["break_type_cn"] = stage_counts["break_type"].map(BREAK_TYPE_CN).fillna(stage_counts["break_type"])
        fig1 = px.bar(
            stage_counts,
            x="stage_cn",
            y="count",
            color="break_type_cn",
            barmode="stack",
            title="各环节断链事件次数",
            labels={"stage_cn": "环节", "count": "断链次数", "break_type_cn": "断链类型"},
            color_discrete_sequence=["#ef4444", "#3b82f6", "#f59e0b"],
        )
        fig1.update_layout(height=420)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        temp_breaks = breaks_df[breaks_df["break_type"].isin(["temp_over_high", "temp_over_low"])]
        if not temp_breaks.empty:
            stage_hours = temp_breaks.groupby("stage_cn")["duration_hours"].sum().reset_index()
            fig2 = px.bar(
                stage_hours,
                x="stage_cn",
                y="duration_hours",
                color="stage_cn",
                title="各环节累计超温时长(小时)",
                labels={"stage_cn": "环节", "duration_hours": "累计超温时长(小时)"},
                color_discrete_sequence=STAGE_COLORS,
            )
            fig2.update_layout(height=420, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("暂无温度超标断链")

    if not stage_summary.empty:
        st.markdown("#### 各环节损耗对比（有/无断链）")
        show_df = stage_summary[[
            "stage_cn", "batch_count", "batches_with_break", "break_rate",
            "avg_loss_no_break", "avg_loss_with_break", "loss_delta",
        ]].copy()
        show_df.columns = [
            "环节", "总批次", "有断链批次", "断链率",
            "无损耗均值", "有断链损耗均值", "损耗提升",
        ]
        show_df["断链率"] = (show_df["断链率"] * 100).round(1).astype(str) + "%"
        for c in ["无损耗均值", "有断链损耗均值", "损耗提升"]:
            show_df[c] = (show_df[c] * 100).round(2).astype(str) + "%"
        st.dataframe(show_df, use_container_width=True, hide_index=True)


def plot_loss_vs_break(metrics_df: pd.DataFrame):
    st.subheader("损耗率随断链程度变化")

    if metrics_df.empty:
        return

    col1, col2 = st.columns(2)

    with col1:
        d1 = metrics_df.copy()
        d1["断链次数分组"] = pd.cut(
            d1["break_count"],
            bins=[-0.5, 0.5, 1.5, 2.5, 5.5, 100],
            labels=["0次", "1次", "2次", "3-5次", "6次以上"],
        )
        agg = d1.groupby("断链次数分组")["loss_rate"].agg(["mean", "count"]).reset_index()
        agg["mean"] = (agg["mean"] * 100).round(2)
        fig = px.bar(
            agg, x="断链次数分组", y="mean",
            text="mean",
            title="平均损耗率 vs 断链次数",
            labels={"mean": "平均损耗率(%)", "count": "批次数"},
            color="mean",
            color_continuous_scale="Reds",
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(height=400, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        d2 = metrics_df.copy()
        d2["累计超温时长(小时)"] = d2["total_overtemp_hours"].clip(upper=20)
        fig = px.scatter(
            d2, x="累计超温时长(小时)", y="loss_rate",
            color="category_cn",
            size="break_count",
            size_max=15,
            opacity=0.75,
            title="损耗率 vs 累计超温时长",
            labels={"loss_rate": "损耗率", "category_cn": "品类"},
        )
        fig.update_yaxes(tickformat=".1%")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def plot_group_comparison(batches_df: pd.DataFrame, metrics_df: pd.DataFrame):
    st.subheader("按品类 / 环节 / 线路的损耗对比")

    if metrics_df.empty:
        return

    tab1, tab2, tab3 = st.tabs(["按品类", "按环节", "按线路"])

    with tab1:
        by_cat = metrics_df.groupby("category_cn").agg(
            批次=("batch_id", "count"),
            平均损耗率=("loss_rate", "mean"),
            平均断链次数=("break_count", "mean"),
            平均超温时长=("total_overtemp_hours", "mean"),
        ).reset_index()
        by_cat["平均损耗率"] = (by_cat["平均损耗率"] * 100).round(2)
        by_cat["平均断链次数"] = by_cat["平均断链次数"].round(2)
        by_cat["平均超温时长"] = by_cat["平均超温时长"].round(2)
        fig = px.bar(
            by_cat.sort_values("平均损耗率", ascending=False),
            x="category_cn", y="平均损耗率",
            text="平均损耗率",
            color="平均断链次数",
            color_continuous_scale="OrRd",
            title="各品类平均损耗率 (颜色=平均断链次数)",
            labels={"category_cn": "品类"},
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(by_cat.rename(columns={"category_cn": "品类"}), use_container_width=True, hide_index=True)

    with tab2:
        rows = []
        for s in STAGES:
            col_cnt = f"{s}_break_count"
            has_b = metrics_df[col_cnt] > 0
            rows.append({
                "环节": STAGE_NAMES_CN[s],
                "断链批次占比": float(has_b.mean()) if len(metrics_df) else 0.0,
                "有断链平均损耗": float(metrics_df.loc[has_b, "loss_rate"].mean()) if has_b.any() else 0.0,
                "无断链平均损耗": float(metrics_df.loc[~has_b, "loss_rate"].mean()) if (~has_b).any() else 0.0,
            })
        stage_df = pd.DataFrame(rows)
        for c in ["断链批次占比", "有断链平均损耗", "无断链平均损耗"]:
            stage_df[c] = (stage_df[c] * 100).round(2)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=stage_df["环节"], y=stage_df["无断链平均损耗"],
            name="无断链", marker_color="#22c55e",
        ))
        fig.add_trace(go.Bar(
            x=stage_df["环节"], y=stage_df["有断链平均损耗"],
            name="有断链", marker_color="#ef4444",
        ))
        fig.update_layout(
            barmode="group",
            title="各环节有/无断链的平均损耗率(%)",
            yaxis_title="平均损耗率(%)",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(stage_df, use_container_width=True, hide_index=True)

    with tab3:
        by_line = metrics_df.groupby("line_id").agg(
            批次=("batch_id", "count"),
            平均损耗率=("loss_rate", "mean"),
            平均断链次数=("break_count", "mean"),
        ).reset_index()
        by_line["平均损耗率"] = (by_line["平均损耗率"] * 100).round(2)
        by_line["平均断链次数"] = by_line["平均断链次数"].round(2)
        by_line = by_line.sort_values("平均损耗率", ascending=False)
        fig = px.bar(
            by_line, x="line_id", y="平均损耗率",
            text="平均损耗率",
            color="平均断链次数",
            color_continuous_scale="RdPu",
            title="各线路平均损耗率",
            labels={"line_id": "线路"},
        )
        fig.update_traces(texttemplate="%{text}%", textposition="outside")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(by_line.rename(columns={"line_id": "线路"}), use_container_width=True, hide_index=True)


def plot_factor_importance(factor_df: pd.DataFrame):
    st.subheader("损耗影响因子重要性")
    if factor_df.empty:
        return
    top = factor_df.head(10).copy()
    top["相关性"] = top["correlation"]
    fig = px.bar(
        top.sort_values("abs_correlation", ascending=True),
        y="factor_name", x="相关性",
        color="abs_correlation",
        color_continuous_scale="Blues",
        orientation="h",
        title="Top 10 影响因子 (按相关系数绝对值)",
        labels={"factor_name": "影响因子"},
    )
    fig.update_layout(height=480, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def plot_high_risk(high_risk_df: pd.DataFrame, shelf_life_df: pd.DataFrame):
    st.subheader("临期 / 高风险批次清单")

    if shelf_life_df.empty:
        st.info("暂无数据")
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        risk_cnt = shelf_life_df["risk_level"].value_counts().reset_index()
        risk_cnt.columns = ["风险等级", "数量"]
        risk_cnt["风险等级"] = risk_cnt["风险等级"].map(RISK_CN)
        fig = px.pie(
            risk_cnt, names="风险等级", values="数量",
            color="风险等级",
            color_discrete_map={RISK_CN[k]: v for k, v in RISK_COLOR.items()},
            title="货架期风险等级分布",
        )
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        show = high_risk_df[[
            "batch_id", "category_cn", "line_id",
            "base_shelf_life_hours", "effective_accumulated_hours",
            "remaining_hours", "used_ratio", "risk_level", "loss_rate",
        ]].copy()
        show.columns = [
            "批次号", "品类", "线路",
            "基础货架期(h)", "已消耗等效(h)",
            "剩余等效(h)", "消耗比例", "风险等级", "损耗率",
        ]
        show["消耗比例"] = (show["消耗比例"] * 100).round(1).astype(str) + "%"
        show["损耗率"] = (show["损耗率"] * 100).round(2).astype(str) + "%"
        show["风险等级"] = show["风险等级"].map(RISK_CN)

        def _style(row):
            color = RISK_COLOR.get(row.get("风险等级", ""), "#ffffff")
            return [f"background-color: {color}22"] * len(row)

        st.markdown(f"**共 {len(show)} 个预警批次** (含临期+高危)")
        if not show.empty:
            st.dataframe(
                show.style.apply(_style, axis=1),
                use_container_width=True,
                hide_index=True,
                height=320,
            )
        else:
            st.info("当前筛选范围内暂无预警批次 🎉")


def plot_single_batch(
    batch_id: str,
    batches_df: pd.DataFrame,
    cleaned_temps_df: pd.DataFrame,
    breaks_df: pd.DataFrame,
):
    st.subheader(f"批次 {batch_id} 完整温度轨迹与断链标注")

    batch_row = batches_df[batches_df["batch_id"] == batch_id]
    if batch_row.empty:
        st.warning("找不到该批次")
        return

    row = batch_row.iloc[0]
    category = PRODUCT_CATEGORIES.get(row["category"])
    t_min = category.temp_min if category else 0
    t_max = category.temp_max if category else 5

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("品类", row["category_cn"])
    col_b.metric("线路", row["line_id"])
    col_c.metric("损耗率", f"{row['loss_rate']*100:.2f}%")
    col_d.metric("重量(kg)", f"{row['weight_kg']:.0f}")

    trajectory = build_trajectory(batch_id, batches_df, cleaned_temps_df)
    if trajectory.empty:
        st.warning("该批次无温度轨迹数据")
        return

    timeline = get_stage_timeline(batch_id, cleaned_temps_df)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("温度时间轨迹 & 断链标注", "各环节时长"),
        row_heights=[0.7, 0.3],
    )

    stage_colors = dict(zip(STAGES, STAGE_COLORS))
    for s in STAGES:
        sub = trajectory[trajectory["stage"] == s]
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["timestamp"], y=sub["temperature"],
                mode="lines+markers",
                name=STAGE_NAMES_CN[s],
                line=dict(color=stage_colors.get(s, "#666666"), width=2.5),
                marker=dict(size=4),
            ),
            row=1, col=1,
        )

    fig.add_hline(
        y=t_max, line_dash="dash", line_color="#ef4444",
        annotation_text=f"上限 {t_max}°C",
        annotation_position="top right",
        row=1, col=1,
    )
    fig.add_hline(
        y=t_min, line_dash="dash", line_color="#3b82f6",
        annotation_text=f"下限 {t_min}°C",
        annotation_position="bottom right",
        row=1, col=1,
    )

    batch_breaks = breaks_df[breaks_df["batch_id"] == batch_id]
    if not batch_breaks.empty:
        for _, br in batch_breaks.iterrows():
            color = "#ef4444" if br["break_type"] == "temp_over_high" else (
                "#3b82f6" if br["break_type"] == "temp_over_low" else "#f59e0b"
            )
            label = BREAK_TYPE_CN.get(br["break_type"], br["break_type"])
            fig.add_vrect(
                x0=br["start_time"], x1=br["end_time"],
                fillcolor=color, opacity=0.18,
                line_width=0,
                annotation_text=f"{label} {br['duration_hours']:.1f}h",
                annotation_position="top left",
                annotation_font_size=10,
                row=1, col=1,
            )

    if not timeline.empty:
        for _, tl in timeline.iterrows():
            fig.add_trace(
                go.Bar(
                    x=[tl["start_time"] + (tl["end_time"] - tl["start_time"]) / 2],
                    y=[tl["duration_hours"]],
                    width=(tl["end_time"] - tl["start_time"]).total_seconds() * 1000 * 0.9,
                    name=f"{tl['stage_cn']}",
                    marker_color=stage_colors.get(tl["stage"], "#888"),
                    showlegend=False,
                    text=f"{tl['stage_cn']}<br>{tl['duration_hours']:.1f}h",
                    textposition="inside",
                    insidetextanchor="middle",
                ),
                row=2, col=1,
            )

    fig.update_layout(
        height=600,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="温度(°C)", row=1, col=1)
    fig.update_yaxes(title_text="时长(小时)", row=2, col=1)
    fig.update_xaxes(title_text="时间", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

    if not batch_breaks.empty:
        st.markdown("##### 检测到的断链事件")
        show_br = batch_breaks[[
            "stage_cn", "break_type", "start_time", "end_time", "duration_hours",
            "max_temp", "min_temp", "avg_temp", "severity_score",
        ]].copy()
        show_br.columns = [
            "环节", "断链类型", "开始时间", "结束时间", "持续(h)",
            "最高温", "最低温", "平均温", "严重度",
        ]
        show_br["断链类型"] = show_br["断链类型"].map(BREAK_TYPE_CN).fillna(show_br["断链类型"])
        st.dataframe(show_br, use_container_width=True, hide_index=True)


def main():
    st.set_page_config(page_title="生鲜冷链断链分析平台", layout="wide", page_icon="🥬")
    st.title("🥬 生鲜冷链断链分析平台")
    st.caption("从产地到仓到配送的全链路温度断链与损耗归因分析")

    params = build_sidebar()
    (
        n_batches, seed,
        tmin_override, tmax_override, safe_override,
        min_break_dur, delay_ratio,
        sel_categories, sel_lines, sel_risk,
    ) = params

    batches_df, temps_df = load_data(n_batches, seed)

    date_min = pd.Timestamp(batches_df["harvest_time"].min()).date()
    date_max = pd.Timestamp(batches_df["harvest_time"].max()).date()
    with st.sidebar:
        st.divider()
        date_range = st.date_input("采收日期范围", (date_min, date_max), min_value=date_min, max_value=date_max)
        if len(date_range) != 2:
            date_range = (date_min, date_max)

    with st.spinner("正在执行全链路分析..."):
        cleaned_batches, cleaned_temps, issues, breaks, breaks_df, metrics, stage_summary, factor_importance, shelf_life, high_risk = run_pipeline(
            batches_df, temps_df,
            tmin_override, tmax_override, safe_override,
            min_break_dur, delay_ratio,
        )

    sel_batches, sel_temps, sel_breaks, sel_metrics, sel_sl = _filter_data(
        cleaned_batches, cleaned_temps, breaks_df, metrics, shelf_life,
        sel_categories, sel_lines, sel_risk, date_range,
    )

    sel_stage_summary = get_stage_loss_summary(sel_metrics) if not sel_metrics.empty else pd.DataFrame()
    sel_factor = get_factor_importance(sel_metrics) if not sel_metrics.empty else pd.DataFrame()
    sel_high_risk = identify_high_risk_batches(sel_sl) if not sel_sl.empty else pd.DataFrame()

    with st.expander("📊 总览指标", expanded=True):
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("筛选批次", len(sel_batches))
        k2.metric("断链事件", len(sel_breaks))
        avg_loss = sel_batches["loss_rate"].mean() if len(sel_batches) else 0
        k3.metric("平均损耗率", f"{avg_loss*100:.2f}%")
        k4.metric("预警批次", len(sel_high_risk))
        k5.metric("数据问题", len(issues))

    tab_overview, tab_loss, tab_risk, tab_detail = st.tabs([
        "🔍 断链概览",
        "📈 损耗归因",
        "⚠️ 货架期风险",
        "📋 单批次详情",
    ])

    with tab_overview:
        plot_stage_break_frequency(sel_breaks, sel_stage_summary)
        if issues:
            with st.expander(f"⚠️ 检测到 {len(issues)} 个数据问题 (点击展开)", expanded=False):
                for iss in issues[:50]:
                    icon = "🔴" if iss.severity == "error" else "🟡"
                    st.markdown(f"- {icon} **[{iss.issue_type}]** {iss.batch_id} {('· ' + iss.stage_cn) if iss.stage else ''} — {iss.description}")
                if len(issues) > 50:
                    st.caption(f"... 另有 {len(issues) - 50} 条已省略")

    with tab_loss:
        plot_loss_vs_break(sel_metrics)
        plot_group_comparison(sel_batches, sel_metrics)
        plot_factor_importance(sel_factor)

    with tab_risk:
        plot_high_risk(sel_high_risk, sel_sl)

    with tab_detail:
        if len(sel_batches) > 0:
            default_bid = sel_high_risk.iloc[0]["batch_id"] if not sel_high_risk.empty else sel_batches.iloc[0]["batch_id"]
            batch_id = st.selectbox(
                "选择批次号查看完整轨迹",
                options=list(sel_batches["batch_id"]),
                index=list(sel_batches["batch_id"]).index(default_bid) if default_bid in list(sel_batches["batch_id"]) else 0,
            )
            if batch_id:
                plot_single_batch(batch_id, sel_batches, sel_temps, sel_breaks)
        else:
            st.info("当前筛选条件下无批次数据")


if __name__ == "__main__":
    main()
