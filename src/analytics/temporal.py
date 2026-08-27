"""
src/analytics/temporal.py — Temporal Behaviour + Change-Point Detection
Analyzes behaviour over time per CSE: monthly trends, trend classification,
and statistically defensible change-point detection (no deep learning).

Trends: investigation, escalation, evidence, coverage, gaps, risk proxy
Classification: IMPROVING | STABLE | DETERIORATING | SUDDEN CHANGE

Change-point: simple robust method — rolling mean deviation vs global std,
flags months where metric deviates > z_threshold (configurable, default 2.0).

All explainable; no claims of prediction unless data supports.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

CONFIG_PATH = Path("config/thresholds.yaml")

def _load_cfg() -> dict:
    try:
        if CONFIG_PATH.exists():
            cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            return cfg.get("temporal", {})
    except Exception:
        pass
    return {"z_threshold": 2.0, "min_months": 3}


def _classify_trend(values: list[float], is_inverse: bool = False) -> str:
    """
    Classify trend from time-ordered values.
    is_inverse: True if lower is better (e.g., negative_space_rate), so increase = deteriorating.
    """
    if len(values) < 3:
        return "INSUFFICIENT DATA"
    # Linear slope via least squares (simple, explainable)
    x = np.arange(len(values))
    y = np.array(values, dtype=float)
    # Handle NaN
    mask = ~np.isnan(y)
    if mask.sum() < 3:
        return "INSUFFICIENT DATA"
    slope = np.polyfit(x[mask], y[mask], 1)[0]
    # Std for sudden change check
    std = np.nanstd(y)
    recent_change = abs(y[-1] - y[0]) if len(y) >= 2 else 0

    # Sudden change if last jump > 2*std
    if len(y) >= 4 and abs(y[-1] - y[-2]) > 2 * (std if std > 0 else 1):
        return "SUDDEN CHANGE"

    # Thresholds for slope (normalized by mean)
    mean = np.nanmean(y) if np.nanmean(y) != 0 else 1
    norm_slope = slope / mean

    if is_inverse:
        # Higher gaps = deteriorating, so positive slope = deteriorating
        if norm_slope > 0.05:
            return "DETERIORATING"
        if norm_slope < -0.05:
            return "IMPROVING"
    else:
        # Higher investigation/coverage = improving, so positive slope = improving
        if norm_slope > 0.05:
            return "IMPROVING"
        if norm_slope < -0.05:
            return "DETERIORATING"
    return "STABLE"


def build_temporal_profiles(df: pd.DataFrame, freq: str = "ME") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build monthly temporal profiles per CSE.

    Args:
        df: preprocessed DataFrame with created_time + metrics
        freq: pandas offset alias (ME = month end)

    Returns:
        monthly: DataFrame with one row per cse_id+month with metrics
        trends: per-CSE trend classification per metric
        change_points: detected change points (per CSE per metric)
    """
    if df is None or df.empty or "created_time" not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = df.copy()
    df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")
    df = df.dropna(subset=["created_time"])
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df["month"] = df["created_time"].dt.to_period("M").dt.to_timestamp()

    # Ensure cols
    if "investigation_present" not in df.columns:
        df["investigation_present"] = df.get("investigation_evidence", False).fillna(False).astype(bool) if "investigation_evidence" in df.columns else False
    if "negative_space" not in df.columns:
        df["negative_space"] = False
    if "rapid_closure" not in df.columns:
        df["rapid_closure"] = False

    # Monthly aggregates per CSE
    monthly = (
        df.groupby(["cse_id", "month"])
        .agg(
            total_alerts=("alert_id", "count"),
            investigation_rate=("investigation_present", "mean"),
            escalation_rate=("escalated", "mean"),
            evidence_rate=("investigation_present", "mean"),
            avg_coverage=("activity_coverage", "mean"),
            negative_space_rate=("negative_space", "mean"),
            rapid_closure_rate=("rapid_closure", "mean"),
            avg_resolution_time=("resolution_time", "mean"),
        )
        .reset_index()
    )
    # Risk proxy: weighted combo similar to risk_scoring but simple (0-100)
    # risk_proxy = 40*negative_space_rate + 30*(1-coverage/100) + 30*rapid_rate
    monthly["risk_proxy"] = (
        monthly["negative_space_rate"] * 40
        + (1 - monthly["avg_coverage"] / 100).clip(0, 1) * 30
        + monthly["rapid_closure_rate"] * 30
    ).round(1)
    monthly["risk_proxy"] = monthly["risk_proxy"].clip(0, 100)

    # Per-CSE trend classification
    cfg = _load_cfg()
    trend_rows = []
    for cse, grp in monthly.groupby("cse_id"):
        grp = grp.sort_values("month")
        trend_rows.append({
            "cse_id": cse,
            "months_observed": len(grp),
            "investigation_trend": _classify_trend(grp["investigation_rate"].tolist(), is_inverse=False),
            "escalation_trend": _classify_trend(grp["escalation_rate"].tolist(), is_inverse=False),
            "coverage_trend": _classify_trend(grp["avg_coverage"].tolist(), is_inverse=False),
            "negative_space_trend": _classify_trend(grp["negative_space_rate"].tolist(), is_inverse=True),
            "risk_trend": _classify_trend(grp["risk_proxy"].tolist(), is_inverse=True),
            "first_risk": round(grp["risk_proxy"].iloc[0], 1) if len(grp) else np.nan,
            "last_risk": round(grp["risk_proxy"].iloc[-1], 1) if len(grp) else np.nan,
            "risk_delta": round(grp["risk_proxy"].iloc[-1] - grp["risk_proxy"].iloc[0], 1) if len(grp) >= 2 else 0,
        })
    trends = pd.DataFrame(trend_rows) if trend_rows else pd.DataFrame()

    # Classify overall status: if risk_trend DETERIORATING or SUDDEN CHANGE -> flag
    if not trends.empty:
        trends["overall_status"] = trends["risk_trend"].apply(
            lambda x: "⚠ DETERIORATING" if x == "DETERIORATING" else ("⚡ SUDDEN CHANGE" if x == "SUDDEN CHANGE" else ("↗ IMPROVING" if x == "IMPROVING" else "— STABLE"))
        )

    # Change-point detection: per CSE per metric, flag months where value deviates > z_threshold * std
    z_thr = float(cfg.get("z_threshold", 2.0))
    cp_rows = []
    for cse, grp in monthly.groupby("cse_id"):
        grp = grp.sort_values("month")
        for metric in ["total_alerts", "risk_proxy", "investigation_rate", "escalation_rate"]:
            vals = grp[metric].astype(float)
            mean = vals.mean()
            std = vals.std(ddof=0)
            if std == 0 or np.isnan(std) or len(vals) < cfg.get("min_months", 3):
                continue
            z = (vals - mean) / std
            flagged = grp[abs(z) > z_thr]
            for _, row in flagged.iterrows():
                cp_rows.append({
                    "cse_id": cse,
                    "month": row["month"],
                    "metric": metric,
                    "value": round(float(row[metric]), 3),
                    "mean": round(float(mean), 3),
                    "z_score": round(float((row[metric] - mean) / std), 2),
                    "evidence": f"{metric}={row[metric]:.2f} vs peer-month mean {mean:.2f} (z={((row[metric]-mean)/std):.2f})",
                })
    change_points = pd.DataFrame(cp_rows).sort_values(["cse_id", "month"]).reset_index(drop=True) if cp_rows else pd.DataFrame()

    return monthly, trends, change_points
