"""
src/analytics/risk_engine.py — Advanced Supervisory Risk Intelligence
Transparent multi-factor model, offline, configurable.

Components (weights from config/thresholds.yaml risk_scoring.weights):
  - rule_violations (30%)  — normalized total_rule_score
  - execution_gaps   (25%) — execution_gap_rate + missing workflow
  - anomaly          (20%) — IsolationForest anomaly_score normalized
  - peer_deviation   (15%) — count of peer needs_attention vs max
  - criticality      (10%) — critical-asset gaps + low coverage

All components 0-100, weighted sum 0-100, bands from thresholds.yaml.
Shows exactly how score produced.

API: calculate_advanced_risk(df_flagged, rule_summary, behavioural, peer_per_cse, anomaly_df, temporal_trends) -> risk_df
Legacy shim: calculate_risk_scores(df) remains available for backward compat.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import yaml
from pathlib import Path

CONFIG_PATH = Path("config/thresholds.yaml")

def _load_cfg() -> tuple[dict, dict]:
    weights = {"rule_violations": 0.30, "execution_gaps": 0.25, "anomaly": 0.20, "peer_deviation": 0.15, "criticality": 0.10}
    bands = {"critical": 80, "high": 60, "moderate": 30}
    try:
        if CONFIG_PATH.exists():
            cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            w = cfg.get("risk_scoring", {}).get("weights", {})
            if w:
                weights = {k: float(v) for k, v in w.items()}
                # Normalize if sum !=1
                s = sum(weights.values())
                if abs(s - 1.0) > 0.01 and s > 0:
                    weights = {k: v / s for k, v in weights.items()}
            b = cfg.get("risk_scoring", {}).get("bands", {})
            if b:
                bands = {k: int(v) for k, v in b.items()}
    except Exception:
        pass
    return weights, bands


def _normalize(series: pd.Series, clip: tuple[float, float] = (0, 100)) -> pd.Series:
    """Min-max normalize to 0-100; if constant, return middle or 0."""
    if series.empty:
        return series
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0, index=series.index, dtype=float)
    return ((series - mn) / (mx - mn) * 100).clip(*clip).round(1)


def _classify(score: float, bands: dict) -> str:
    if score >= bands.get("critical", 80):
        return "CRITICAL"
    if score >= bands.get("high", 60):
        return "HIGH"
    if score >= bands.get("moderate", 30):
        return "MODERATE"
    return "LOW"


def calculate_advanced_risk(
    df_flagged: pd.DataFrame,
    rule_summary: pd.DataFrame | None = None,
    behavioural: pd.DataFrame | None = None,
    peer_per_cse: pd.DataFrame | None = None,
    anomaly_df: pd.DataFrame | None = None,
    temporal_trends: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute transparent risk per CSE.

    df_flagged: preprocessed + rule/negative flags (needs cse_id, rule_R*, rapid_closure, negative_space etc.)
    Returns DataFrame with columns: cse_id, risk_score, risk_level, components..., explanation
    """
    if df_flagged is None or df_flagged.empty:
        return pd.DataFrame()

    weights, bands = _load_cfg()

    cses = df_flagged["cse_id"].dropna().unique()
    rows = []

    # Precompute per-CSE aggregates needed
    # Rule violations: total_rule_score per CSE
    rule_map = {}
    if rule_summary is not None and not rule_summary.empty and "total_rule_score" in rule_summary.columns:
        rule_map = rule_summary.set_index("cse_id")["total_rule_score"].to_dict()
    else:
        # Fallback: sum rule_R* per CSE from df_flagged
        rule_cols = [c for c in df_flagged.columns if c.startswith("rule_R")]
        if rule_cols:
            rule_map = df_flagged.groupby("cse_id")[rule_cols].sum().sum(axis=1).to_dict()

    # For normalization, need max rule score
    max_rule = max(rule_map.values()) if rule_map else 1
    if max_rule == 0:
        max_rule = 1

    # Behavioural for execution gaps & criticality
    beh_map = {}
    if behavioural is not None and not behavioural.empty:
        beh_map = behavioural.set_index("cse_id").to_dict(orient="index")

    # Peer deviations count
    peer_counts = {}
    if peer_per_cse is not None and not peer_per_cse.empty and "needs_attention" in peer_per_cse.columns:
        peer_counts = peer_per_cse[peer_per_cse["needs_attention"]].groupby("cse_id").size().to_dict()
    max_peer = max(peer_counts.values()) if peer_counts else 1
    if max_peer == 0:
        max_peer = 1

    # Anomaly scores
    anomaly_map = {}
    if anomaly_df is not None and not anomaly_df.empty and "anomaly_score" in anomaly_df.columns:
        anomaly_map = anomaly_df.set_index("cse_id")["anomaly_score"].to_dict()
        max_anom = anomaly_df["anomaly_score"].max() if anomaly_df["anomaly_score"].max() > 0 else 1
    else:
        max_anom = 1

    # Temporal trend bonus
    trend_map = {}
    if temporal_trends is not None and not temporal_trends.empty and "risk_trend" in temporal_trends.columns:
        trend_map = temporal_trends.set_index("cse_id")["risk_trend"].to_dict()

    for cse in cses:
        sub = df_flagged[df_flagged["cse_id"] == cse]
        total = len(sub)

        # 1) Rule violations 0-100
        raw_rule = rule_map.get(cse, 0)
        rule_score = min(100, raw_rule / max_rule * 100) if max_rule else 0

        # 2) Execution gaps — use rapid_closure_rate + workflow gaps
        rapid_rate = sub["rapid_closure"].mean() if "rapid_closure" in sub.columns else 0
        beh = beh_map.get(cse, {})
        exec_gap_rate = beh.get("execution_gap_rate", rapid_rate) if beh else rapid_rate
        # Also consider workflow missing: avg_missing_steps /5
        avg_missing = sub["missing_count"].mean() if "missing_count" in sub.columns else 0
        exec_component = (exec_gap_rate * 50 + (avg_missing / 5 * 100) * 0.5) * 1.0  # blend
        exec_score = min(100, exec_component * 2)  # scale to 0-100 (heuristic)
        # Fallback simpler: use negative_space style max
        if exec_score == 0 and "execution_gap" in sub.columns:
            exec_score = sub["execution_gap"].mean() * 100 if "execution_gap" in sub.columns else 0
        exec_score = float(np.clip(exec_score, 0, 100))

        # 3) Anomaly — normalize score 0-100
        raw_anom = anomaly_map.get(cse, 0)
        anomaly_score = min(100, raw_anom / max_anom * 100) if max_anom else 0
        # If CSE flagged as anomaly, bump to at least 60
        if anomaly_df is not None and not anomaly_df.empty and "anomaly" in anomaly_df.columns:
            is_anom = anomaly_df[anomaly_df["cse_id"] == cse]["anomaly"].values[0] if cse in anomaly_df["cse_id"].values else False
            if bool(is_anom):
                anomaly_score = max(anomaly_score, 60)

        # 4) Peer deviation — count needs_attention / max
        peer_cnt = peer_counts.get(cse, 0)
        peer_score = min(100, peer_cnt / max_peer * 100) if max_peer else 0

        # 5) Criticality — critical gaps + low coverage
        crit_gaps = (sub["asset_criticality"].astype(str).str.upper() == "CRITICAL").sum()
        crit_low = ((sub["asset_criticality"].astype(str).str.upper() == "CRITICAL") & (sub["activity_coverage"] < 50)).sum() if "activity_coverage" in sub.columns else 0
        crit_rate = crit_low / crit_gaps if crit_gaps > 0 else 0
        avg_cov = sub["activity_coverage"].mean() if "activity_coverage" in sub.columns else 100
        criticality_score = (crit_rate * 50 + (1 - avg_cov / 100) * 50) if pd.notna(avg_cov) else crit_rate * 50
        criticality_score = float(np.clip(criticality_score, 0, 100))

        # Temporal bonus: DETERIORATING +10, SUDDEN CHANGE +15
        trend_bonus = 0
        trend = trend_map.get(cse, "STABLE")
        if trend == "DETERIORATING":
            trend_bonus = 10
        elif trend == "SUDDEN CHANGE":
            trend_bonus = 15

        # Weighted sum (weights sum to 1.0)
        risk = (
            rule_score * weights.get("rule_violations", 0.30)
            + exec_score * weights.get("execution_gaps", 0.25)
            + anomaly_score * weights.get("anomaly", 0.20)
            + peer_score * weights.get("peer_deviation", 0.15)
            + criticality_score * weights.get("criticality", 0.10)
            + trend_bonus * 0.05  # small extra, but keep within 0-100 via clip
        )
        risk = float(np.clip(risk, 0, 100))
        # If trend_bonus pushes beyond bands, it helps surface deteriorating

        level = _classify(risk, bands)

        # Explain breakdown
        rows.append({
            "cse_id": cse,
            "total_alerts": total,
            "rule_score": round(rule_score, 1),
            "execution_score": round(exec_score, 1),
            "anomaly_score": round(anomaly_score, 1),
            "peer_score": round(peer_score, 1),
            "criticality_score": round(criticality_score, 1),
            "trend_bonus": trend_bonus,
            "risk_score": round(risk, 1),
            "risk_level": level,
            "priority": _classify(risk, bands),  # alias
            "components": f"R:{rule_score:.0f}×{weights['rule_violations']:.0%} E:{exec_score:.0f}×{weights['execution_gaps']:.0%} A:{anomaly_score:.0f}×{weights['anomaly']:.0%} P:{peer_score:.0f}×{weights['peer_deviation']:.0%} C:{criticality_score:.0f}×{weights['criticality']:.0%}",
        })

    risk_df = pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)
    # Add rank
    risk_df["rank"] = risk_df.index + 1
    return risk_df
