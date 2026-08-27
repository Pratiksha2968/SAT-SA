"""
src/analytics/review_queue.py — Supervisory Review Queue
Prioritizes CSEs/cases for manual review based on:
  risk_score, evidence strength, persistence, trend, criticality, peer deviation

Not just sorting by score — weighted priority with explainable rank.

Returns structured queue objects for UI + audit.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def _priority_label(score: float, trend: str | None, critical_gaps: int, risk_level: str = "") -> str:
    # Priority aligns with risk_level but bumps for deteriorating + critical gaps
    if risk_level:
        # Use risk_level as base, but allow upgrade for trend
        base = {"CRITICAL": "CRITICAL", "HIGH": "HIGH", "MODERATE": "MEDIUM", "LOW": "LOW"}.get(risk_level.upper(), "LOW")
        # Upgrade MEDIUM to HIGH if deteriorating + high gaps
        if base == "MEDIUM" and trend in ("DETERIORATING", "SUDDEN CHANGE", "⚠ DETERIORATING", "⚡ SUDDEN CHANGE") and critical_gaps >= 5:
            return "HIGH"
        return base
    # Fallback score-based
    if score >= 80:
        return "CRITICAL"
    if score >= 60:
        return "HIGH"
    if score >= 30:
        return "MEDIUM"
    return "LOW"


def build_review_queue(
    risk_df: pd.DataFrame,
    findings: pd.DataFrame | None = None,
    behavioural: pd.DataFrame | None = None,
    temporal_trends: pd.DataFrame | None = None,
    peer_per_cse: pd.DataFrame | None = None,
    gaps_detail: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build prioritized supervisory review queue.

    Args:
        risk_df: output of calculate_advanced_risk
        findings: RuleFinding DataFrame
        behavioural: behavioural profiles
        temporal_trends: temporal trends
        peer_per_cse: peer deviations
        gaps_detail: negative-space gaps detail

    Returns:
        DataFrame sorted by priority + risk_score with structured fields:
        rank, priority, cse_id, risk_score, finding_summary, evidence, period, peer_deviation, trend, recommended_action
    """
    if risk_df is None or risk_df.empty:
        return pd.DataFrame()

    # Helpers maps
    trend_map = {}
    if temporal_trends is not None and not temporal_trends.empty:
        trend_map = temporal_trends.set_index("cse_id")["overall_status"].to_dict() if "overall_status" in temporal_trends.columns else temporal_trends.set_index("cse_id")["risk_trend"].to_dict()

    # Count findings per CSE
    finding_counts = {}
    finding_top_rule = {}
    if findings is not None and not findings.empty:
        finding_counts = findings.groupby("cse_id").size().to_dict()
        # Top rule per CSE
        top = findings.groupby(["cse_id", "rule_id"]).size().reset_index(name="cnt").sort_values(["cse_id", "cnt"], ascending=[True, False]).drop_duplicates("cse_id")
        finding_top_rule = top.set_index("cse_id")["rule_id"].to_dict()

    # Gaps critical count
    gap_crit_map = {}
    if gaps_detail is not None and not gaps_detail.empty and "gap_severity" in gaps_detail.columns:
        gap_crit_map = gaps_detail[gaps_detail["gap_severity"] == "CRITICAL"].groupby("cse_id").size().to_dict()

    # Peer deviation count needing attention
    peer_cnt_map = {}
    if peer_per_cse is not None and not peer_per_cse.empty:
        peer_cnt_map = peer_per_cse[peer_per_cse.get("needs_attention", False)].groupby("cse_id").size().to_dict() if "needs_attention" in peer_per_cse.columns else {}

    rows = []
    for _, r in risk_df.iterrows():
        cse = r["cse_id"]
        score = r["risk_score"]
        trend = trend_map.get(cse, "— STABLE")
        crit_gaps = int(gap_crit_map.get(cse, 0))
        peer_cnt = int(peer_cnt_map.get(cse, 0))
        fc = int(finding_counts.get(cse, 0))
        top_rule = finding_top_rule.get(cse, "—")

        priority = _priority_label(score, trend, crit_gaps, r.get("risk_level", ""))

        # Build finding summary (structured object §17)
        if fc > 0:
            finding_summary = f"{fc} findings, top {top_rule}"
        else:
            finding_summary = "No rule violations"

        peer_dev = f"{peer_cnt} peer deviations" if peer_cnt > 0 else "No significant peer deviation"
        gap_evidence = f"{crit_gaps} critical gaps" if crit_gaps > 0 else ""
        evidence = "; ".join([x for x in [finding_summary, peer_dev, gap_evidence] if x])

        # Recommended action (neutral supervisory language)
        if priority == "CRITICAL":
            action = "Immediate manual supervisory review recommended"
        elif priority == "HIGH":
            action = "Supervisory review recommended within 7 days"
        elif priority == "MEDIUM":
            action = "Review recommended when capacity allows"
        else:
            action = "No immediate action, monitor trends"

        rows.append({
            "priority": priority,
            "cse_id": cse,
            "risk_score": score,
            "risk_level": r.get("risk_level", ""),
            "finding_summary": finding_summary,
            "findings_count": fc,
            "top_rule": top_rule,
            "evidence": evidence,
            "peer_deviation": peer_dev,
            "trend": trend,
            "critical_gaps": crit_gaps,
            "recommended_action": action,
        })

    queue = pd.DataFrame(rows)
    # Sort: CRITICAL > HIGH > MEDIUM > LOW, then risk_score desc
    order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    queue["priority_order"] = queue["priority"].map(order)
    queue = queue.sort_values(["priority_order", "risk_score"], ascending=[True, False]).reset_index(drop=True)
    queue["rank"] = queue.index + 1
    queue.drop(columns=["priority_order"], inplace=True)

    return queue
