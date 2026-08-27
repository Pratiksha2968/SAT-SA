"""
src/analytics/root_cause.py — Risk-Delta + Root-Cause Analysis
Explains *why* risk is high and *why it changed*.

Shows per-component contributions and, when previous risk available,
risk delta with contributors: +12 Critical not escalated etc.

Deterministic, offline.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def analyze_root_cause(risk_df: pd.DataFrame, rule_summary: pd.DataFrame | None = None, peer_per_cse: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Decompose risk into contributors per CSE.

    Args:
        risk_df: output of calculate_advanced_risk
        rule_summary: per-CSE rule counts (for specific rule contributors)
        peer_per_cse: per-CSE peer deviations

    Returns:
        DataFrame with columns: cse_id, risk_score, top_contributor, contributors list, detail text
    """
    if risk_df is None or risk_df.empty:
        return pd.DataFrame()

    rows = []
    for _, r in risk_df.iterrows():
        cse = r["cse_id"]
        # Build contributor list sorted by weighted impact
        # Use component scores already in risk_df
        comps = [
            ("Rule violations", r.get("rule_score", 0) * 0.30),
            ("Execution gaps", r.get("execution_score", 0) * 0.25),
            ("Behavioural anomaly", r.get("anomaly_score", 0) * 0.20),
            ("Peer deviation", r.get("peer_score", 0) * 0.15),
            ("Criticality/low coverage", r.get("criticality_score", 0) * 0.10),
        ]
        # Sort by contribution descending
        comps_sorted = sorted(comps, key=lambda x: x[1], reverse=True)
        top = comps_sorted[0][0] if comps_sorted else "Unknown"

        # Build evidence strings from rule_summary / peer
        details = []
        if rule_summary is not None and not rule_summary.empty and cse in rule_summary["cse_id"].values:
            rs = rule_summary[rule_summary["cse_id"] == cse].iloc[0]
            # Find top rules by count
            rule_cols = [c for c in rule_summary.columns if c.endswith("_count") and c.startswith("R")]
            top_rules = sorted([(c, int(rs[c])) for c in rule_cols if rs[c] > 0], key=lambda x: x[1], reverse=True)[:3]
            for rc, cnt in top_rules:
                rid = rc.replace("_count", "")
                details.append(f"{rid}: {cnt} violations")

        if peer_per_cse is not None and not peer_per_cse.empty:
            pcs = peer_per_cse[(peer_per_cse["cse_id"] == cse) & (peer_per_cse["needs_attention"])]
            for _, prow in pcs.head(2).iterrows():
                details.append(f"Peer: {prow['metric']} z={prow['z_score']:.1f}")

        contributors = [f"{name} (+{contrib:.0f})" for name, contrib in comps_sorted if contrib > 2]

        rows.append({
            "cse_id": cse,
            "risk_score": r["risk_score"],
            "risk_level": r["risk_level"],
            "top_contributor": top,
            "contributors": contributors,
            "contributor_detail": "; ".join(details) if details else "No dominant rule",
            "breakdown": f"R={r.get('rule_score',0):.0f} E={r.get('execution_score',0):.0f} A={r.get('anomaly_score',0):.0f} P={r.get('peer_score',0):.0f} C={r.get('criticality_score',0):.0f}",
        })

    return pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)


def risk_delta(current: pd.DataFrame, previous: pd.DataFrame | None) -> pd.DataFrame:
    """
    Compute risk delta between two risk snapshots.
    If previous is None, returns current with delta 0.
    Returns DataFrame with delta, delta contributors, status.
    """
    if current is None or current.empty:
        return pd.DataFrame()
    if previous is None or previous.empty:
        df = current.copy()
        df["risk_delta"] = 0.0
        df["previous_score"] = np.nan
        df["delta_status"] = "BASELINE"
        return df

    # Merge on cse_id
    merged = current.merge(previous[["cse_id", "risk_score"]].rename(columns={"risk_score": "previous_score"}), on="cse_id", how="left")
    merged["risk_delta"] = merged["risk_score"] - merged["previous_score"]
    merged["risk_delta"] = merged["risk_delta"].fillna(0).round(1)

    def _status(d):
        if d > 10:
            return "🔺 DETERIORATED"
        if d < -10:
            return "✅ IMPROVED"
        if abs(d) > 5:
            return "↗ CHANGED"
        return "— STABLE"

    merged["delta_status"] = merged["risk_delta"].apply(_status)

    # Contributor delta: diff per component
    for comp in ["rule_score", "execution_score", "anomaly_score", "peer_score", "criticality_score"]:
        if comp in current.columns and comp in previous.columns:
            prev_map = previous.set_index("cse_id")[comp].to_dict() if comp in previous.columns else {}
            merged[f"{comp}_delta"] = merged.apply(lambda r: round(r[comp] - prev_map.get(r["cse_id"], 0), 1), axis=1)

    return merged.sort_values("risk_delta", ascending=False).reset_index(drop=True)
