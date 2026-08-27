"""
src/analytics/what_if.py — What-If Supervisory Simulator
Simulates risk impact of corrective actions. Clearly labeled SIMULATION,
not guaranteed outcome.

Actions:
  - fix_investigation: set missing_investigation -> False for CSE
  - fix_escalation: set escalated -> True for CRITICAL not escalated
  - fix_evidence: set investigation_present True
  - fix_coverage: bump observed_activity to expected_activity
  - fix_all: combine above

Recomputes risk via same calculate_advanced_risk logic on patched DataFrame.
Deterministic, offline.
"""
from __future__ import annotations
import pandas as pd
import numpy as np


def simulate_what_if(
    df_flagged: pd.DataFrame,
    cse_id: str,
    fixes: list[str],
    rule_summary: pd.DataFrame | None = None,
    behavioural: pd.DataFrame | None = None,
    peer_per_cse: pd.DataFrame | None = None,
    anomaly_df: pd.DataFrame | None = None,
    temporal_trends: pd.DataFrame | None = None,
) -> dict:
    """
    Simulate risk after fixes for a single CSE.

    Fixes: subset of ["fix_investigation","fix_escalation","fix_evidence","fix_coverage","fix_all"]
    Returns dict with original risk, simulated risk, delta, actions, explanation.
    """
    from src.analytics.rules import evaluate_rules
    from src.analytics.negative_space import analyze_negative_space
    from src.analytics.behavioural import build_behavioural_profiles
    from src.analytics.peer_analysis import analyze_peers
    from src.analytics.risk_engine import calculate_advanced_risk

    # Import here to avoid circular
    if "fix_all" in fixes:
        fixes = ["fix_investigation", "fix_escalation", "fix_evidence", "fix_coverage"]

    # Baseline risk
    def _compute(df_current: pd.DataFrame) -> float:
        # Re-run pipeline subset for this df_current
        # For speed, we reuse preprocessed df but recompute flags
        try:
            # Re-evaluate rules/negative/behavioural/peer on patched df
            df_r, _, rs = evaluate_rules(df_current)
            df_n, _, _ = analyze_negative_space(df_r)
            beh = build_behavioural_profiles(df_n)
            _, peer = analyze_peers(df_n, beh)
            risk = calculate_advanced_risk(df_n, rs, beh, peer, anomaly_df, temporal_trends)
            row = risk[risk["cse_id"] == cse_id]
            return float(row["risk_score"].values[0]) if not row.empty else 0.0
        except Exception as e:
            return 0.0

    baseline = _compute(df_flagged)
    # Patch
    patched = df_flagged.copy()
    mask_cse = patched["cse_id"] == cse_id

    if "fix_investigation" in fixes:
        # Set investigation_present True for those missing in this CSE
        if "investigation_present" in patched.columns:
            patched.loc[mask_cse & (~patched["investigation_present"].fillna(False)), "investigation_present"] = True
        if "investigation_evidence" in patched.columns:
            patched.loc[mask_cse, "investigation_evidence"] = True
        if "investigation_start" in patched.columns:
            # Fill missing investigation_start with acknowledged_time + 5 min
            missing = mask_cse & patched["investigation_start"].isna() & patched["acknowledged_time"].notna()
            patched.loc[missing, "investigation_start"] = patched.loc[missing, "acknowledged_time"] + pd.to_timedelta(5, unit="m")

    if "fix_escalation" in fixes:
        if "escalated" in patched.columns:
            # Only for CRITICAL not escalated
            crit_mask = mask_cse & (patched["severity"].astype(str).str.upper() == "CRITICAL") & (~patched["escalated"].fillna(False))
            patched.loc[crit_mask, "escalated"] = True

    if "fix_evidence" in fixes:
        if "investigation_present" in patched.columns:
            patched.loc[mask_cse, "investigation_present"] = True
        if "investigation_evidence" in patched.columns:
            patched.loc[mask_cse, "investigation_evidence"] = True

    if "fix_coverage" in fixes:
        if {"expected_activity", "observed_activity"}.issubset(patched.columns):
            patched.loc[mask_cse, "observed_activity"] = patched.loc[mask_cse, "expected_activity"]

    simulated = _compute(patched)
    delta = round(simulated - baseline, 1)

    # Build explanation
    actions_text = ", ".join(fixes) if fixes else "none"
    impact = "reduction" if delta < 0 else ("increase" if delta > 0 else "no change")
    explanation = (
        f"Simulation for {cse_id}: baseline {baseline:.1f} → simulated {simulated:.1f} (Δ {delta:+.1f}, {impact}). "
        f"Actions: {actions_text}. "
        "This is a simulation — not a guaranteed outcome. It shows which corrective action would have the greatest impact."
    )

    return {
        "cse_id": cse_id,
        "baseline_risk": round(baseline, 1),
        "simulated_risk": round(simulated, 1),
        "delta": delta,
        "fixes": fixes,
        "explanation": explanation,
        "is_simulation": True,
    }


def what_if_grid(
    df_flagged: pd.DataFrame,
    cse_id: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Run grid of single-fix and fix-all simulations for a CSE.
    Returns DataFrame with one row per scenario for UI display.
    """
    scenarios = [
        (["fix_investigation"], "Resolve missing investigation"),
        (["fix_escalation"], "Resolve escalation gaps"),
        (["fix_evidence"], "Resolve evidence gaps"),
        (["fix_coverage"], "Restore activity coverage"),
        (["fix_investigation", "fix_evidence"], "Fix investigation + evidence"),
        (["fix_all"], "Resolve all major gaps"),
    ]
    rows = []
    for fixes, label in scenarios:
        res = simulate_what_if(df_flagged, cse_id, fixes, **kwargs)
        rows.append({
            "scenario": label,
            "fixes": ", ".join(fixes),
            "baseline": res["baseline_risk"],
            "simulated": res["simulated_risk"],
            "delta": res["delta"],
            "explanation": res["explanation"],
        })
    return pd.DataFrame(rows)
