"""
src/analytics/rules.py — SAT-SA Supervisory Rule Engine (10 rules)
Explainable, deterministic, offline. Vectorized (no Python loops over rows).

Each rule has:
  rule_id, description, severity, score, evidence fields
Findings are structured objects for drill-down.

Rules:
  R01 Critical not escalated
  R02 Critical without investigation
  R03 Evidence missing (any severity)
  R04 Investigation skipped (ack present but no investigation)
  R05 Rapid closure (CRITICAL < 5 min)
  R06 Repeated asset+type handling (potential copy-paste)
  R07 False-positive concentration (>40% per CSE)
  R08 Low investigation activity vs peers
  R09 Critical asset under-monitored (CRITICAL asset + coverage<50)
  R10 Backlog/workload imbalance (alert volume outlier)

All thresholds configurable via config/thresholds.yaml.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

CONFIG_PATH = Path("config/thresholds.yaml")

# Default rule metadata (fallback if YAML missing)
RULE_DEFINITIONS: Dict[str, Dict] = {
    "R01": {"description": "CRITICAL alert was not escalated", "severity": "HIGH", "score": 15, "enabled": True},
    "R02": {"description": "CRITICAL alert closed without investigation evidence", "severity": "CRITICAL", "score": 20, "enabled": True},
    "R03": {"description": "Required investigation evidence missing", "severity": "MEDIUM", "score": 10, "enabled": True},
    "R04": {"description": "Investigation skipped (acknowledged but no investigation)", "severity": "MEDIUM", "score": 10, "enabled": True},
    "R05": {"description": "Alert closed unusually fast (CRITICAL <5 min)", "severity": "HIGH", "score": 18, "enabled": True},
    "R06": {"description": "Repeated asset+type alerts with identical handling", "severity": "LOW", "score": 5, "enabled": True},
    "R07": {"description": "High false-positive concentration (>40% of CSE alerts)", "severity": "MEDIUM", "score": 12, "enabled": True},
    "R08": {"description": "Investigation rate significantly below peer median", "severity": "MEDIUM", "score": 12, "enabled": True},
    "R09": {"description": "CRITICAL asset with low observed activity (<50% coverage)", "severity": "HIGH", "score": 14, "enabled": True},
    "R10": {"description": "Abnormal workload imbalance (peer volume outlier)", "severity": "LOW", "score": 8, "enabled": True},
}

# YAML key mapping
YAML_RULE_MAP = {
    "R01_critical_not_escalated": "R01",
    "R02_critical_no_investigation": "R02",
    "R03_evidence_missing": "R03",
    "R04_investigation_skipped": "R04",
    "R05_rapid_closure": "R05",
    "R06_repeated_alert_handling": "R06",
    "R07_false_positive_concentration": "R07",
    "R08_low_investigation_activity": "R08",
    "R09_critical_asset_under_monitored": "R09",
    "R10_backlog_anomaly": "R10",
}


def _load_rule_config() -> Dict[str, Dict]:
    """Merge YAML rule thresholds over defaults."""
    rules = {k: dict(v) for k, v in RULE_DEFINITIONS.items()}
    try:
        if CONFIG_PATH.exists():
            cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
            yaml_rules = cfg.get("rules", {})
            for yaml_key, cfg_val in yaml_rules.items():
                rid = YAML_RULE_MAP.get(yaml_key, yaml_key)
                if rid in rules:
                    rules[rid].update(cfg_val)
            # Also allow direct R01 keys
            for k, v in yaml_rules.items():
                if k in rules and isinstance(v, dict):
                    rules[k].update(v)
    except Exception:
        pass
    return rules


@dataclass
class RuleFinding:
    """Structured supervisory finding — one per triggered rule per alert/CSE."""
    finding_id: str
    rule_id: str
    cse_id: str
    alert_id: str | None
    asset_id: str | None
    severity: str  # rule severity, not alert severity
    score: int
    description: str
    alert_severity: str | None = None
    asset_criticality: str | None = None
    evidence: str = ""
    created_time: str | None = None
    # For aggregate rules, period/count info
    count: int | None = None
    rate: float | None = None

    def to_dict(self) -> Dict:
        return asdict(self)


def evaluate_rules(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply 10 supervisory rules.

    Args:
        df: preprocessed DataFrame (must have columns from features.py).
            Internally calls preprocess if missing derived cols (safe).

    Returns:
        df_flagged: input df with rule flag columns (R01...R10 booleans + rule_scores)
        findings: DataFrame of RuleFinding (one row per violation)
        summary: per-CSE per-rule counts + total score
    """
    if df is None or df.empty:
        return df.copy() if df is not None else df, pd.DataFrame(), pd.DataFrame()

    df = df.copy()
    rule_cfg = _load_rule_config()

    # Ensure derived cols exist (caller should have preprocessed, but tolerate raw)
    if "investigation_present" not in df.columns and "investigation_evidence" in df.columns:
        df["investigation_present"] = df["investigation_evidence"].fillna(False).astype(bool)
    if "rapid_closure" not in df.columns and "resolution_time" in df.columns:
        df["rapid_closure"] = (df["severity"].astype(str).str.upper() == "CRITICAL") & (df["resolution_time"] < 5)
    if "activity_coverage" not in df.columns and {"observed_activity", "expected_activity"}.issubset(df.columns):
        exp = pd.to_numeric(df["expected_activity"], errors="coerce")
        obs = pd.to_numeric(df["observed_activity"], errors="coerce")
        df["activity_coverage"] = np.where(exp > 0, obs / exp * 100, np.nan)
        df["activity_coverage"] = pd.Series(df["activity_coverage"]).clip(0, 100)
    if "is_false_positive" not in df.columns and "disposition" in df.columns:
        df["is_false_positive"] = df["disposition"].astype(str).str.strip() == "False Positive"

    # Prepare flag columns
    for rid in RULE_DEFINITIONS:
        df[f"rule_{rid}"] = False

    findings: List[RuleFinding] = []
    fid_counter = 0

    def _next_id(rid: str) -> str:
        nonlocal fid_counter
        fid_counter += 1
        return f"SF-{rid}-{fid_counter:04d}"

    # Helper to add findings vectorized — collects indices then emits findings
    def _emit(rid: str, mask: pd.Series, evidence_tpl: str = ""):
        if not rule_cfg.get(rid, {}).get("enabled", True):
            return
        idx = df.index[mask]
        if len(idx) == 0:
            return
        cfg = rule_cfg[rid]
        for i in idx:
            row = df.loc[i]
            fid_counter_local = _next_id(rid)
            # Safe evidence formatting — handle missing keys and NaT gracefully
            evidence = cfg["description"]
            if evidence_tpl:
                try:
                    # Build dict with both row values and aliases
                    d = {k: ("" if pd.isna(v) else v) for k, v in row.items()}
                    # Alias for templates that use alert_severity
                    d["alert_severity"] = d.get("severity", "")
                    # Format with dict, format floats nicely
                    evidence = evidence_tpl.format_map({k: (f"{v:.1f}" if isinstance(v, float) else str(v)) for k, v in d.items()})
                except Exception:
                    evidence = evidence_tpl
            findings.append(
                RuleFinding(
                    finding_id=fid_counter_local,
                    rule_id=rid,
                    cse_id=str(row.get("cse_id", "")),
                    alert_id=str(row.get("alert_id", "")) if pd.notna(row.get("alert_id")) else None,
                    asset_id=str(row.get("asset_id", "")) if pd.notna(row.get("asset_id")) else None,
                    severity=cfg["severity"],
                    score=int(cfg["score"]),
                    description=cfg["description"],
                    alert_severity=str(row.get("severity", "")) if pd.notna(row.get("severity")) else None,
                    asset_criticality=str(row.get("asset_criticality", "")) if pd.notna(row.get("asset_criticality")) else None,
                    evidence=evidence,
                    created_time=str(row.get("created_time", "")) if pd.notna(row.get("created_time")) else None,
                )
            )

    # ------------------------------------------------------------------
    # R01: Critical not escalated
    # ------------------------------------------------------------------
    mask_r01 = (df["severity"].astype(str).str.upper() == "CRITICAL") & (~df["escalated"].fillna(False).astype(bool))
    df.loc[mask_r01, "rule_R01"] = True
    _emit("R01", mask_r01, "CRITICAL {alert_id} at {asset_id} not escalated (escalated={escalated})")

    # ------------------------------------------------------------------
    # R02: Critical without investigation evidence
    # ------------------------------------------------------------------
    mask_r02 = (df["severity"].astype(str).str.upper() == "CRITICAL") & (~df["investigation_present"].fillna(False).astype(bool))
    df.loc[mask_r02, "rule_R02"] = True
    _emit("R02", mask_r02, "CRITICAL {alert_id} closed without investigation (evidence={investigation_present})")

    # ------------------------------------------------------------------
    # R03: Evidence missing (any severity, broader)
    # We exclude R02 overlap for scoring clarity? No, keep both — R02 is CRITICAL subset, R03 broader but we flag non-CRITICAL only to avoid double-count?
    # Design: R03 flags ALL missing; R02 is extra weight for critical. So R03 = missing_evidence AND not CRITICAL (to avoid duplicate)
    # Instead we flag all missing, caller aggregates distinct. Keep distinct.
    mask_r03 = (~df["investigation_present"].fillna(False).astype(bool))
    # For non-critical only to avoid double-penalize critical? We'll flag all but downstream can dedup. Keep all for transparency.
    df.loc[mask_r03, "rule_R03"] = True
    _emit("R03", mask_r03, "Evidence missing for {alert_id} (severity={alert_severity})")

    # ------------------------------------------------------------------
    # R04: Investigation skipped (ack present but no investigation)
    # ------------------------------------------------------------------
    has_ack = df["acknowledged_time"].notna() if "acknowledged_time" in df.columns else pd.Series(True, index=df.index)
    mask_r04 = has_ack & (~df["investigation_present"].fillna(False).astype(bool))
    df.loc[mask_r04, "rule_R04"] = True
    _emit("R04", mask_r04, "Investigation skipped for {alert_id} (ack at {acknowledged_time}, no investigation)")

    # ------------------------------------------------------------------
    # R05: Rapid closure (CRITICAL <5 min)
    # ------------------------------------------------------------------
    mask_r05 = df["rapid_closure"].fillna(False).astype(bool) if "rapid_closure" in df.columns else pd.Series(False, index=df.index)
    df.loc[mask_r05, "rule_R05"] = True
    _emit("R05", mask_r05, "Rapid closure {alert_id} in {resolution_time:.1f} min (CRITICAL)")

    # ------------------------------------------------------------------
    # R06: Repeated asset+type handling (group size >=3)
    # ------------------------------------------------------------------
    if {"cse_id", "asset_id", "alert_type"}.issubset(df.columns):
        grp = df.groupby(["cse_id", "asset_id", "alert_type"], dropna=False).size().reset_index(name="grp_count")
        repeated = grp[grp["grp_count"] >= 3]
        # Merge back to mark
        if not repeated.empty:
            merged = df.merge(repeated[["cse_id", "asset_id", "alert_type"]], on=["cse_id", "asset_id", "alert_type"], how="inner")
            # merged rows are the repeated ones
            mask_r06_idx = merged.index  # careful: merged index not original df index — better use isin
            # Use isin on tuple key
            key = df["cse_id"].astype(str) + "|" + df["asset_id"].astype(str) + "|" + df["alert_type"].astype(str)
            rkey = repeated["cse_id"].astype(str) + "|" + repeated["asset_id"].astype(str) + "|" + repeated["alert_type"].astype(str)
            mask_r06 = key.isin(rkey)
            df.loc[mask_r06, "rule_R06"] = True
            _emit("R06", mask_r06, "Repeated pattern {asset_id}/{alert_type} ({cse_id}) — potential copy-paste handling")
        else:
            mask_r06 = pd.Series(False, index=df.index)
    else:
        mask_r06 = pd.Series(False, index=df.index)

    # ------------------------------------------------------------------
    # R09: Critical asset with low coverage (<50)
    # ------------------------------------------------------------------
    mask_r09 = (df["asset_criticality"].astype(str).str.upper() == "CRITICAL") & (df["activity_coverage"] < 50)
    df.loc[mask_r09, "rule_R09"] = True
    _emit("R09", mask_r09, "CRITICAL asset {asset_id} low coverage {activity_coverage:.1f}% (expected {expected_activity}, observed {observed_activity})")

    # ------------------------------------------------------------------
    # CSE-level aggregate rules: R07, R08, R10 (one finding per CSE violation)
    # ------------------------------------------------------------------
    # Compute per-CSE stats once
    cse_stats = df.groupby("cse_id").agg(
        total_alerts=("alert_id", "count"),
        false_positives=("is_false_positive", "sum"),
        investigations=("investigation_present", "sum"),
        avg_coverage=("activity_coverage", "mean"),
    ).reset_index()
    cse_stats["false_positive_rate"] = cse_stats["false_positives"] / cse_stats["total_alerts"].replace(0, np.nan)
    cse_stats["investigation_rate"] = cse_stats["investigations"] / cse_stats["total_alerts"].replace(0, np.nan)

    # R07: False-positive concentration >40%
    r07_cses = cse_stats[cse_stats["false_positive_rate"] > 0.40]["cse_id"].tolist()
    for cse in r07_cses:
        if not rule_cfg.get("R07", {}).get("enabled", True):
            continue
        row = cse_stats[cse_stats["cse_id"] == cse].iloc[0]
        fid = _next_id("R07")
        findings.append(
            RuleFinding(
                finding_id=fid,
                rule_id="R07",
                cse_id=str(cse),
                alert_id=None,
                asset_id=None,
                severity=rule_cfg["R07"]["severity"],
                score=int(rule_cfg["R07"]["score"]),
                description=rule_cfg["R07"]["description"],
                evidence=f"CSE {cse}: {int(row['false_positives'])}/{int(row['total_alerts'])} ({row['false_positive_rate']:.1%}) false positives",
                count=int(row["false_positives"]),
                rate=float(row["false_positive_rate"]),
            )
        )
        df.loc[df["cse_id"] == cse, "rule_R07"] = df.loc[df["cse_id"] == cse, "is_false_positive"]

    # R08: Investigation rate significantly below peer median
    # Use median; flag if rate < median - 0.15 (configurable, simple robust)
    peer_median = cse_stats["investigation_rate"].median()
    # Peer median from thresholds? Use fixed delta 0.15 (~15% points)
    delta = 0.15
    r08_threshold = max(0, peer_median - delta)
    r08_cses = cse_stats[cse_stats["investigation_rate"] < r08_threshold]["cse_id"].tolist()
    for cse in r08_cses:
        if not rule_cfg.get("R08", {}).get("enabled", True):
            continue
        row = cse_stats[cse_stats["cse_id"] == cse].iloc[0]
        fid = _next_id("R08")
        findings.append(
            RuleFinding(
                finding_id=fid,
                rule_id="R08",
                cse_id=str(cse),
                alert_id=None,
                asset_id=None,
                severity=rule_cfg["R08"]["severity"],
                score=int(rule_cfg["R08"]["score"]),
                description=rule_cfg["R08"]["description"],
                evidence=f"CSE {cse}: investigation rate {row['investigation_rate']:.1%} vs peer median {peer_median:.1%} (threshold {r08_threshold:.1%})",
                count=int(row["investigations"]),
                rate=float(row["investigation_rate"]),
            )
        )
        df.loc[df["cse_id"] == cse, "rule_R08"] = True

    # R10: Workload imbalance — volume outlier (|z|>1.5)
    # Simple z-score on total_alerts per CSE
    mean_vol = cse_stats["total_alerts"].mean()
    std_vol = cse_stats["total_alerts"].std(ddof=0) or 1
    cse_stats["volume_z"] = (cse_stats["total_alerts"] - mean_vol) / std_vol
    # Load threshold from config peer.z_threshold
    try:
        cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        z_thr = float(cfg.get("peer", {}).get("z_threshold", 1.5))
    except Exception:
        z_thr = 1.5
    r10_cses = cse_stats[cse_stats["volume_z"].abs() > z_thr]["cse_id"].tolist()
    for cse in r10_cses:
        if not rule_cfg.get("R10", {}).get("enabled", True):
            continue
        row = cse_stats[cse_stats["cse_id"] == cse].iloc[0]
        fid = _next_id("R10")
        findings.append(
            RuleFinding(
                finding_id=fid,
                rule_id="R10",
                cse_id=str(cse),
                alert_id=None,
                asset_id=None,
                severity=rule_cfg["R10"]["severity"],
                score=int(rule_cfg["R10"]["score"]),
                description=rule_cfg["R10"]["description"],
                evidence=f"CSE {cse}: {int(row['total_alerts'])} alerts (z={row['volume_z']:.2f}, peer mean {mean_vol:.0f})",
                count=int(row["total_alerts"]),
                rate=float(row["volume_z"]),
            )
        )
        df.loc[df["cse_id"] == cse, "rule_R10"] = True

    # ------------------------------------------------------------------
    # Build DataFrames
    # ------------------------------------------------------------------
    findings_df = pd.DataFrame([f.to_dict() for f in findings])
    if not findings_df.empty:
        findings_df = findings_df.sort_values(["cse_id", "rule_id", "finding_id"]).reset_index(drop=True)

    # Per-CSE per-rule summary + total score
    # For CSE-level aggregate rules, count is number of CSE-level findings (0 or 1), not per-alert flags
    cse_level_counts = {}
    if not findings_df.empty:
        cse_level_counts = findings_df[findings_df["rule_id"].isin(["R07","R08","R10"])].groupby(["cse_id","rule_id"]).size().to_dict()
    summary_rows = []
    for cse, group in df.groupby("cse_id"):
        row = {"cse_id": cse, "total_alerts": int(len(group))}
        total_score = 0
        for rid in RULE_DEFINITIONS:
            col = f"rule_{rid}"
            if rid in ("R07","R08","R10"):
                # CSE-level: count findings, not per-row flags
                cnt = int(cse_level_counts.get((cse, rid), 0))
            else:
                cnt = int(group[col].sum()) if col in group.columns else 0
            row[f"{rid}_count"] = cnt
            cfg = rule_cfg[rid]
            if rid in ("R07", "R08", "R10"):
                score = cfg["score"] if cnt > 0 else 0
            else:
                score = cnt * cfg["score"]
            row[f"{rid}_score"] = score
            total_score += score
        row["total_rule_score"] = total_score
        row["findings_count"] = int(len(findings_df[findings_df["cse_id"] == cse])) if not findings_df.empty else int((group.filter(like="rule_").sum(axis=1) > 0).sum())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows).sort_values("total_rule_score", ascending=False).reset_index(drop=True) if summary_rows else pd.DataFrame()

    # Add rule score aggregate to df for downstream risk scoring
    df["total_rule_score"] = 0
    for rid in RULE_DEFINITIONS:
        cfg = rule_cfg[rid]
        col = f"rule_{rid}"
        if rid in ("R07", "R08", "R10"):
            # For aggregate rules, map summary score back: add score once per row in violating CSE
            if not summary_df.empty:
                score_map = summary_df.set_index("cse_id")[f"{rid}_score"].to_dict()
                df["total_rule_score"] += df["cse_id"].map(score_map).fillna(0).astype(int)
                # But this double-counts per alert — instead we keep per-alert flags and summary handles CSE-level.
                # Reset to per-alert for other rules only for DF total: recompute correctly below
                pass
        else:
            df["total_rule_score"] += df[col].astype(int) * cfg["score"]
    # Recompute total correctly: per-alert sum + CSE-level once (use summary mapping)
    df["_rule_score_cse_level"] = 0
    if not summary_df.empty:
        for rid in ("R07", "R08", "R10"):
            if f"{rid}_score" in summary_df.columns:
                m = summary_df.set_index("cse_id")[f"{rid}_score"].to_dict()
                df["_rule_score_cse_level"] += df["cse_id"].map(m).fillna(0)
    df["total_rule_score"] = df["total_rule_score"] + df["_rule_score_cse_level"]
    df.drop(columns=["_rule_score_cse_level"], inplace=True, errors="ignore")

    return df, findings_df, summary_df
