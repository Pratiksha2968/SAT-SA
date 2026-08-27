"""
generate_data.py — SAT-SA Synthetic SOC Data Generator

Creates a realistic offline SOC case-management dataset for 5 CSEs.
Designed to demonstrate supervisory analytics, NOT attack detection.

Running:
    python generate_data.py
Produces:
    data/raw/soc_alerts.csv  (~4000 rows)
    data/processed/ (empty, used by later pipeline steps)
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

NUM_RECORDS = 4000  # within 3000–5000 required range

CSES = ["CSE-A", "CSE-B", "CSE-C", "CSE-D", "CSE-E"]

SEVERITIES = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

ALERT_TYPES = [
    "Malware",
    "Brute Force",
    "Suspicious Login",
    "Data Exfiltration",
    "Phishing",
    "Privilege Escalation",
    "Unauthorized Access",
    "Network Anomaly",
]

DISPOSITIONS = [
    "True Positive",
    "False Positive",
    "Benign True Positive",
    "Under Investigation",
    "Escalated to Tier-2",
]

ASSET_CRITICALITIES = ["CRITICAL", "HIGH", "MEDIUM"]

# Reproducible generation — remove or change seed for different demo data
np.random.seed(42)
random.seed(42)

# ------------------------------------------------------------
# Per-CSE behavioural profiles
# Tuned so CSE-C is clearly highest-risk for the demo story
# ------------------------------------------------------------
CSE_PROFILES = {
    # Healthy SOC: prompt acknowledgement, thorough investigation, healthy escalation, good activity coverage
    "CSE-A": {
        "severity_weights": [10, 25, 35, 30],  # low % critical
        "investigation_rate": 0.92,
        "escalation_rate": 0.22,
        "rapid_critical_close_prob": 0.02,  # almost never
        "ack_range": (2, 30),
        "closure_range": (45, 720),
        "expected_range": (80, 200),
        "observed_coverage": (0.75, 1.10),  # 75%–110% -> clipped to 100%
        "disposition_weights": [35, 25, 15, 15, 10],
    },
    # Moderate weakness: slightly slower, fewer investigations
    "CSE-B": {
        "severity_weights": [14, 28, 32, 26],
        "investigation_rate": 0.78,
        "escalation_rate": 0.14,
        "rapid_critical_close_prob": 0.10,
        "ack_range": (5, 90),
        "closure_range": (30, 600),
        "expected_range": (70, 200),
        "observed_coverage": (0.55, 0.95),
        "disposition_weights": [30, 30, 15, 15, 10],
    },
    # Intentionally weak — highest risk in demo
    "CSE-C": {
        "severity_weights": [28, 32, 24, 16],  # many criticals
        "investigation_rate": 0.45,             # less than half have evidence
        "escalation_rate": 0.03,                # almost never escalated
        "rapid_critical_close_prob": 0.55,      # over half critical closed <5 min
        "ack_range": (1, 60),
        "closure_range": (20, 480),
        "expected_range": (90, 200),
        "observed_coverage": (0.10, 0.55),      # large visibility gap
        "disposition_weights": [20, 15, 10, 35, 20],
    },
    # Healthy like CSE-A
    "CSE-D": {
        "severity_weights": [9, 24, 36, 31],
        "investigation_rate": 0.94,
        "escalation_rate": 0.24,
        "rapid_critical_close_prob": 0.01,
        "ack_range": (2, 25),
        "closure_range": (60, 900),
        "expected_range": (80, 200),
        "observed_coverage": (0.78, 1.10),
        "disposition_weights": [35, 25, 15, 15, 10],
    },
    # Some anomalies / moderate weakness
    "CSE-E": {
        "severity_weights": [16, 26, 30, 28],
        "investigation_rate": 0.70,
        "escalation_rate": 0.10,
        "rapid_critical_close_prob": 0.18,
        "ack_range": (5, 120),
        "closure_range": (25, 600),
        "expected_range": (70, 200),
        "observed_coverage": (0.45, 0.85),
        "disposition_weights": [25, 30, 15, 20, 10],
    },
}

INVESTIGATION_NOTES_GOOD = [
    "Analyst reviewed logs and confirmed activity with evidence archived.",
    "Reviewed source IP, asset logs and correlated with previous alerts.",
    "Investigation completed; evidence recorded and escalated per playbook.",
    "Alert correlated with threat intel; full timeline documented.",
    "Analyst investigation performed; packet capture and EDR evidence attached.",
]

INVESTIGATION_NOTES_WEAK = [
    "",  # empty = suspicious
    "Closed quickly — no notes.",
    "N/A",
]

# ============================================================
# GENERATION
# ============================================================

start_date = datetime(2026, 1, 1)
end_date = datetime(2026, 6, 30)

records = []

for i in range(NUM_RECORDS):
    cse = random.choice(CSES)
    profile = CSE_PROFILES[cse]

    # Severity — per-CSE weights
    severity = random.choices(SEVERITIES, weights=profile["severity_weights"])[0]

    alert_type = random.choice(ALERT_TYPES)

    # Realistic timestamp spread across Jan–Jun 2026, business-hours skewed
    # Random day + random minute-of-day
    random_days = random.randint(0, (end_date - start_date).days)
    # Skew toward 09:00–18:00 with some off-hours
    if random.random() < 0.75:
        random_minute = random.randint(9 * 60, 18 * 60)
    else:
        random_minute = random.randint(0, 24 * 60 - 1)
    created_time = start_date + timedelta(days=random_days, minutes=random_minute)
    # Add small random seconds for realism
    created_time += timedelta(seconds=random.randint(0, 59))

    # Acknowledgement time
    ack_min = random.randint(*profile["ack_range"])
    acknowledged_time = created_time + timedelta(minutes=ack_min, seconds=random.randint(0, 59))

    # Investigation evidence decision
    has_evidence = random.random() < profile["investigation_rate"]

    if has_evidence:
        investigation_start = acknowledged_time + timedelta(minutes=random.randint(1, 25))
        # Healthy CSEs have longer investigations; CSE-C shorter when present
        if cse == "CSE-C":
            inv_duration = random.randint(2, 30)
        elif cse in ("CSE-B", "CSE-E"):
            inv_duration = random.randint(5, 180)
        else:
            inv_duration = random.randint(15, 300)
        investigation_end = investigation_start + timedelta(minutes=inv_duration, seconds=random.randint(0, 59))
        notes = random.choice(INVESTIGATION_NOTES_GOOD)
    else:
        investigation_start = pd.NaT
        investigation_end = pd.NaT
        notes = random.choice(INVESTIGATION_NOTES_WEAK)

    # Closure time — rapid critical logic
    # SAT-SA execution-gap rule is: CRITICAL closed in <5 minutes (from creation)
    # So rapid closures are measured from created_time for strong signal
    is_critical = severity == "CRITICAL"
    should_rapid = is_critical and (random.random() < profile["rapid_critical_close_prob"])

    if should_rapid:
        # Critical closed in 1–4 minutes after CREATION (very suspicious)
        closure_delta = random.randint(1, 4)
        closure_time = created_time + timedelta(minutes=closure_delta, seconds=random.randint(0, 59))
        # Often no evidence when rapidly closed (strengthens signal)
        if random.random() < 0.65:
            has_evidence = False
            investigation_start = pd.NaT
            investigation_end = pd.NaT
            notes = random.choice(["", "Closed quickly — no notes.", "N/A"])
        # Rapid cases are almost never escalated
        escalated = random.random() < 0.02
    else:
        # Normal closure — after acknowledgement
        closure_delta = random.randint(*profile["closure_range"])
        closure_time = acknowledged_time + timedelta(minutes=closure_delta, seconds=random.randint(0, 59))

    # Escalation
    # Critical alerts should ideally be escalated more; CSE-C breaks this pattern
    base_esc = profile["escalation_rate"]
    if is_critical and cse != "CSE-C":
        base_esc = min(0.45, base_esc + 0.15)
    escalated = random.random() < base_esc
    # Rapid-closed critical alerts in CSE-C almost never escalated
    if should_rapid and cse == "CSE-C":
        escalated = random.random() < 0.02

    # Disposition — correlated with evidence/escalation
    disposition = random.choices(DISPOSITIONS, weights=profile["disposition_weights"])[0]
    if not has_evidence and not escalated and is_critical and should_rapid:
        # Suspicious pattern often marked as quickly closed / under investigation
        disposition = random.choice(["Under Investigation", "False Positive"])

    # Asset fields
    asset_id = f"ASSET-{random.randint(1, 100):03d}"
    asset_criticality = random.choices(
        ASSET_CRITICALITIES, weights=[30, 35, 35]
    )[0]
    # Make critical assets slightly more likely to get CRITICAL alerts
    if is_critical and random.random() < 0.55:
        asset_criticality = "CRITICAL"

    # Expected vs observed activity (negative-space signal)
    expected_activity = random.randint(*profile["expected_range"])
    coverage = random.uniform(*profile["observed_coverage"])
    observed_activity = int(expected_activity * coverage)
    # Add small noise
    observed_activity = max(0, observed_activity + random.randint(-5, 5))
    # For healthy CSEs, clip observed to expected (good coverage); CSE-C stays low
    # Keep raw — preprocessing will compute coverage % and clip to 100

    records.append({
        "cse_id": cse,
        "alert_id": f"ALT-{i+1:05d}",
        "asset_id": asset_id,
        "severity": severity,
        "alert_type": alert_type,
        "created_time": created_time,
        "acknowledged_time": acknowledged_time,
        "investigation_start": investigation_start,
        "investigation_end": investigation_end,
        "closure_time": closure_time,
        "escalated": escalated,
        "disposition": disposition,
        "investigation_evidence": has_evidence,
        "investigation_notes": notes,
        "asset_criticality": asset_criticality,
        "expected_activity": expected_activity,
        "observed_activity": observed_activity,
    })

# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(records)

# Ensure output dirs exist
for d in ["data/raw", "data/processed"]:
    Path(d).mkdir(parents=True, exist_ok=True)

output_file = Path("data/raw/soc_alerts.csv")
df.to_csv(output_file, index=False)

print("=" * 55)
print("SAT-SA synthetic dataset generated successfully!")
print("=" * 55)
print(f"Records generated : {len(df)}")
print(f"CSEs              : {df['cse_id'].nunique()}  ({', '.join(sorted(df['cse_id'].unique()))})")
print(f"Date range        : {df['created_time'].min()} to {df['created_time'].max()}")
print(f"Output file       : {output_file}")
print()
print("Records per CSE:")
print(df["cse_id"].value_counts().sort_index().to_string())
print()
print("Severity distribution:")
print(df["severity"].value_counts().to_string())
print()
print("Disposition distribution:")
print(df["disposition"].value_counts().to_string())
print()
print("Columns:")
print(", ".join(df.columns.tolist()))
print()
# Quick sanity: show per-CSE signals that matter for demo
print("Per-CSE operational preview (demo story):")
preview = df.groupby("cse_id").agg(
    critical_rate=("severity", lambda x: (x == "CRITICAL").mean() * 100),
    investigation_rate=("investigation_evidence", "mean"),
    escalation_rate=("escalated", "mean"),
    avg_observed_ratio=("observed_activity", lambda x: (x / df.loc[x.index, "expected_activity"]).mean() * 100),
).round(2)
print(preview.to_string())
