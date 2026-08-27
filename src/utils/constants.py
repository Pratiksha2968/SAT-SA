"""
src/utils/constants.py — SAT-SA fixed enumerations & limits
Keeps magic strings out of analytics code.
"""

# Required / optional schema (mirrors config/thresholds.yaml for offline fallback)
REQUIRED_COLUMNS = [
    "cse_id",
    "alert_id",
    "asset_id",
    "severity",
    "alert_type",
    "created_time",
    "acknowledged_time",
    "closure_time",
    "escalated",
    "disposition",
    "investigation_evidence",
    "asset_criticality",
    "expected_activity",
    "observed_activity",
]

OPTIONAL_COLUMNS = [
    "investigation_start",
    "investigation_end",
    "investigation_notes",
]

SEVERITY_ALLOWLIST = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
SEVERITY_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

DISPOSITION_ALLOWLIST = [
    "True Positive",
    "False Positive",
    "Benign True Positive",
    "Under Investigation",
    "Escalated to Tier-2",
]

ASSET_CRITICALITY_ALLOWLIST = ["CRITICAL", "HIGH", "MEDIUM"]

TIME_COLUMNS = [
    "created_time",
    "acknowledged_time",
    "investigation_start",
    "investigation_end",
    "closure_time",
]

BOOLEAN_COLUMNS = ["escalated", "investigation_evidence"]

# String → bool mapping (case-insensitive, stripped)
BOOLEAN_TRUE_SET = {"true", "1", "yes", "y", "t"}
BOOLEAN_FALSE_SET = {"false", "0", "no", "n", "f", ""}

# Limits (also in thresholds.yaml, but hard fail-safe here)
MAX_FILE_MB = 50
MAX_ROWS = 200_000

# Severity weights for criticality context (higher = more attention if gap occurs)
CRITICALITY_WEIGHT = {"CRITICAL": 1.0, "HIGH": 0.6, "MEDIUM": 0.3, "LOW": 0.1}
ASSET_CRITICALITY_WEIGHT = {"CRITICAL": 1.0, "HIGH": 0.6, "MEDIUM": 0.3}
