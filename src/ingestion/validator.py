"""
src/ingestion/validator.py — SAT-SA data validation
Checks required columns, duplicates, timestamps, sequences, enums.
Returns structured ValidationResult with human-readable messages.
No exceptions for bad rows — caller decides to warn or block.
"""
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass, field
from typing import List

from src.utils.constants import (
    REQUIRED_COLUMNS,
    SEVERITY_ALLOWLIST,
    DISPOSITION_ALLOWLIST,
    ASSET_CRITICALITY_ALLOWLIST,
    TIME_COLUMNS,
)


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning"
    code: str
    message: str
    count: int = 1
    sample: str = ""


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "errors": [vars(e) for e in self.errors],
            "warnings": [vars(w) for w in self.warnings],
            "stats": self.stats,
        }

    def summary_text(self) -> str:
        lines = []
        if self.errors:
            lines.append(f"{len(self.errors)} error(s):")
            for e in self.errors:
                lines.append(f"  [ERROR {e.code}] {e.message} (n={e.count}) {e.sample}")
        if self.warnings:
            lines.append(f"{len(self.warnings)} warning(s):")
            for w in self.warnings:
                lines.append(f"  [WARN {w.code}] {w.message} (n={w.count}) {w.sample}")
        if not lines:
            lines.append("No validation issues — dataset looks healthy.")
        return "\n".join(lines)


def _sample_str(series: pd.Series, n: int = 3) -> str:
    try:
        vals = series.dropna().astype(str).head(n).tolist()
        return ", ".join(vals) if vals else ""
    except Exception:
        return ""


def validate_dataset(df: pd.DataFrame | None) -> ValidationResult:
    """
    Validate a SAT-SA dataframe.
    - Checks required columns, duplicate alert_id, missing cse_id
    - Validates severity/disposition/criticality enums
    - Parses timestamps, checks impossible sequences
    - Validates boolean-like fields, numeric ranges
    Returns ValidationResult (never raises).
    """
    errors: List[ValidationIssue] = []
    warnings: List[ValidationIssue] = []
    stats: dict = {}

    if df is None:
        return ValidationResult(is_valid=False, errors=[ValidationIssue("error", "NULL_DF", "Dataframe is None")])
    if not isinstance(df, pd.DataFrame):
        return ValidationResult(is_valid=False, errors=[ValidationIssue("error", "NOT_DF", f"Expected DataFrame, got {type(df)}")])
    if df.empty:
        return ValidationResult(is_valid=False, errors=[ValidationIssue("error", "EMPTY", "Dataset is empty — no rows to analyze")])

    stats["rows"] = int(len(df))
    stats["columns"] = list(df.columns)

    # 1) Required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(ValidationIssue("error", "MISSING_COLUMNS", f"Missing required columns: {missing}", count=len(missing), sample=str(missing)))

    # Early exit if no required columns at all
    if "cse_id" not in df.columns or "alert_id" not in df.columns:
        stats["missing_required"] = missing
        return ValidationResult(is_valid=False, errors=errors, warnings=warnings, stats=stats)

    # 2) Missing CSE IDs
    if df["cse_id"].isna().sum() > 0 or (df["cse_id"].astype(str).str.strip() == "").sum() > 0:
        n = int(df["cse_id"].isna().sum() + (df["cse_id"].astype(str).str.strip() == "").sum())
        errors.append(ValidationIssue("error", "MISSING_CSE_ID", "Rows with missing/empty cse_id", count=n, sample=_sample_str(df.loc[df["cse_id"].isna(), "alert_id"])))

    # 3) Duplicate alert_id
    if "alert_id" in df.columns:
        dup = int(df.duplicated(subset=["alert_id"]).sum())
        if dup > 0:
            errors.append(ValidationIssue("error", "DUP_ALERT_ID", f"Duplicate alert_id values (must be unique)", count=dup, sample=_sample_str(df.loc[df.duplicated("alert_id", keep=False), "alert_id"])))
        stats["unique_alerts"] = int(df["alert_id"].nunique())

    # 4) Severity allowlist
    if "severity" in df.columns:
        bad = df[~df["severity"].astype(str).str.upper().str.strip().isin(SEVERITY_ALLOWLIST)]
        if len(bad) > 0:
            warnings.append(ValidationIssue("warning", "BAD_SEVERITY", f"Invalid severity values (allow {SEVERITY_ALLOWLIST})", count=len(bad), sample=_sample_str(bad["severity"])))

    # 5) Disposition allowlist (if present)
    if "disposition" in df.columns:
        bad = df[~df["disposition"].astype(str).str.strip().isin(DISPOSITION_ALLOWLIST)]
        # Allow NaN? No, disposition is required — NaN is an error
        if len(bad) > 0:
            warnings.append(ValidationIssue("warning", "BAD_DISPOSITION", f"Unknown disposition values", count=len(bad), sample=_sample_str(bad["disposition"])))

    # 6) Asset criticality
    if "asset_criticality" in df.columns:
        bad = df[~df["asset_criticality"].astype(str).str.upper().str.strip().isin(ASSET_CRITICALITY_ALLOWLIST)]
        if len(bad) > 0:
            warnings.append(ValidationIssue("warning", "BAD_ASSET_CRIT", f"Invalid asset_criticality", count=len(bad), sample=_sample_str(bad["asset_criticality"])))

    # 7) Timestamps — parse check
    for col in TIME_COLUMNS:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce")
        # For required time cols, null is error; for optional, warning
        null_n = int(parsed.isna().sum() - df[col].isna().sum())  # newly failed parses
        # Count rows where original non-null but parsed failed
        failed = int(((df[col].notna()) & (parsed.isna())).sum())
        if failed > 0:
            # investigation_start/end may be legitimately null (no investigation)
            level = "warning" if col in ("investigation_start", "investigation_end") else "error"
            bucket = warnings if level == "warning" else errors
            bucket.append(ValidationIssue(level, f"BAD_TIME_{col.upper()}", f"Unparseable timestamps in {col}", count=failed, sample=_sample_str(df.loc[(df[col].notna()) & (parsed.isna()), col])))

    # 8) Time sequence checks (vectorized, NaT-safe)
    try:
        ct = pd.to_datetime(df["created_time"], errors="coerce")
        at = pd.to_datetime(df["acknowledged_time"], errors="coerce")
        cl = pd.to_datetime(df["closure_time"], errors="coerce")
        ist = pd.to_datetime(df["investigation_start"], errors="coerce") if "investigation_start" in df.columns else pd.Series([pd.NaT]*len(df))
        iet = pd.to_datetime(df["investigation_end"], errors="coerce") if "investigation_end" in df.columns else pd.Series([pd.NaT]*len(df))

        # closure before created/ack
        bad_cl_before_ct = int(((cl.notna()) & (ct.notna()) & (cl < ct)).sum())
        if bad_cl_before_ct > 0:
            errors.append(ValidationIssue("error", "TIME_CLOSURE_BEFORE_CREATE", "closure_time is before created_time", count=bad_cl_before_ct))
        bad_cl_before_at = int(((cl.notna()) & (at.notna()) & (cl < at)).sum())
        if bad_cl_before_at > 0:
            warnings.append(ValidationIssue("warning", "TIME_CLOSURE_BEFORE_ACK", "closure_time is before acknowledged_time", count=bad_cl_before_at))

        # investigation before ack
        bad_ist_before_at = int(((ist.notna()) & (at.notna()) & (ist < at)).sum())
        if bad_ist_before_at > 0:
            warnings.append(ValidationIssue("warning", "TIME_INVEST_BEFORE_ACK", "investigation_start is before acknowledged_time", count=bad_ist_before_at))

        # investigation_end before start
        bad_iet_before_ist = int(((iet.notna()) & (ist.notna()) & (iet < ist)).sum())
        if bad_iet_before_ist > 0:
            errors.append(ValidationIssue("error", "TIME_INVEST_END_BEFORE_START", "investigation_end before investigation_start", count=bad_iet_before_ist))

        # Negative durations (created->closure negative already caught, but explicit)
        dur = (cl - ct).dt.total_seconds() / 60
        neg = int((dur < 0).sum())
        if neg > 0:
            # already reported as closure_before_create, avoid double-count; just stat
            stats["negative_durations"] = neg

    except Exception as e:
        warnings.append(ValidationIssue("warning", "TIME_CHECK_FAILED", f"Time sequence checks skipped due to error: {e}"))

    # 9) Boolean fields — check for unexpected string values
    for col in ["escalated", "investigation_evidence"]:
        if col not in df.columns:
            continue
        # Native bool is fine; object strings should be in allowlist
        if df[col].dtype == object:
            s = df[col].astype(str).str.lower().str.strip()
            # Allow "true"/"false" plus allowlist
            from src.utils.constants import BOOLEAN_TRUE_SET, BOOLEAN_FALSE_SET
            allowed = BOOLEAN_TRUE_SET | BOOLEAN_FALSE_SET
            bad = s[~s.isin(allowed)]
            if len(bad) > 0:
                warnings.append(ValidationIssue("warning", f"BAD_BOOL_{col.upper()}", f"Unexpected values in {col} (expected boolean)", count=len(bad), sample=_sample_str(bad)))

    # 10) Numeric ranges
    for col in ["expected_activity", "observed_activity"]:
        if col in df.columns:
            neg = int((pd.to_numeric(df[col], errors="coerce") < 0).sum())
            if neg > 0:
                errors.append(ValidationIssue("error", f"NEG_{col.upper()}", f"Negative values in {col}", count=neg))
            # observed >> expected unrealistic but not error — warn if > 5x
            if col == "observed_activity" and "expected_activity" in df.columns:
                exp = pd.to_numeric(df["expected_activity"], errors="coerce")
                obs = pd.to_numeric(df["observed_activity"], errors="coerce")
                weird = int(((obs > exp * 5) & (exp > 0)).sum())
                if weird > 0:
                    warnings.append(ValidationIssue("warning", "OBSERVED_GT_EXPECTED", "observed_activity > 5× expected_activity (possible data error)", count=weird))

    # Final validity: any error → invalid (warnings alone still valid)
    is_valid = len(errors) == 0
    stats["error_count"] = len(errors)
    stats["warning_count"] = len(warnings)

    return ValidationResult(is_valid=is_valid, errors=errors, warnings=warnings, stats=stats)
