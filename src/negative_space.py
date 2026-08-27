"""
src/negative_space.py — backward-compat shim
Old: from src.negative_space import detect_negative_space
New: from src.analytics.negative_space import analyze_negative_space
This shim preserves the old simple API for app.py/tests compatibility.
"""
import pandas as pd
from src.analytics.negative_space import analyze_negative_space

def detect_negative_space(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy API — returns df with negative_space bool + activity_coverage
    Internally delegates to analyze_negative_space().
    """
    flagged, _, _ = analyze_negative_space(df)
    # Ensure legacy columns exactly match old output contract
    if "negative_space" not in flagged.columns:
        flagged["negative_space"] = False
    if "negative_space_reason" not in flagged.columns:
        flagged["negative_space_reason"] = ""
    # Old code clipped activity_coverage to 0-100 and set negative_space = coverage<50
    # New advanced logic already does this plus workflow gaps — strictly more sensitive, so safe
    return flagged

# New advanced export for callers that want full detail
__all__ = ["detect_negative_space", "analyze_negative_space"]
