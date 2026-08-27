import pandas as pd


def detect_execution_gaps(df):
    """
    Detect potential execution gaps in SOC operations.
    """

    df = df.copy()

    # Default values
    df["execution_gap"] = False
    df["execution_gap_reason"] = ""

    # Rule 1:
    # Critical alert closed in less than 5 minutes
    rapid_critical = (
        (df["severity"] == "CRITICAL") &
        (df["closure_minutes"] < 5)
    )

    # Rule 2:
    # No investigation evidence
    no_investigation = (
        df["investigation_evidence"] == False
    )

    # Rule 3:
    # Alert was not escalated
    not_escalated = (
        df["escalated"] == False
    )

    # Strong execution gap
    strong_gap = (
        rapid_critical &
        no_investigation
    )

    df.loc[
        strong_gap,
        "execution_gap"
    ] = True

    df.loc[
        strong_gap,
        "execution_gap_reason"
    ] = (
        "Critical alert closed very quickly "
        "without investigation evidence."
    )

    # Secondary execution gap
    secondary_gap = (
        rapid_critical &
        not_escalated &
        ~strong_gap
    )

    df.loc[
        secondary_gap,
        "execution_gap"
    ] = True

    df.loc[
        secondary_gap,
        "execution_gap_reason"
    ] = (
        "Critical alert closed very quickly "
        "without escalation."
    )

    return df