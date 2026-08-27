import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(df):
    """
    Detect unusual SOC operational behaviour using
    Isolation Forest.

    The model identifies statistical outliers.
    It does NOT determine whether an entity is compromised.
    """

    df = df.copy()

    # -----------------------------------
    # Create CSE-level behavioural profile
    # -----------------------------------

    # Ensure expected columns exist (backward compat with new rule engine)
    if "execution_gap" not in df.columns:
        # New pipeline uses rule_R05 / rapid_closure; fallback to rapid_closure
        if "rapid_closure" in df.columns:
            df["execution_gap"] = df["rapid_closure"].fillna(False).astype(bool)
        elif "rule_R05" in df.columns:
            df["execution_gap"] = df["rule_R05"].fillna(False).astype(bool)
        else:
            df["execution_gap"] = False
    if "negative_space" not in df.columns:
        df["negative_space"] = False
    if "closure_minutes" not in df.columns and "resolution_time" in df.columns:
        df["closure_minutes"] = df["resolution_time"]
    if "activity_coverage" not in df.columns:
        df["activity_coverage"] = 100.0
    if "escalated" not in df.columns:
        df["escalated"] = False

    profile = df.groupby("cse_id").agg(

        total_alerts=("alert_id", "count"),

        critical_alerts=(
            "severity",
            lambda x: (x == "CRITICAL").sum()
        ),

        execution_gaps=("execution_gap", "sum"),

        negative_space=("negative_space", "sum"),

        avg_activity_coverage=(
            "activity_coverage",
            "mean"
        ),

        avg_closure_minutes=(
            "closure_minutes",
            "mean"
        ),

        escalation_rate=(
            "escalated",
            "mean"
        )
    ).reset_index()


    # -----------------------------------
    # Convert boolean rate to percentage
    # -----------------------------------

    profile["escalation_rate"] *= 100


    # -----------------------------------
    # Features for ML
    # -----------------------------------

    features = [
        "total_alerts",
        "critical_alerts",
        "execution_gaps",
        "negative_space",
        "avg_activity_coverage",
        "avg_closure_minutes",
        "escalation_rate"
    ]

    X = profile[features]


    # -----------------------------------
    # Isolation Forest
    # -----------------------------------

    model = IsolationForest(
        contamination=0.20,
        random_state=42
    )

    model.fit(X)


    # Prediction

    profile["anomaly_prediction"] = model.predict(X)


    # -1 = anomaly
    #  1 = normal

    profile["anomaly"] = (
        profile["anomaly_prediction"] == -1
    )


    # -----------------------------------
    # Anomaly score
    # -----------------------------------

    profile["anomaly_score"] = (
        -model.score_samples(X)
    )


    profile["anomaly_score"] = (
        profile["anomaly_score"]
        .round(4)
    )


    # -----------------------------------
    # Explanation
    # -----------------------------------

    profile["anomaly_reason"] = ""

    profile.loc[
        profile["anomaly"],
        "anomaly_reason"
    ] = (
        "Operational behaviour significantly "
        "deviates from the observed peer baseline."
    )


    return profile