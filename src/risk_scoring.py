import pandas as pd


def calculate_risk_scores(df):
    """
    Calculate a supervisory risk score for each CSE.

    SAT-SA uses this score as a screening mechanism to
    prioritize CSEs for manual supervisory review.

    Risk components:
        - Execution Gap       : 40%
        - Negative Space      : 30%
        - Activity Gap        : 30%

    IMPORTANT:
    The score is NOT a cybersecurity compromise score.
    It is a supervisory prioritization score.
    """

    df = df.copy()

    # ==========================================================
    # 1. AGGREGATE CSE-LEVEL METRICS
    # ==========================================================

    summary = df.groupby("cse_id").agg(

        total_alerts=("alert_id", "count"),

        execution_gaps=("execution_gap", "sum"),

        negative_space=("negative_space", "sum"),

        avg_activity_coverage=(
            "activity_coverage",
            "mean"
        ),

        critical_alerts=(
            "severity",
            lambda x: (
                x.astype(str)
                .str.upper()
                == "CRITICAL"
            ).sum()
        )

    ).reset_index()


    # ==========================================================
    # 2. CALCULATE RATES
    # ==========================================================

    summary["execution_gap_rate"] = (
        summary["execution_gaps"]
        / summary["total_alerts"].replace(0, 1)
    ) * 100


    summary["negative_space_rate"] = (
        summary["negative_space"]
        / summary["total_alerts"].replace(0, 1)
    ) * 100


    # ==========================================================
    # 3. NORMALIZE RISK COMPONENTS
    # ==========================================================

    # ----------------------------------------------------------
    # Execution Gap Component
    # ----------------------------------------------------------

    summary["execution_component"] = (
        summary["execution_gap_rate"]
        .clip(lower=0, upper=100)
    )


    # ----------------------------------------------------------
    # Negative Space Component
    # ----------------------------------------------------------

    summary["negative_space_component"] = (
        summary["negative_space_rate"]
        .clip(lower=0, upper=100)
    )


    # ----------------------------------------------------------
    # Activity Gap Component
    # ----------------------------------------------------------

    summary["activity_component"] = (
        100 - summary["avg_activity_coverage"]
    ).clip(lower=0, upper=100)


    # ==========================================================
    # 4. WEIGHTED RISK SCORE
    # ==========================================================

    summary["execution_weight"] = (
        summary["execution_component"] * 0.40
    )


    summary["negative_space_weight"] = (
        summary["negative_space_component"] * 0.30
    )


    summary["activity_weight"] = (
        summary["activity_component"] * 0.30
    )


    summary["risk_score"] = (

        summary["execution_weight"]

        +

        summary["negative_space_weight"]

        +

        summary["activity_weight"]

    ).round(2)


    # ==========================================================
    # 5. RISK CATEGORY
    # ==========================================================

    def classify_risk(score):

        if score >= 70:
            return "HIGH"

        elif score >= 40:
            return "MEDIUM"

        else:
            return "LOW"


    summary["risk_level"] = (
        summary["risk_score"]
        .apply(classify_risk)
    )


    # ==========================================================
    # 6. SUPERVISORY PRIORITY
    # ==========================================================

    summary = summary.sort_values(
        "risk_score",
        ascending=False
    ).reset_index(drop=True)


    summary["priority"] = (
        summary.index + 1
    )


    # ==========================================================
    # 7. SUPERVISORY INTERPRETATION
    # ==========================================================

    def generate_reason(row):

        reasons = []

        if row["execution_component"] >= 20:
            reasons.append(
                "Elevated execution-gap activity"
            )

        if row["negative_space_component"] >= 20:
            reasons.append(
                "Potential monitoring/visibility gaps"
            )

        if row["activity_component"] >= 40:
            reasons.append(
                "Low security activity coverage"
            )

        if not reasons:
            reasons.append(
                "No major supervisory signal"
            )

        return "; ".join(reasons)


    summary["supervisory_reason"] = (
        summary.apply(
            generate_reason,
            axis=1
        )
    )


    # ==========================================================
    # 8. ROUND DISPLAY VALUES
    # ==========================================================

    numeric_columns = [

        "execution_gap_rate",

        "negative_space_rate",

        "avg_activity_coverage",

        "execution_component",

        "negative_space_component",

        "activity_component",

        "execution_weight",

        "negative_space_weight",

        "activity_weight",

        "risk_score"

    ]


    for column in numeric_columns:

        if column in summary.columns:

            summary[column] = (
                summary[column]
                .round(2)
            )


    return summary