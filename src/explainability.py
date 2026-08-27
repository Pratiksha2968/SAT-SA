def generate_explanation(
    cse_id,
    risk_row,
    anomaly_row
):
    """
    Generate a human-readable supervisory explanation.
    """

    reasons = []

    # Execution gaps
    if risk_row["execution_gaps"] > 0:
        reasons.append(
            f"{int(risk_row['execution_gaps'])} "
            "potential execution gaps detected."
        )

    # Negative space
    if risk_row["negative_space"] > 0:
        reasons.append(
            f"{int(risk_row['negative_space'])} "
            "potential visibility gaps detected."
        )

    # Activity coverage
    coverage = risk_row["avg_activity_coverage"]

    if coverage < 50:
        reasons.append(
            f"Average security activity coverage "
            f"is only {coverage:.1f}%."
        )

    # ML anomaly
    if anomaly_row["anomaly"]:
        reasons.append(
            "Operational behaviour significantly "
            "deviates from the peer baseline."
        )

    # Risk
    risk_score = risk_row["risk_score"]

    if risk_score >= 70:
        priority = "HIGH"
    elif risk_score >= 40:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    return {
        "cse_id": cse_id,
        "risk_score": risk_score,
        "priority": priority,
        "reasons": reasons
    }