from src.risk_scoring import calculate_risk_scores
from src.data_loader import load_soc_data
from src.preprocessing import preprocess_data
from src.execution_gap import detect_execution_gaps
from src.negative_space import detect_negative_space
from src.anomaly_detection import detect_anomalies
from src.explainability import generate_explanation


# Load data
file_path = "data/raw/soc_alerts.csv"

df = load_soc_data(file_path)

# Preprocess
df = preprocess_data(df)

# Execution gap detection
df = detect_execution_gaps(df)

# Negative space detection
df = detect_negative_space(df)

df = detect_negative_space(df)

risk_summary = calculate_risk_scores(df)

anomaly_summary = detect_anomalies(df)

print("\n===================================")
print("SAT-SA ANALYTICS")
print("===================================")

print("\nTotal records:")
print(len(df))


# -----------------------------------
# Execution Gap
# -----------------------------------

execution_gaps = df["execution_gap"].sum()

print("\nPotential execution gaps:")
print(execution_gaps)


# -----------------------------------
# Negative Space
# -----------------------------------

negative_space = df["negative_space"].sum()

print("\nPotential visibility gaps:")
print(negative_space)


# -----------------------------------
# Activity Coverage
# -----------------------------------

print("\nAverage activity coverage:")

print(
    f"{df['activity_coverage'].mean():.2f}%"
)


# -----------------------------------
# CSE Summary
# -----------------------------------

summary = df.groupby("cse_id").agg(

    total_alerts=("alert_id", "count"),

    execution_gaps=("execution_gap", "sum"),

    negative_space=("negative_space", "sum"),

    avg_activity_coverage=(
        "activity_coverage",
        "mean"
    )
)


print("\n===================================")
print("CSE SUMMARY")
print("===================================")

print(summary.round(2))


# -----------------------------------
# Example Negative Space Findings
# -----------------------------------

print("\n===================================")
print("EXAMPLE VISIBILITY GAPS")
print("===================================")

findings = df[
    df["negative_space"]
][[
    "cse_id",
    "asset_id",
    "expected_activity",
    "observed_activity",
    "activity_coverage",
    "negative_space_reason"
]]

print(
    findings.head(10).to_string(index=False)
)

print("\n===================================")
print("SUPERVISORY RISK PRIORITY")
print("===================================")

print(
    risk_summary[
        [
            "priority",
            "cse_id",
            "risk_score",
            "risk_level",
            "execution_gaps",
            "negative_space",
            "avg_activity_coverage"
        ]
    ].to_string(index=False)
)

print("\n===================================")
print("ML ANOMALY DETECTION")
print("===================================")

print(
    anomaly_summary[
        [
            "cse_id",
            "anomaly",
            "anomaly_score",
            "anomaly_reason"
        ]
    ].to_string(index=False)
)

print("\n===================================")
print("SUPERVISORY EXPLANATION")
print("===================================")

# Take highest-risk CSE
top_cse = risk_summary.iloc[0]

cse_id = top_cse["cse_id"]

# Find corresponding anomaly information
anomaly_row = anomaly_summary[
    anomaly_summary["cse_id"] == cse_id
].iloc[0]

explanation = generate_explanation(
    cse_id,
    top_cse,
    anomaly_row
)

print(f"\nCSE: {explanation['cse_id']}")

print(
    f"Risk Score: "
    f"{explanation['risk_score']:.2f}/100"
)

print(
    f"Priority: "
    f"{explanation['priority']}"
)

print("\nWHY WAS IT FLAGGED?")

for reason in explanation["reasons"]:
    print(f"• {reason}")