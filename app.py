import streamlit as st
import pandas as pd
import plotly.express as px


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="SAT-SA | SOC Supervisory Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SIMPLE THEME-SAFE CSS
# ============================================================
# IMPORTANT:
# We intentionally do NOT create HTML cards for the main UI.
# Streamlit native components automatically adapt to
# Light/Dark mode.
# ============================================================

st.markdown(
    """
    <style>

    /* Page spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* Sidebar spacing */
    section[data-testid="stSidebar"] {
        padding-top: 1rem;
    }

    /* Small spacing helpers */
    .small-text {
        font-size: 13px;
        opacity: 0.75;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        background: rgba(34, 197, 94, 0.12);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.30);
    }

    /* Footer */
    .footer {
        text-align: center;
        opacity: 0.65;
        font-size: 12px;
        padding-top: 20px;
        padding-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# IMPORT SAT-SA MODULES
# ============================================================

try:

    from src.data_loader import load_soc_data
    from src.preprocessing import preprocess_data
    from src.execution_gap import detect_execution_gaps
    from src.negative_space import detect_negative_space
    from src.risk_scoring import calculate_risk_scores
    from src.anomaly_detection import detect_anomalies

except ImportError as e:

    st.error("SAT-SA module import failed.")

    st.code(str(e))

    st.info(
        "Make sure you are running Streamlit from "
        "the SAT-SA project directory."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛡️ SAT-SA")

    st.caption(
        "Supervisory Analytics Tool for SOC Assessment"
    )

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Select page",
        [
            "Overview",
            "CSE Investigation",
            "Supporting Records",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.subheader("Deployment")

    st.success("● OFFLINE / LOCAL")

    st.caption(
        "All analytics run locally. "
        "No cloud or external AI APIs are required."
    )

    st.divider()

    st.subheader("Data Input")

    uploaded_file = st.file_uploader(
        "Upload SOC dataset",
        type=["csv"],
        help="Upload a CSV containing SOC alert/case records.",
    )


# ============================================================
# LOAD DATA
# ============================================================

try:

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

    else:

        df = load_soc_data(
            "data/raw/soc_alerts.csv"
        )

except Exception as e:

    st.error("Unable to load the SOC dataset.")

    st.code(str(e))

    st.stop()


# ============================================================
# VALIDATE BASIC DATA
# ============================================================

if df is None or df.empty:

    st.warning(
        "The SOC dataset is empty."
    )

    st.stop()


if "cse_id" not in df.columns:

    st.error(
        "The dataset does not contain the required "
        "`cse_id` column."
    )

    st.stop()


# ============================================================
# RUN SAT-SA ANALYTICS PIPELINE
# ============================================================

try:

    # 1. Cleaning / preprocessing
    df = preprocess_data(df)

    # 2. Execution gap detection
    df = detect_execution_gaps(df)

    # 3. Negative-space detection
    df = detect_negative_space(df)

    # 4. Risk scoring
    risk_summary = calculate_risk_scores(df)

    # 5. ML anomaly detection
    anomaly_summary = detect_anomalies(df)

except Exception as e:

    st.error(
        "SAT-SA analytics pipeline failed."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# NORMALIZE RESULT DATAFRAMES
# ============================================================

if not isinstance(risk_summary, pd.DataFrame):

    risk_summary = pd.DataFrame(risk_summary)


if not isinstance(anomaly_summary, pd.DataFrame):

    anomaly_summary = pd.DataFrame(anomaly_summary)


# ============================================================
# GLOBAL METRICS
# ============================================================

total_cses = df["cse_id"].nunique()

total_records = len(df)


if "execution_gap" in df.columns:

    execution_gaps = int(
        df["execution_gap"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

else:

    execution_gaps = 0


if "negative_space" in df.columns:

    negative_spaces = int(
        df["negative_space"]
        .fillna(False)
        .astype(bool)
        .sum()
    )

else:

    negative_spaces = 0


# High-risk count
high_risk = 0

if (
    "risk_level" in risk_summary.columns
):

    high_risk = int(
        (
            risk_summary["risk_level"]
            .astype(str)
            .str.upper()
            == "HIGH"
        ).sum()
    )


# ============================================================
# HEADER
# ============================================================

header_col1, header_col2 = st.columns(
    [5, 1]
)

with header_col1:

    st.title("🛡️ SAT-SA")

    st.caption(
        "Supervisory Analytics Tool for SOC Assessment"
    )

with header_col2:

    st.markdown(
        "<div style='text-align:right; margin-top:18px;'>"
        "<span class='status-badge'>● SYSTEM ONLINE</span>"
        "</div>",
        unsafe_allow_html=True,
    )


st.divider()


# ============================================================
# TOP KPI SECTION
# ============================================================

st.subheader("Supervisory Overview")

st.caption(
    "High-level view of SOC operational behaviour "
    "across Critical Sector Entities."
)


k1, k2, k3, k4, k5 = st.columns(5)


with k1:

    st.metric(
        label="CSEs Assessed",
        value=f"{total_cses:,}",
        help="Number of Critical Sector Entities in the dataset.",
    )


with k2:

    st.metric(
        label="SOC Records",
        value=f"{total_records:,}",
        help="Total alert/case records processed.",
    )


with k3:

    st.metric(
        label="Execution Gaps",
        value=f"{execution_gaps:,}",
        help="Potential gaps between expected and observed SOC execution.",
    )


with k4:

    st.metric(
        label="Negative Space",
        value=f"{negative_spaces:,}",
        help="Potential visibility or monitoring gaps.",
    )


with k5:

    st.metric(
        label="High Priority",
        value=f"{high_risk:,}",
        help="CSEs currently classified as high supervisory priority.",
    )


st.divider()


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":

    st.header("Supervisory Priority")

    st.caption(
        "Risk-ranked view of CSEs requiring supervisory attention."
    )


    # --------------------------------------------------------
    # PREPARE RISK TABLE
    # --------------------------------------------------------

    ranking = risk_summary.copy()


    # Make sure risk_score exists
    if "risk_score" not in ranking.columns:

        st.error(
            "`risk_score` was not found in the risk scoring output."
        )

        st.dataframe(
            ranking,
            use_container_width=True,
            hide_index=True,
        )

        st.stop()


    # Sort
    ranking = ranking.sort_values(
        "risk_score",
        ascending=False,
    ).reset_index(drop=True)


    # CSE column
    if "cse_id" not in ranking.columns:

        st.error(
            "`cse_id` was not found in the risk scoring output."
        )

        st.stop()


    # --------------------------------------------------------
    # DISPLAY TABLE
    # --------------------------------------------------------

    display_ranking = pd.DataFrame()

    display_ranking["CSE"] = ranking[
        "cse_id"
    ]


    display_ranking["Risk Score"] = (
        ranking["risk_score"]
        .round(1)
    )


    if "risk_level" in ranking.columns:

        display_ranking["Priority"] = (
            ranking["risk_level"]
            .astype(str)
            .str.upper()
            .map(
                {
                    "HIGH": "🔴 HIGH",
                    "MEDIUM": "🟠 MEDIUM",
                    "LOW": "🟢 LOW",
                }
            )
            .fillna("⚪ REVIEW")
        )

    else:

        display_ranking["Priority"] = "⚪ REVIEW"


    if "execution_gaps" in ranking.columns:

        display_ranking["Execution Gaps"] = (
            ranking["execution_gaps"]
            .fillna(0)
            .astype(int)
        )

    else:

        display_ranking["Execution Gaps"] = 0


    if "negative_space" in ranking.columns:

        display_ranking["Negative Space"] = (
            ranking["negative_space"]
            .fillna(0)
            .astype(int)
        )

    else:

        display_ranking["Negative Space"] = 0


    if (
        "avg_activity_coverage"
        in ranking.columns
    ):

        display_ranking["Activity Coverage"] = (
            ranking["avg_activity_coverage"]
            .round(1)
            .astype(str)
            + "%"
        )

    else:

        display_ranking["Activity Coverage"] = "N/A"


    st.dataframe(
        display_ranking,
        use_container_width=True,
        hide_index=True,
    )


    st.divider()


    # ========================================================
    # CHARTS
    # ========================================================

    chart_left, chart_right = st.columns(2)


    # --------------------------------------------------------
    # RISK SCORE CHART
    # --------------------------------------------------------

    with chart_left:

        st.subheader("Risk Score by CSE")

        chart_data = ranking[
            [
                "cse_id",
                "risk_score",
            ]
        ].copy()


        chart_data = chart_data.sort_values(
            "risk_score"
        )


        fig = px.bar(
            chart_data,
            x="risk_score",
            y="cse_id",
            orientation="h",
            text="risk_score",
        )


        fig.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
        )


        fig.update_layout(
            height=400,
            margin=dict(
                l=10,
                r=30,
                t=20,
                b=20,
            ),
            xaxis_title="Risk Score",
            yaxis_title="",
        )


        st.plotly_chart(
            fig,
            use_container_width=True,
        )


    # --------------------------------------------------------
    # ACTIVITY COVERAGE
    # --------------------------------------------------------

    with chart_right:

        st.subheader(
            "Security Activity Coverage"
        )


        if (
            "avg_activity_coverage"
            in ranking.columns
        ):

            coverage_data = ranking[
                [
                    "cse_id",
                    "avg_activity_coverage",
                ]
            ].copy()


            fig2 = px.bar(
                coverage_data,
                x="cse_id",
                y="avg_activity_coverage",
                text="avg_activity_coverage",
            )


            fig2.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside",
            )


            fig2.update_layout(
                height=400,
                margin=dict(
                    l=10,
                    r=20,
                    t=20,
                    b=20,
                ),
                xaxis_title="CSE",
                yaxis_title="Coverage (%)",
                yaxis=dict(
                    range=[0, 100]
                ),
            )


            st.plotly_chart(
                fig2,
                use_container_width=True,
            )

        else:

            st.info(
                "Activity coverage information "
                "is not available."
            )


    st.divider()


    # ========================================================
    # SUPERVISORY SIGNALS
    # ========================================================

    st.subheader("Supervisory Signals")

    st.caption(
        "Comparison of the main signals identified "
        "by the analytics engine."
    )


    signal_columns = [
        "cse_id"
    ]


    if "execution_gaps" in ranking.columns:

        signal_columns.append(
            "execution_gaps"
        )


    if "negative_space" in ranking.columns:

        signal_columns.append(
            "negative_space"
        )


    signal_data = ranking[
        signal_columns
    ].copy()


    rename_map = {
        "cse_id": "CSE",
        "execution_gaps": "Execution Gaps",
        "negative_space": "Negative Space",
    }


    signal_data = signal_data.rename(
        columns=rename_map
    )


    value_columns = [
        col
        for col in [
            "Execution Gaps",
            "Negative Space",
        ]
        if col in signal_data.columns
    ]


    if value_columns:

        fig3 = px.bar(
            signal_data,
            x="CSE",
            y=value_columns,
            barmode="group",
            height=400,
        )


        fig3.update_layout(
            margin=dict(
                l=10,
                r=20,
                t=20,
                b=20,
            ),
            xaxis_title="CSE",
            yaxis_title="Number of Signals",
        )


        st.plotly_chart(
            fig3,
            use_container_width=True,
        )

    else:

        st.info(
            "No supervisory signal data is available."
        )


    st.divider()


    # ========================================================
    # HOW SAT-SA WORKS
    # ========================================================

    st.subheader("How SAT-SA Works")

    flow1, flow2, flow3, flow4, flow5 = st.columns(5)


    with flow1:

        st.info(
            "**1. INGEST**\n\n"
            "SOC alerts and case records are submitted."
        )


    with flow2:

        st.info(
            "**2. NORMALIZE**\n\n"
            "Records are cleaned and standardized."
        )


    with flow3:

        st.info(
            "**3. ANALYZE**\n\n"
            "Execution gaps, negative space and anomalies are detected."
        )


    with flow4:

        st.info(
            "**4. SCORE**\n\n"
            "Signals are combined into supervisory risk scores."
        )


    with flow5:

        st.success(
            "**5. REVIEW**\n\n"
            "Supervisors prioritize entities for manual examination."
        )


# ============================================================
# PAGE 2 — CSE INVESTIGATION
# ============================================================

elif page == "CSE Investigation":

    st.header("CSE Investigation")

    st.caption(
        "Evidence-backed supervisory examination "
        "of a selected Critical Sector Entity."
    )


    # --------------------------------------------------------
    # CSE SELECTION
    # --------------------------------------------------------

    cse_list = sorted(
        risk_summary["cse_id"]
        .dropna()
        .unique()
        .tolist()
    )


    if not cse_list:

        st.warning(
            "No CSEs are available."
        )

        st.stop()


    selected_cse = st.selectbox(
        "Select Critical Sector Entity",
        cse_list,
    )


    # Get risk row
    selected_rows = risk_summary[
        risk_summary["cse_id"]
        == selected_cse
    ]


    if selected_rows.empty:

        st.error(
            "Risk information for this CSE is unavailable."
        )

        st.stop()


    risk_row = selected_rows.iloc[0]


    # Get anomaly row
    anomaly_rows = anomaly_summary[
        anomaly_summary["cse_id"]
        == selected_cse
    ]


    anomaly_detected = False

    anomaly_score = None


    if not anomaly_rows.empty:

        anomaly_row = anomaly_rows.iloc[0]


        if "anomaly" in anomaly_row.index:

            anomaly_value = anomaly_row[
                "anomaly"
            ]


            if isinstance(
                anomaly_value,
                str,
            ):

                anomaly_detected = (
                    anomaly_value.lower()
                    == "true"
                )

            else:

                anomaly_detected = bool(
                    anomaly_value
                )


        if "anomaly_score" in anomaly_row.index:

            try:

                anomaly_score = float(
                    anomaly_row[
                        "anomaly_score"
                    ]
                )

            except:

                anomaly_score = None


    # --------------------------------------------------------
    # PRIORITY
    # --------------------------------------------------------

    risk_level = "REVIEW"


    if "risk_level" in risk_row.index:

        risk_level = str(
            risk_row["risk_level"]
        ).upper()


    if risk_level == "HIGH":

        st.error(
            f"🔴 HIGH SUPERVISORY PRIORITY — {selected_cse}"
        )

    elif risk_level == "MEDIUM":

        st.warning(
            f"🟠 MEDIUM SUPERVISORY PRIORITY — {selected_cse}"
        )

    else:

        st.success(
            f"🟢 {risk_level} SUPERVISORY PRIORITY — {selected_cse}"
        )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    m1, m2, m3, m4 = st.columns(4)


    with m1:

        try:

            score = float(
                risk_row["risk_score"]
            )

            st.metric(
                "Risk Score",
                f"{score:.1f}/100",
            )

        except:

            st.metric(
                "Risk Score",
                "N/A",
            )


    with m2:

        if "execution_gaps" in risk_row.index:

            st.metric(
                "Execution Gaps",
                int(
                    risk_row[
                        "execution_gaps"
                    ]
                ),
            )

        else:

            st.metric(
                "Execution Gaps",
                0,
            )


    with m3:

        if "negative_space" in risk_row.index:

            st.metric(
                "Negative Space",
                int(
                    risk_row[
                        "negative_space"
                    ]
                ),
            )

        else:

            st.metric(
                "Negative Space",
                0,
            )


    with m4:

        if (
            "avg_activity_coverage"
            in risk_row.index
        ):

            try:

                coverage = float(
                    risk_row[
                        "avg_activity_coverage"
                    ]
                )

                st.metric(
                    "Activity Coverage",
                    f"{coverage:.1f}%",
                )

            except:

                st.metric(
                    "Activity Coverage",
                    "N/A",
                )

        else:

            st.metric(
                "Activity Coverage",
                "N/A",
            )


    st.divider()


    # ========================================================
    # ML ANOMALY
    # ========================================================

    st.subheader(
        "🤖 Behavioural Anomaly Detection"
    )


    if anomaly_detected:

        st.warning(
            "⚠️ Behavioural anomaly detected."
        )

        st.write(
            "The operational behaviour of this CSE "
            "deviates from the observed peer pattern."
        )


        if anomaly_score is not None:

            st.metric(
                "Anomaly Score",
                f"{anomaly_score:.2f}",
            )


        st.caption(
            "Important: an anomaly is a supervisory "
            "screening signal. It does not by itself "
            "indicate a cyberattack or compromise."
        )

    else:

        st.success(
            "✓ No significant behavioural anomaly "
            "was detected for this CSE."
        )


    st.divider()


    # ========================================================
    # WHY FLAGGED
    # ========================================================

    st.subheader("🔍 Why Was This CSE Flagged?")


    execution_count = 0

    negative_count = 0


    if "execution_gaps" in risk_row.index:

        try:

            execution_count = int(
                risk_row[
                    "execution_gaps"
                ]
            )

        except:

            execution_count = 0


    if "negative_space" in risk_row.index:

        try:

            negative_count = int(
                risk_row[
                    "negative_space"
                ]
            )

        except:

            negative_count = 0


    findings = []


    if execution_count > 0:

        findings.append(
            (
                "🔴 Execution Gap",
                f"{execution_count:,} potential "
                "execution gaps were identified. "
                "These may indicate a mismatch between "
                "expected SOC handling and observed evidence.",
            )
        )


    if negative_count > 0:

        findings.append(
            (
                "🟠 Negative Space",
                f"{negative_count:,} potential visibility "
                "gaps were identified. Unexpectedly low "
                "security activity may require manual review.",
            )
        )


    if anomaly_detected:

        findings.append(
            (
                "🔵 Behavioural Anomaly",
                "The CSE's operational profile deviates "
                "from the observed peer behaviour.",
            )
        )


    if (
        "avg_activity_coverage"
        in risk_row.index
    ):

        try:

            coverage = float(
                risk_row[
                    "avg_activity_coverage"
                ]
            )


            if coverage < 50:

                findings.append(
                    (
                        "🟠 Low Activity Coverage",
                        f"Observed security activity coverage "
                        f"is {coverage:.1f}%, which may indicate "
                        "a monitoring visibility gap.",
                    )
                )

        except:

            pass


    if findings:

        for title, description in findings:

            with st.container(border=True):

                st.markdown(
                    f"### {title}"
                )

                st.write(
                    description
                )

    else:

        st.success(
            "No major supervisory signal was identified "
            "for this CSE."
        )


    st.divider()


    # ========================================================
    # UNDERLYING RECORDS
    # ========================================================

    st.subheader(
        "📋 Supporting Evidence"
    )

    st.caption(
        "Underlying SOC records used to support "
        "the supervisory findings."
    )


    cse_records = df[
        df["cse_id"]
        == selected_cse
    ].copy()


    if cse_records.empty:

        st.info(
            "No supporting records found."
        )

    else:

        st.write(
            f"Showing {min(len(cse_records), 100):,} "
            f"of {len(cse_records):,} records."
        )


        st.dataframe(
            cse_records.head(100),
            use_container_width=True,
            hide_index=True,
        )


        csv = cse_records.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            label="⬇️ Download CSE Evidence",
            data=csv,
            file_name=(
                f"{selected_cse}_evidence.csv"
            ),
            mime="text/csv",
        )


# ============================================================
# PAGE 3 — SUPPORTING RECORDS
# ============================================================

elif page == "Supporting Records":

    st.header("Supporting SOC Records")

    st.caption(
        "Explore the underlying records processed "
        "by the SAT-SA analytics engine."
    )


    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    f1, f2, f3 = st.columns(3)


    with f1:

        cse_options = sorted(
            df["cse_id"]
            .dropna()
            .unique()
            .tolist()
        )


        selected_cses = st.multiselect(
            "CSE",
            cse_options,
        )


    with f2:

        if "severity" in df.columns:

            severity_options = sorted(
                df["severity"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        else:

            severity_options = []


        selected_severity = st.multiselect(
            "Severity",
            severity_options,
        )


    with f3:

        execution_filter = st.selectbox(
            "Execution Gap",
            [
                "All",
                "Flagged",
                "Not Flagged",
            ],
        )


    # --------------------------------------------------------
    # APPLY FILTERS
    # --------------------------------------------------------

    filtered_df = df.copy()


    if selected_cses:

        filtered_df = filtered_df[
            filtered_df["cse_id"]
            .isin(selected_cses)
        ]


    if (
        selected_severity
        and "severity" in filtered_df.columns
    ):

        filtered_df = filtered_df[
            filtered_df["severity"]
            .astype(str)
            .isin(selected_severity)
        ]


    if (
        execution_filter != "All"
        and "execution_gap" in filtered_df.columns
    ):

        if execution_filter == "Flagged":

            filtered_df = filtered_df[
                filtered_df[
                    "execution_gap"
                ]
                .fillna(False)
                .astype(bool)
            ]

        else:

            filtered_df = filtered_df[
                ~filtered_df[
                    "execution_gap"
                ]
                .fillna(False)
                .astype(bool)
            ]


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    st.info(
        f"{len(filtered_df):,} records "
        "match the selected filters."
    )


    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
    )


    csv_filtered = filtered_df.to_csv(
        index=False
    ).encode("utf-8")


    st.download_button(
        label="⬇️ Download Filtered Dataset",
        data=csv_filtered,
        file_name="sat_sa_filtered_dataset.csv",
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div class="footer">

        <b>SAT-SA</b> · Supervisory Analytics Tool for SOC Assessment

        <br>

        Offline analytical prototype · Human-in-the-loop assessment

        <br><br>

        Findings generated by SAT-SA are supervisory screening
        signals and require human validation.

    </div>
    """,
    unsafe_allow_html=True,
)