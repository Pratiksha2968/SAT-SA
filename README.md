# SAT-SA — Supervisory Analytics Tool for SOC Assessment

> **Prototype supervisory analytics platform designed to assist authorized cybersecurity supervisors in prioritizing SOC assessments.**
> Decision support, not automated certification or final security judgment.
> **Offline · Local · No cloud · No external AI APIs**

SAT-SA analyses **offline SOC case-management data** from multiple Critical Sector Entities (CSEs) to answer:
**Which CSE requires supervisory attention, why, and what evidence supports that decision?**

---

## 1. Problem Being Solved
Supervisors oversee multiple CSE SOCs but lack a standardized, explainable way to prioritize attention. SAT-SA provides:
- Execution-gap detection (fast closure without evidence)
- Negative-space / visibility-gap detection (missing expected activity)
- Behavioural anomaly detection (IsolationForest)
- Peer comparison
- Explainable 0–100 supervisory risk score with evidence drill-down

It is **NOT** a SIEM, IDS, or attack-detection system.

---

## 2. Architecture

```
CSE SOC Data (CSV)
      ↓
Data Ingestion (data_loader.py)
      ↓
Data Validation & Normalization (preprocessing.py)
      ↓
Analytics Engine
 ├── Execution Gap Detection (execution_gap.py)
 ├── Negative Space Detection (negative_space.py)
 ├── Anomaly Detection — IsolationForest (anomaly_detection.py)
 ├── Peer Comparison (peer_comparison.py)
 └── Trend Analysis
      ↓
Supervisory Risk Scoring (risk_scoring.py)  — Prototype Supervisory Risk Model
      ↓
CSE Prioritization
      ↓
Explainable Streamlit Dashboard (app.py)
      ↓
Human Supervisor
```

All processing: `CSV/SQLite → Python → Scikit-learn → Streamlit` — fully offline.

---

## 3. Technology Stack
- Python 3.11+
- Streamlit, Pandas, NumPy, Scikit-learn, Plotly, SQLite/CSV
- No cloud, no OpenAI/Gemini/Claude APIs, no internet required

---

## 4. Project Structure (Phase 1 complete)

```
SAT-SA/
├── app.py
├── requirements.txt
├── README.md
├── generate_data.py
├── data/
│   ├── raw/soc_alerts.csv      # generated synthetic dataset
│   └── processed/              # for cleaned outputs
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── execution_gap.py
│   ├── negative_space.py
│   ├── anomaly_detection.py
│   ├── peer_comparison.py
│   ├── risk_scoring.py
│   └── utils.py
└── tests/test_analytics.py
```

---

## 5. Phase 1 — What Was Built

**Phase 1 includes:**
- Project structure
- `requirements.txt`
- `generate_data.py` (synthetic dataset generator)
- Sample dataset `data/raw/soc_alerts.csv`

### Dataset
`generate_data.py` creates **~4000 rows** across 5 CSEs with intentional behavioural differences:

| CSE   | Behaviour |
|-------|-----------|
| CSE-A | Generally healthy |
| CSE-B | Moderate weaknesses |
| CSE-C | **Intentionally highest-risk** (many critical alerts rapidly closed, low investigation evidence ~42%, escalation ~3%, low activity coverage ~32%) |
| CSE-D | Generally healthy |
| CSE-E | Some anomalies / moderate weaknesses |

Columns (17 total):
`cse_id, alert_id, asset_id, severity, alert_type, created_time, acknowledged_time, investigation_start, investigation_end, closure_time, escalated, disposition, investigation_evidence, investigation_notes, asset_criticality, expected_activity, observed_activity`

---

## 6. Installation

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 7. Generate Sample Data

```bash
python generate_data.py
```

Expected output:
```
=======================================================
SAT-SA synthetic dataset generated successfully!
=======================================================
Records generated : 4000
CSEs              : 5  (CSE-A, CSE-B, CSE-C, CSE-D, CSE-E)
...
Output file       : data/raw/soc_alerts.csv
```

Verify:
```bash
head data/raw/soc_alerts.csv
wc -l data/raw/soc_alerts.csv
```

---

## 8. Run the Dashboard (after Phase 5)

```bash
streamlit run app.py
# open http://localhost:8501
```

---

## 9. Offline Requirement
All analytics run locally. No telemetry leaves the machine. No internet is needed after `pip install`.

---

## 10. Demo Scenario
CSE-C is the designed high-risk entity:
- 26–28% critical-alert rate vs ~9–17% peers
- ~14% execution-gap rate vs <3% peers
- ~90% negative-space rate vs <12% peers
- ~32% activity coverage vs >64% peers
Dashboard automatically ranks CSE-C at the top and shows **WHY** and **WHAT EVIDENCE**.

---

## 11. Limitations
- Synthetic data only — not real SOC telemetry
- Risk weights are prototype defaults, not official NCIIPC weights
- IsolationForest uses only 5 CSE aggregates (small sample) — illustrative only

## 12. Future Improvements
- SQLite backend, PDF report, peer z-scores, time-series trends, configurable thresholds UI

---

## 13. Disclaimer
> SAT-SA is a prototype supervisory analytics platform designed to assist authorized cybersecurity supervisors in prioritizing SOC assessments. It provides decision support, not automated certification or final security judgment.
