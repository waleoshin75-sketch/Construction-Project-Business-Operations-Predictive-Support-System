# Construction Project Business Operations: Predictive Decision Support System

**An Intelligent Multi-Model Engine for Financial, Timelines and Operational Risks Mitigation**

**Python | pandas | scikit-learn | Streamlit** | Six messy tables turned into five working prediction models and a live web app

[Live Streamlit App](https://construction-project-business-operations-predictive-support-sy.streamlit.app/) | [GitHub Repository](https://github.com/waleoshin75-sketch/construction-project-business-operations-predictive-support-system)

---

## At a glance

I took six dirty construction tables, cleaned and modelled them, then built one prediction model for each of five business problems. All five run inside a Streamlit app where a user fills in a form and gets a risk score, a risk level and a suggested action.

| # | Business problem | Best model | Result on unseen projects |
|---|---|---|---|
| 1 | Which cost lines will overrun their budget by more than 15%? | Logistic Regression | ROC AUC 0.83. Reviewing the riskiest 40% of lines finds 58% of overrun dollars |
| 2 | Which tasks will finish more than 5 days late? | Logistic Regression | ROC AUC 0.80. The riskiest 40% of tasks hold 63% of delay days |
| 3 | Which safety incidents will be serious? | Logistic Regression | ROC AUC 0.86. The riskiest 40% of incidents hold 83% of lost worker days |
| 4 | Which material orders will arrive late? | Logistic Regression | ROC AUC 0.60. Weak: a rough guide only |
| 5 | Which orders are priced wrongly? | Isolation Forest | All 62 known anomalies caught in a review list of 150 orders. A one-line price rule caught all 62 too |

The data is fictional and was built with realistic quality problems, so every step of a real workflow is shown. Loss figures are measured in the dollars and days recorded in the tables.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset Overview](#2-dataset-overview)
3. [Data Dictionary](#3-data-dictionary)
4. [Data Quality Audit](#4-data-quality-audit)
5. [Data Cleaning and Error Checking](#5-data-cleaning-and-error-checking)
6. [Feature Engineering](#6-feature-engineering)
7. [Create Relationship Table](#7-create-relationship-table)
8. [Exploratory Data Analysis](#8-exploratory-data-analysis)
9. [Analytical Base Table](#9-analytical-base-table)
10. [Encode Categorical Columns](#10-encode-categorical-columns)
11. [Separate X and y](#11-separate-x-and-y)
12. [Train Test Split](#12-train-test-split)
13. [Feature Scaling of Numerical Columns](#13-feature-scaling-of-numerical-columns)
14. [Train Model](#14-train-model)
15. [Predict X_test](#15-predict-x_test)
16. [Compare X and y](#16-compare-x-and-y)
17. [Evaluate Model](#17-evaluate-model)
18. [Business Analysis](#18-business-analysis)
19. [Save Model, Scaler and Features](#19-save-model-scaler-and-features)
20. [Download Model, Scaler and Features](#20-download-model-scaler-and-features)
21. [Key Findings and Insights](#21-key-findings-and-insights)
22. [Recommendations](#22-recommendations)
23. [Conclusion](#23-conclusion)

---
## 1. Project Overview

A construction firm runs many projects at once, and its cost, schedule, safety and procurement data sits in six separate, untidy tables. Five decisions depend on that data, and each one becomes a model:

| # | Business objective | ML task | Target |
|---|---|---|---|
| 1 | Control project cost overruns | Classification | Cost line overran its budget by more than 15% (46.7% of lines) |
| 2 | Reduce schedule delays | Classification | Task finished more than 5 days late (40.9% of finished tasks) |
| 3 | Improve site safety | Classification | Incident was a lost-time injury or fatality (25.1%) |
| 4 | Make material procurement more reliable | Classification | Order arrived after the promised date (45.9%) |
| 5 | Detect procurement price and invoice anomalies | Unsupervised anomaly detection | None. An answer key is used only to score the model at the end. |

**The workflow:** data quality audit, cleaning, feature engineering, relationship table, EDA, then for each objective an analytical base table, encoding, X/y split, train/test split, scaling, training three models plus a baseline, prediction, evaluation, business analysis, saving and deployment in a Streamlit app.

**Rules I held myself to**
- Whole projects stay together in the train or test split, so the results describe projects the model has never seen.
- No leakage: columns that reveal the answer (actual cost, actual dates, severity, lost days, delivery date, the answer key) never enter a model.
- Every result is turned into a decision: how much of the loss is found by reviewing only the riskiest slice.

**Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, joblib, Streamlit.

### Repository structure

```
├── README.md
├── app.py                          the Streamlit app (section 20)
├── requirements.txt
├── images/                         every chart shown in this README
├── construction_raw/               the six raw (dirty) tables, data_dictionary.csv and business_objectives.csv
├── construction_models/            model, scaler, encoder and features .joblib files (21 files)
└── notebook                        all the code shown below, one cell per section
```

---

## 2. Dataset Overview

Six tables arrive dirty. After cleaning, 135 duplicate rows and 27 rows pointing to projects that do not exist are gone.

| Table | Role | Rows (raw) | Rows (clean) | What it holds |
|---|---|---:|---:|---|
| `dim_projects` | Dimension | 84 | 80 | Project, country, type, contract, budget, size |
| `dim_vendors` | Dimension | 48 | 45 | Supplier and subcontractor ratings and history |
| `fact_cost_ledger` | Fact | 2,465 | 2,413 | Budget and actual cost per project, quarter and category |
| `fact_tasks` | Fact | 4,045 | 3,985 | Planned and actual task dates, crew and status |
| `fact_material_orders` | Fact | 3,035 | 3,000 | Orders, prices, promised and delivered dates |
| `fact_safety_incidents` | Fact | 768 | 750 | Incidents, conditions, worker experience and severity |

After the relationship phase there is also a `dim_date` table (1,938 days, 2022 to 2027) and five modelling tables, one per objective.

### Code: load the six tables

```python
# ============================================================
# CONSTRUCTION BUSINESS OPERATION PREDICTION SYSTEM
# ============================================================

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "construction_raw"

# %%
# Load and display Projects dimension data
df_projects = pd.read_csv(f"{DATA_DIR}/dim_projects.csv")
df_projects.head(5)

# %%
# Load and display Vendors dimension data
df_vendors = pd.read_csv(f"{DATA_DIR}/dim_vendors.csv")
df_vendors.head(5)

# %%
# Load and display Cost Ledger fact data
df_cost_ledger = pd.read_csv(f"{DATA_DIR}/fact_cost_ledger.csv")
df_cost_ledger.head(5)

# %%
# Load and display Material Orders fact data
df_material_orders = pd.read_csv(f"{DATA_DIR}/fact_material_orders.csv")
df_material_orders.head(5)

# %%
# Load and display Safety Incidents fact data
df_safety_incidents = pd.read_csv(f"{DATA_DIR}/fact_safety_incidents.csv")
df_safety_incidents.head(5)

# %%
# Load and display Tasks fact data
df_tasks = pd.read_csv(f"{DATA_DIR}/fact_tasks.csv")
df_tasks.head(5)
```

---

## 3. Data Dictionary

65 columns across the six tables are documented in [`data_dictionary.csv`](construction_raw/data_dictionary.csv), each with its meaning and known data problems. The columns that matter most:

| Table | Key columns |
|---|---|
| `dim_projects` | `project_id`, `country`, `project_type` (Residential, Commercial, Industrial, Infrastructure), `contract_type` (Fixed Price, Cost Plus, Design-Build), `client_sector`, `budget_usd`, `start_date`, `planned_duration_days`, `site_area_sqm`, `num_floors`, `project_manager_experience_years` |
| `dim_vendors` | `vendor_id`, `vendor_type`, `specialty`, `country`, `reliability_rating`, `safety_rating`, `years_active`, `prequalified` |
| `fact_cost_ledger` | `ledger_id`, `period`, `cost_category` (Materials, Labour, Equipment, Subcontractors, Overheads), `budgeted_usd`, `actual_usd`, `approved_change_orders_usd` |
| `fact_tasks` | `task_id`, `task_type`, `planned_start`, `planned_end`, `actual_start`, `actual_end`, `crew_size`, `task_status`, `rework_required` |
| `fact_material_orders` | `order_id`, `material`, `quantity`, `unit`, `unit_price_usd`, `total_cost_usd`, `order_date`, `promised_date`, `delivered_date`, `quality_passed` |
| `fact_safety_incidents` | `incident_id`, `incident_type`, `shift`, `ppe_compliant`, `work_at_height`, `worker_experience_years`, `weather_condition`, `severity` |

**Leakage columns** exist in the data on purpose and are kept out of every model: `actual_usd` and change orders (objective 1), the actual dates and task status (2), `severity`, `lost_days` and `reported_delay_hours` (3), the delivered date and quality result (4), and the answer key `anomaly_answer_key.csv` (5).

The business objectives themselves are listed in [`business_objectives.csv`](construction_raw/business_objectives.csv).

---

## 4. Data Quality Audit

The audit changes nothing. It counts every problem first, one reusable check per table.

| Problem | What the audit found |
|---|---|
| Duplicate rows | 135 across the six tables (tasks 45, orders 35, ledger 30, incidents 18, projects 4, vendors 3) |
| Missing values | Completed-task end date 13.5%, floors on projects 23.8%, change orders 7.9%, crew size 2.0%, vendor on incidents 4.8% |
| Text instead of numbers | 27 project budgets written as `12.5M` or `$12,500,000`; ratings like `4/5` and `five` |
| Period styles | 913 ledger periods not in `YYYY-QN` style (`Q4 2023`, `2023/Q4`) |
| Impossible values | 10 negative actual costs, 12 crew sizes of 0 or 999, 14 non-positive quantities, worker experience above 60 years |
| Impossible dates | 97 tasks ending before they started, 62 planned ends before planned starts, 49 orders delivered before they were ordered |
| Unreadable dates | Junk like `??` and `not delivered` in every date column |
| Spelling variants | 30 spellings of 10 countries, 35 of 7 task types, 29 of 6 incident types |
| Orphan keys | 12 ledger rows and 15 task rows point to projects that do not exist |

### Code: data quality audit

```python
# ============================================================
# DATA QUALITY AUDIT
# ============================================================

# %%
import re
import numpy as np
import pandas as pd

# ---------- use the notebook's df_ table names ----------
dim_projects          = df_projects
dim_vendors           = df_vendors
fact_cost_ledger      = df_cost_ledger
fact_tasks            = df_tasks
fact_material_orders  = df_material_orders
fact_safety_incidents = df_safety_incidents

# ---------- helper functions (used only to MEASURE problems, nothing is changed) ----------
def text_present(series):
    """True where a cell holds something other than blank or NaN."""
    return series.notna() & series.astype("string").str.strip().ne("")

def plain_number(series):
    """Numeric value only when the cell is already a clean number."""
    return pd.to_numeric(series, errors="coerce")

def money_number(series):
    """Number after removing $, USD, commas and a trailing K/M. NaN if still unreadable."""
    def convert(value):
        if pd.isna(value):
            return np.nan
        text = re.sub(r"(?i)usd|\$|,|\s", "", str(value))
        factor = 1
        if text[-1:].upper() == "M":
            factor, text = 1_000_000, text[:-1]
        elif text[-1:].upper() == "K":
            factor, text = 1_000, text[:-1]
        try:
            return float(text) * factor
        except ValueError:
            return np.nan
    return series.map(convert)

def parsed_date(series):
    return series.map(lambda v: pd.to_datetime(v, format="mixed", errors="coerce"))

def clean_id(series):
    return series.astype("string").str.strip().str.upper()

def count_rule(table, check, mask):
    return (table, check, int(mask.sum()))

# ---------- 1. Table overview ----------
audit_tables = {
    "dim_projects":          (dim_projects,          "project_id"),
    "dim_vendors":           (dim_vendors,           "vendor_id"),
    "fact_cost_ledger":      (fact_cost_ledger,      "ledger_id"),
    "fact_tasks":            (fact_tasks,            "task_id"),
    "fact_material_orders":  (fact_material_orders,  "order_id"),
    "fact_safety_incidents": (fact_safety_incidents, "incident_id"),
}

overview = pd.DataFrame([
    {
        "table": name,
        "rows": len(frame),
        "columns": len(frame.columns),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_key": int(frame[key].isna().sum()),
        "duplicate_key_rows": int(frame[key].duplicated(keep=False).sum()),
        "missing_cells": int(frame.isna().sum().sum()),
    }
    for name, (frame, key) in audit_tables.items()
]).set_index("table")

# ---------- 2. Missing values by column ----------
missing = pd.DataFrame([
    {"table": name, "column": column, "missing": int(count),
     "missing_pct": round(100 * count / len(frame), 1)}
    for name, (frame, _) in audit_tables.items()
    for column, count in frame.isna().sum().items()
    if count
])

# ---------- 3. Business rule checks ----------
p, v, l = dim_projects, dim_vendors, fact_cost_ledger
t, m, s = fact_tasks, fact_material_orders, fact_safety_incidents

# numbers and dates used by the rules
budget        = money_number(p["budget_usd"])
duration      = plain_number(p["planned_duration_days"])
site_area     = plain_number(p["site_area_sqm"])
reliability   = plain_number(v["reliability_rating"])
ledger_budget = money_number(l["budgeted_usd"])
ledger_actual = money_number(l["actual_usd"])
ledger_change = money_number(l["approved_change_orders_usd"])
crew          = plain_number(t["crew_size"])
qty           = plain_number(m["quantity"])
unit_price    = money_number(m["unit_price_usd"])
total_cost    = money_number(m["total_cost_usd"])
experience    = plain_number(s["worker_experience_years"])

task_dates = {c: parsed_date(t[c]) for c in ["planned_start", "planned_end", "actual_start", "actual_end"]}
order_dates = {c: parsed_date(m[c]) for c in ["order_date", "promised_date", "delivered_date"]}

rules = pd.DataFrame([
    # dim_projects
    count_rule("dim_projects", "Budget stored as text (symbols or M/K)", text_present(p["budget_usd"]) & plain_number(p["budget_usd"]).isna()),
    count_rule("dim_projects", "Budget unreadable after cleaning", text_present(p["budget_usd"]) & budget.isna()),
    count_rule("dim_projects", "Start date unparsed", text_present(p["start_date"]) & parsed_date(p["start_date"]).isna()),
    count_rule("dim_projects", "Planned duration nonnumeric", text_present(p["planned_duration_days"]) & duration.isna()),
    count_rule("dim_projects", "Planned duration zero or negative", duration.le(0)),
    count_rule("dim_projects", "Site area zero or negative", site_area.le(0)),
    count_rule("dim_projects", "Site area above 1,000,000 sqm", site_area.gt(1_000_000)),
    # dim_vendors
    count_rule("dim_vendors", "Reliability rating nonnumeric", text_present(v["reliability_rating"]) & reliability.isna()),
    count_rule("dim_vendors", "Reliability rating outside 1-5", reliability.notna() & ~reliability.between(1, 5)),
    count_rule("dim_vendors", "Safety rating outside 1-5", v["safety_rating"].notna() & ~v["safety_rating"].between(1, 5)),
    # fact_cost_ledger
    count_rule("fact_cost_ledger", "Budgeted amount unreadable", text_present(l["budgeted_usd"]) & ledger_budget.isna()),
    count_rule("fact_cost_ledger", "Actual amount unreadable", text_present(l["actual_usd"]) & ledger_actual.isna()),
    count_rule("fact_cost_ledger", "Actual amount negative", ledger_actual.lt(0)),
    count_rule("fact_cost_ledger", "Budgeted amount zero or negative", ledger_budget.le(0)),
    count_rule("fact_cost_ledger", "Change orders unreadable", text_present(l["approved_change_orders_usd"]) & ledger_change.isna()),
    count_rule("fact_cost_ledger", "Period not in YYYY-QN style", ~l["period"].astype("string").str.fullmatch(r"\d{4}-Q[1-4]").fillna(False)),
    # fact_tasks
    count_rule("fact_tasks", "Crew size zero, negative or 999", crew.le(0) | crew.ge(999)),
    count_rule("fact_tasks", "Planned start unparsed", text_present(t["planned_start"]) & task_dates["planned_start"].isna()),
    count_rule("fact_tasks", "Planned end unparsed", text_present(t["planned_end"]) & task_dates["planned_end"].isna()),
    count_rule("fact_tasks", "Actual start unparsed", text_present(t["actual_start"]) & task_dates["actual_start"].isna()),
    count_rule("fact_tasks", "Actual end unparsed", text_present(t["actual_end"]) & task_dates["actual_end"].isna()),
    count_rule("fact_tasks", "Actual end before actual start (swapped)", task_dates["actual_end"].lt(task_dates["actual_start"])),
    count_rule("fact_tasks", "Planned end before planned start", task_dates["planned_end"].lt(task_dates["planned_start"])),
    count_rule("fact_tasks", "Actual end missing (task not finished)", t["actual_end"].isna()),
    # fact_material_orders
    count_rule("fact_material_orders", "Quantity zero or negative", qty.le(0)),
    count_rule("fact_material_orders", "Quantity nonnumeric", text_present(m["quantity"]) & qty.isna()),
    count_rule("fact_material_orders", "Unit price unreadable", text_present(m["unit_price_usd"]) & unit_price.isna()),
    count_rule("fact_material_orders", "Total cost unreadable", text_present(m["total_cost_usd"]) & total_cost.isna()),
    count_rule("fact_material_orders", "Total differs from quantity x price by >2%", ((total_cost - qty.abs() * unit_price).abs() / total_cost.replace(0, np.nan)).gt(0.02)),
    count_rule("fact_material_orders", "Order date unparsed", text_present(m["order_date"]) & order_dates["order_date"].isna()),
    count_rule("fact_material_orders", "Promised date unparsed", text_present(m["promised_date"]) & order_dates["promised_date"].isna()),
    count_rule("fact_material_orders", "Delivered date unparsed (text such as 'not delivered')", text_present(m["delivered_date"]) & order_dates["delivered_date"].isna()),
    count_rule("fact_material_orders", "Delivered before ordered", order_dates["delivered_date"].lt(order_dates["order_date"])),
    # fact_safety_incidents
    count_rule("fact_safety_incidents", "Worker experience negative", experience.lt(0)),
    count_rule("fact_safety_incidents", "Worker experience above 60 years", experience.gt(60)),
    count_rule("fact_safety_incidents", "Incident date unparsed", text_present(s["incident_date"]) & parsed_date(s["incident_date"]).isna()),
    count_rule("fact_safety_incidents", "Lost days negative", s["lost_days"].lt(0)),
    count_rule("fact_safety_incidents", "Reported delay hours negative", s["reported_delay_hours"].lt(0)),
], columns=["table", "check", "rows"])
rules = rules[rules["rows"] > 0].reset_index(drop=True)

# ---------- 4. Relationships (orphan keys) ----------
project_ids = set(clean_id(dim_projects["project_id"].dropna()))
vendor_ids = set(clean_id(dim_vendors["vendor_id"].dropna()))

def orphan_row(label, frame, column, valid_ids):
    keys = clean_id(frame[column].dropna())
    return (label, len(keys), int((~keys.isin(valid_ids)).sum()))

relationships = pd.DataFrame([
    orphan_row("fact_cost_ledger.project_id -> dim_projects", fact_cost_ledger, "project_id", project_ids),
    orphan_row("fact_tasks.project_id -> dim_projects", fact_tasks, "project_id", project_ids),
    orphan_row("fact_material_orders.project_id -> dim_projects", fact_material_orders, "project_id", project_ids),
    orphan_row("fact_safety_incidents.project_id -> dim_projects", fact_safety_incidents, "project_id", project_ids),
    orphan_row("fact_tasks.vendor_id -> dim_vendors", fact_tasks, "vendor_id", vendor_ids),
    orphan_row("fact_material_orders.vendor_id -> dim_vendors", fact_material_orders, "vendor_id", vendor_ids),
    orphan_row("fact_safety_incidents.vendor_id -> dim_vendors", fact_safety_incidents, "vendor_id", vendor_ids),
], columns=["relationship", "nonmissing_keys", "orphan_keys"])

# ---------- 5. Spelling variants in category columns ----------
variant_columns = [
    ("dim_projects", dim_projects, "country"),
    ("dim_projects", dim_projects, "project_type"),
    ("dim_projects", dim_projects, "contract_type"),
    ("dim_projects", dim_projects, "client_sector"),
    ("dim_vendors", dim_vendors, "vendor_type"),
    ("dim_vendors", dim_vendors, "country"),
    ("dim_vendors", dim_vendors, "prequalified"),
    ("fact_cost_ledger", fact_cost_ledger, "cost_category"),
    ("fact_tasks", fact_tasks, "task_type"),
    ("fact_tasks", fact_tasks, "task_status"),
    ("fact_tasks", fact_tasks, "rework_required"),
    ("fact_material_orders", fact_material_orders, "material"),
    ("fact_material_orders", fact_material_orders, "unit"),
    ("fact_material_orders", fact_material_orders, "quality_passed"),
    ("fact_safety_incidents", fact_safety_incidents, "incident_type"),
    ("fact_safety_incidents", fact_safety_incidents, "shift"),
    ("fact_safety_incidents", fact_safety_incidents, "ppe_compliant"),
    ("fact_safety_incidents", fact_safety_incidents, "work_at_height"),
    ("fact_safety_incidents", fact_safety_incidents, "weather_condition"),
    ("fact_safety_incidents", fact_safety_incidents, "severity"),
]

def squash(series):
    return series.dropna().astype("string").str.strip().str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()

variants = pd.DataFrame([
    {
        "table": table,
        "column": column,
        "raw_spellings": int(frame[column].dropna().nunique()),
        "after_trim_and_lowercase": int(squash(frame[column]).nunique()),
    }
    for table, frame, column in variant_columns
])

# ---------- 6. Anomaly answer key check (objective 5) ----------
answer_key = pd.read_csv(f"{DATA_DIR}/anomaly_answer_key.csv")
key_summary = pd.DataFrame([
    ("Answer key rows", len(answer_key)),
    ("Unique order ids in orders table", int(fact_material_orders["order_id"].nunique())),
    ("Orders missing from answer key", int((~fact_material_orders["order_id"].isin(answer_key["order_id"])).sum())),
    ("Planted anomalies in key", int(answer_key["is_anomaly"].sum())),
    ("Anomaly rate (%)", round(100 * answer_key["is_anomaly"].mean(), 2)),
], columns=["check", "value"])

display(overview)
display(missing)
display(rules)
display(relationships)
display(variants)
display(key_summary)
```

---

## 5. Data Cleaning and Error Checking

Cleaning is a set of small, reusable helpers applied table by table. Four of them do most of the work: `tidy_text`, `canonical`, `parse_money` and `parse_date`.

| Issue | Fix |
|---|---|
| Duplicate rows | Dropped (135 rows) |
| Keys with different case or spaces | `tidy_text` on every ID column |
| Money written as text | `parse_money` (`12.5M`, `$12,500,000`, `USD 12,500`) |
| Spelling variants | `canonical` maps every spelling to one official label |
| Six date styles | `parse_date`, unreadable values left empty |
| Swapped actual start and end | 39 tasks put back in order |
| Orphan project codes | 27 rows removed (12 ledger, 15 tasks) |
| Impossible values | Set to missing, then filled with the median of the same group (task type, project type, material) |
| Missing quantity | Rebuilt as total cost / unit price (30 orders), else the material median |
| Price spikes | **Kept on purpose**: they are what objective 5 has to find |

**Error checking:** every cleaned table passes automatic checks, and all of them show zero problems: no duplicate keys, no end before start, no delivery before order, no orphan project codes and no unmapped labels.

### Code: data cleaning and error checking

```python
# ============================================================
# DATA CLEANING PHASE
# ============================================================

# %%
import re
from pathlib import Path

CLEAN_DIR = Path("construction_clean")
CLEAN_DIR.mkdir(exist_ok=True)

cleaning_log = []

def log(table, action, rows):
    cleaning_log.append({"table": table, "action": action, "rows": int(rows)})

# ---------- helper functions ----------
MISSING_TOKENS = {"", "n/a", "na", "null", "none", "nan", "-", "tbd", "pending", "??",
                  "00/00/0000", "not delivered"}

def tidy_text(series):
    """Trim, collapse repeated spaces and turn placeholder text into a real missing value."""
    text = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return text.mask(text.str.lower().isin(MISSING_TOKENS))

def squash(series):
    return series.astype("string").str.lower().str.replace(r"[^a-z0-9]+", "", regex=True)

def canonical(series, valid_values, aliases=None):
    """Map any spelling to one official label. Unknown spellings become missing."""
    lookup = {re.sub(r"[^a-z0-9]+", "", v.lower()): v for v in valid_values}
    for alias, official in (aliases or {}).items():
        lookup[re.sub(r"[^a-z0-9]+", "", alias.lower())] = official
    text = tidy_text(series).str.replace("\u00b3", "3", regex=False)   # m\u00b3 becomes m3
    return squash(text).map(lookup).astype("string")

def yes_no(series):
    key = tidy_text(series).str.lower()
    mapping = {"yes": "Yes", "y": "Yes", "true": "Yes", "1": "Yes",
               "no": "No", "n": "No", "false": "No", "0": "No"}
    return key.map(mapping).astype("string")

def parse_number(series):
    return pd.to_numeric(tidy_text(series), errors="coerce")

def parse_money(series):
    """Handles 12.5M, $12,500,000, USD 12,500,000 and '12 500.00'."""
    def convert(value):
        if pd.isna(value):
            return np.nan
        text = re.sub(r"(?i)usd|\$|,|\s", "", str(value))
        factor = 1
        if text[-1:].upper() == "M":
            factor, text = 1_000_000, text[:-1]
        elif text[-1:].upper() == "K":
            factor, text = 1_000, text[:-1]
        try:
            return float(text) * factor
        except ValueError:
            return np.nan
    return tidy_text(series).map(convert).astype(float)

# Each messy date style has its own separator, so every format can be read exactly.
DATE_PATTERNS = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),
    (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "%d/%m/%Y"),
    (re.compile(r"^\d{2}-\d{2}-\d{4}$"), "%m-%d-%Y"),
    (re.compile(r"^\d{4}/\d{2}/\d{2}$"), "%Y/%m/%d"),
    (re.compile(r"^\d{1,2} [A-Za-z]{3} \d{4}$"), "%d %b %Y"),
    (re.compile(r"^[A-Za-z]+ \d{1,2}, \d{4}$"), "%B %d, %Y"),
]

def parse_date(series):
    def one(value):
        if pd.isna(value):
            return pd.NaT
        text = str(value).strip()
        for pattern, fmt in DATE_PATTERNS:
            if pattern.match(text):
                return pd.to_datetime(text, format=fmt, errors="coerce")
        return pd.NaT
    return pd.to_datetime(tidy_text(series).map(one))

def fill_with_group_median(frame, column, group):
    group_median = frame.groupby(group)[column].transform("median")
    frame[column] = frame[column].fillna(group_median).fillna(frame[column].median())

COUNTRIES = {
    "United States": ["usa", "u.s.a", "us", "united states"],
    "United Kingdom": ["uk", "u.k.", "great britain", "united kingdom"],
    "UAE": ["uae", "u.a.e", "united arab emirates"],
    "Saudi Arabia": ["ksa", "saudi arabia"],
    "Nigeria": ["ng", "nigeria"],
    "South Africa": ["rsa", "s. africa", "south africa"],
    "India": ["in", "india"],
    "Australia": ["aus", "australia"],
    "Germany": ["deutschland", "germany"],
    "Brazil": ["brasil", "brazil"],
}
COUNTRY_ALIASES = {alias: official for official, aliases in COUNTRIES.items() for alias in aliases}

PROJECT_TYPES = ["Residential", "Commercial", "Infrastructure", "Industrial"]
CONTRACT_TYPES = ["Fixed Price", "Cost Plus", "Design-Build"]
VENDOR_TYPES = ["Subcontractor", "Supplier"]
TASK_TYPES = ["Excavation", "Foundation", "Structural Frame", "MEP Installation", "Finishing", "Roofing", "Inspection"]
TASK_STATUSES = ["Completed", "In Progress", "Cancelled"]
COST_CATEGORIES = ["Materials", "Labour", "Equipment", "Subcontractors", "Overheads"]
MATERIALS = ["Cement", "Steel Rebar", "Timber", "Glass Panels", "Electrical Fittings"]
UNIT_ALIASES = {"bags": "bags", "bag": "bags", "tonnes": "tonnes", "t": "tonnes", "mt": "tonnes",
                "tons": "tonnes", "m3": "m3", "cubic meters": "m3", "cum": "m3", "sheets": "sheets",
                "sht": "sheets", "pieces": "pieces", "pcs": "pieces", "nos": "pieces"}
INCIDENT_TYPES = ["Fall from Height", "Struck by Object", "Electrical Contact", "Equipment Collision", "Trench Collapse", "Slip or Trip"]
SHIFTS = ["Day", "Night"]
WEATHER = ["Clear", "Rain", "Heat", "Wind"]
SEVERITIES = ["Near Miss", "Minor", "Lost Time Injury", "Fatality"]

# ---------- 1. dim_projects ----------
projects = df_projects.copy()
log("dim_projects", "Exact duplicate rows removed", projects.duplicated().sum())
projects = projects.drop_duplicates().copy()

projects["project_id"] = tidy_text(projects["project_id"]).str.upper()
projects["project_name"] = tidy_text(projects["project_name"]).str.title()
projects["country"] = canonical(projects["country"], COUNTRIES, COUNTRY_ALIASES)
projects["project_type"] = canonical(projects["project_type"], PROJECT_TYPES)
projects["contract_type"] = canonical(projects["contract_type"], CONTRACT_TYPES)
projects["client_sector"] = tidy_text(projects["client_sector"]).str.title()
projects["budget_usd"] = parse_money(projects["budget_usd"])
projects["start_date"] = parse_date(projects["start_date"])

# '18 months' becomes 18 x 30 days. 'N/A' and negatives become missing.
duration_text = tidy_text(projects["planned_duration_days"]).str.lower()
months = pd.to_numeric(duration_text.str.extract(r"^(\d+)\s*months?$")[0], errors="coerce")
duration = pd.to_numeric(duration_text, errors="coerce").fillna(months * 30)
projects["planned_duration_days"] = duration.where(duration > 0)

projects["site_area_sqm"] = pd.to_numeric(projects["site_area_sqm"], errors="coerce")
bad_area = (projects["site_area_sqm"] <= 0) | (projects["site_area_sqm"] >= 1_000_000)
log("dim_projects", "Impossible site area set to missing", bad_area.sum())
projects.loc[bad_area, "site_area_sqm"] = np.nan

for column in ["planned_duration_days", "site_area_sqm", "num_floors"]:
    log("dim_projects", f"{column} missing, filled with median of project type", projects[column].isna().sum())
    fill_with_group_median(projects, column, "project_type")
    projects[column] = projects[column].round().astype(int)

df_projects_clean = projects.sort_values("project_id").reset_index(drop=True)
valid_project_ids = set(df_projects_clean["project_id"])

# ---------- 2. dim_vendors ----------
vendors = df_vendors.copy()
log("dim_vendors", "Exact duplicate rows removed", vendors.duplicated().sum())
vendors = vendors.drop_duplicates().copy()

vendors["vendor_id"] = tidy_text(vendors["vendor_id"]).str.upper()
vendors["vendor_name"] = tidy_text(vendors["vendor_name"]).str.title()
vendors["vendor_type"] = canonical(vendors["vendor_type"], VENDOR_TYPES)
vendors["specialty"] = tidy_text(vendors["specialty"]).str.title()
vendors["country"] = canonical(vendors["country"], COUNTRIES, COUNTRY_ALIASES)
vendors["prequalified"] = yes_no(vendors["prequalified"])

# '4/5' means 4 and 'five' means 5
rating_text = tidy_text(vendors["reliability_rating"]).str.lower()
rating_text = rating_text.str.replace(r"^(\d)\s*/\s*5$", r"\1", regex=True)
rating_text = rating_text.replace({"one": "1", "two": "2", "three": "3", "four": "4", "five": "5"})
vendors["reliability_rating"] = pd.to_numeric(rating_text, errors="coerce")
vendors["safety_rating"] = pd.to_numeric(vendors["safety_rating"], errors="coerce")

for column in ["reliability_rating", "safety_rating"]:
    log("dim_vendors", f"{column} missing, filled with median of vendor type", vendors[column].isna().sum())
    fill_with_group_median(vendors, column, "vendor_type")
    vendors[column] = vendors[column].round().astype(int)

df_vendors_clean = vendors.sort_values("vendor_id").reset_index(drop=True)
valid_vendor_ids = set(df_vendors_clean["vendor_id"])

# ---------- 3. fact_cost_ledger ----------
ledger = df_cost_ledger.copy()
log("fact_cost_ledger", "Exact duplicate rows removed", ledger.duplicated().sum())
ledger = ledger.drop_duplicates().copy()

ledger["ledger_id"] = tidy_text(ledger["ledger_id"]).str.upper()
ledger["project_id"] = tidy_text(ledger["project_id"]).str.upper()
orphans = ~ledger["project_id"].isin(valid_project_ids)
log("fact_cost_ledger", "Rows with unknown project removed", orphans.sum())
ledger = ledger[~orphans].copy()

# 2023-Q4, Q4 2023, 2023 Q4 and 2023/Q4 all become 2023-Q4
period = tidy_text(ledger["period"]).str.upper()
year = period.str.extract(r"(\d{4})")[0]
quarter = period.str.extract(r"Q([1-4])")[0]
ledger["period"] = year + "-Q" + quarter
ledger["cost_category"] = canonical(ledger["cost_category"], COST_CATEGORIES)
ledger["budgeted_usd"] = parse_money(ledger["budgeted_usd"])
ledger["actual_usd"] = parse_money(ledger["actual_usd"])
ledger["approved_change_orders_usd"] = parse_money(ledger["approved_change_orders_usd"])

bad_actual = ledger["actual_usd"].lt(0) | ledger["actual_usd"].isna() | ledger["budgeted_usd"].le(0) | ledger["budgeted_usd"].isna()
log("fact_cost_ledger", "Rows removed: negative or missing budget/actual cost", bad_actual.sum())
ledger = ledger[~bad_actual].copy()

log("fact_cost_ledger", "Missing change orders set to 0", ledger["approved_change_orders_usd"].isna().sum())
ledger["approved_change_orders_usd"] = ledger["approved_change_orders_usd"].fillna(0.0)

df_cost_ledger_clean = ledger.sort_values("ledger_id").reset_index(drop=True)

# ---------- 4. fact_tasks ----------
tasks = df_tasks.copy()
log("fact_tasks", "Exact duplicate rows removed", tasks.duplicated().sum())
tasks = tasks.drop_duplicates().copy()

tasks["task_id"] = tidy_text(tasks["task_id"]).str.upper()
tasks["project_id"] = tidy_text(tasks["project_id"]).str.upper()
tasks["vendor_id"] = tidy_text(tasks["vendor_id"]).str.upper()
orphans = ~tasks["project_id"].isin(valid_project_ids)
log("fact_tasks", "Rows with unknown project removed", orphans.sum())
tasks = tasks[~orphans].copy()

tasks["task_type"] = canonical(tasks["task_type"], TASK_TYPES)
tasks["task_status"] = canonical(tasks["task_status"], TASK_STATUSES, {"Complete": "Completed"})
tasks["rework_required"] = yes_no(tasks["rework_required"])
for column in ["planned_start", "planned_end", "actual_start", "actual_end"]:
    tasks[column] = parse_date(tasks[column])

swapped = tasks["actual_end"] < tasks["actual_start"]
log("fact_tasks", "Swapped actual start/end dates put back in order", swapped.sum())
tasks.loc[swapped, ["actual_start", "actual_end"]] = tasks.loc[swapped, ["actual_end", "actual_start"]].values

tasks["crew_size"] = pd.to_numeric(tasks["crew_size"], errors="coerce")
bad_crew = tasks["crew_size"].le(0) | tasks["crew_size"].gt(100)
tasks.loc[bad_crew, "crew_size"] = np.nan
log("fact_tasks", "crew_size impossible or missing, filled with median of task type", tasks["crew_size"].isna().sum())
fill_with_group_median(tasks, "crew_size", "task_type")
tasks["crew_size"] = tasks["crew_size"].round().astype(int)

df_tasks_clean = tasks.sort_values("task_id").reset_index(drop=True)

# ---------- 5. fact_material_orders ----------
orders = df_material_orders.copy()
log("fact_material_orders", "Exact duplicate rows removed", orders.duplicated().sum())
orders = orders.drop_duplicates().copy()

orders["order_id"] = tidy_text(orders["order_id"]).str.upper()
orders["project_id"] = tidy_text(orders["project_id"]).str.upper()
orders["vendor_id"] = tidy_text(orders["vendor_id"]).str.upper()
unknown_vendor = orders["vendor_id"].notna() & ~orders["vendor_id"].isin(valid_vendor_ids)
log("fact_material_orders", "Unknown vendor codes set to missing (order kept)", unknown_vendor.sum())
orders.loc[unknown_vendor, "vendor_id"] = pd.NA

orders["material"] = canonical(orders["material"], MATERIALS)
orders["unit"] = canonical(orders["unit"], [], UNIT_ALIASES)
orders["quality_passed"] = yes_no(orders["quality_passed"])
orders["unit_price_usd"] = parse_money(orders["unit_price_usd"])   # price spikes kept on purpose (objective 5)
orders["total_cost_usd"] = parse_money(orders["total_cost_usd"])
for column in ["order_date", "promised_date", "delivered_date"]:
    orders[column] = parse_date(orders[column])

orders["quantity"] = parse_number(orders["quantity"])
negative_qty = orders["quantity"] < 0
log("fact_material_orders", "Negative quantity sign corrected", negative_qty.sum())
orders["quantity"] = orders["quantity"].abs()

# a missing quantity can be rebuilt as total cost / unit price; otherwise use the material median
rebuilt = (orders["total_cost_usd"] / orders["unit_price_usd"]).round()
can_rebuild = orders["quantity"].isna() & rebuilt.gt(0)
log("fact_material_orders", "Missing quantity rebuilt from total cost / unit price", can_rebuild.sum())
orders.loc[can_rebuild, "quantity"] = rebuilt[can_rebuild]
log("fact_material_orders", "Missing quantity filled with material median", orders["quantity"].isna().sum())
fill_with_group_median(orders, "quantity", "material")

log("fact_material_orders", "Delivered date missing (not delivered or unreadable)", orders["delivered_date"].isna().sum())
df_material_orders_clean = orders.sort_values("order_id").reset_index(drop=True)

# ---------- 6. fact_safety_incidents ----------
incidents = df_safety_incidents.copy()
log("fact_safety_incidents", "Exact duplicate rows removed", incidents.duplicated().sum())
incidents = incidents.drop_duplicates().copy()

incidents["incident_id"] = tidy_text(incidents["incident_id"]).str.upper()
incidents["project_id"] = tidy_text(incidents["project_id"]).str.upper()
incidents["vendor_id"] = tidy_text(incidents["vendor_id"]).str.upper()
incidents["incident_date"] = parse_date(incidents["incident_date"])
incidents["incident_type"] = canonical(incidents["incident_type"], INCIDENT_TYPES)
incidents["shift"] = canonical(incidents["shift"], SHIFTS)
incidents["weather_condition"] = canonical(incidents["weather_condition"], WEATHER)
incidents["severity"] = canonical(incidents["severity"], SEVERITIES, {"LTI": "Lost Time Injury"})
incidents["ppe_compliant"] = yes_no(incidents["ppe_compliant"])
incidents["work_at_height"] = yes_no(incidents["work_at_height"])

incidents["worker_experience_years"] = pd.to_numeric(incidents["worker_experience_years"], errors="coerce")
bad_experience = incidents["worker_experience_years"].lt(0) | incidents["worker_experience_years"].gt(60)
incidents.loc[bad_experience, "worker_experience_years"] = np.nan
log("fact_safety_incidents", "worker_experience_years impossible or missing, filled with median", incidents["worker_experience_years"].isna().sum())
incidents["worker_experience_years"] = incidents["worker_experience_years"].fillna(incidents["worker_experience_years"].median()).round().astype(int)

df_safety_incidents_clean = incidents.sort_values("incident_id").reset_index(drop=True)

# ---------- 7. Save clean tables ----------
clean_tables = {
    "dim_projects": df_projects_clean,
    "dim_vendors": df_vendors_clean,
    "fact_cost_ledger": df_cost_ledger_clean,
    "fact_tasks": df_tasks_clean,
    "fact_material_orders": df_material_orders_clean,
    "fact_safety_incidents": df_safety_incidents_clean,
}
for name, frame in clean_tables.items():
    frame.to_csv(CLEAN_DIR / f"{name}.csv", index=False)

# ---------- 8. Verification ----------
raw_tables = {"dim_projects": df_projects, "dim_vendors": df_vendors, "fact_cost_ledger": df_cost_ledger,
              "fact_tasks": df_tasks, "fact_material_orders": df_material_orders,
              "fact_safety_incidents": df_safety_incidents}
summary = pd.DataFrame([
    {"table": name, "rows_before": len(raw_tables[name]), "rows_after": len(frame),
     "duplicate_keys": int(frame.iloc[:, 0].duplicated().sum()),
     "missing_cells": int(frame.isna().sum().sum())}
    for name, frame in clean_tables.items()
]).set_index("table")

remaining_missing = pd.DataFrame([
    {"table": name, "column": column, "missing": int(count)}
    for name, frame in clean_tables.items()
    for column, count in frame.isna().sum().items() if count
])

checks = pd.DataFrame([
    ("Tasks: actual end before actual start", int((df_tasks_clean["actual_end"] < df_tasks_clean["actual_start"]).sum())),
    ("Tasks: planned end before planned start", int((df_tasks_clean["planned_end"] < df_tasks_clean["planned_start"]).sum())),
    ("Orders: delivered before ordered", int((df_material_orders_clean["delivered_date"] < df_material_orders_clean["order_date"]).sum())),
    ("Orders: promised before ordered", int((df_material_orders_clean["promised_date"] < df_material_orders_clean["order_date"]).sum())),
    ("Ledger: orphan project codes", int((~df_cost_ledger_clean["project_id"].isin(valid_project_ids)).sum())),
    ("Tasks: orphan project codes", int((~df_tasks_clean["project_id"].isin(valid_project_ids)).sum())),
    ("Orders: orphan project codes", int((~df_material_orders_clean["project_id"].isin(valid_project_ids)).sum())),
    ("Incidents: orphan project codes", int((~df_safety_incidents_clean["project_id"].isin(valid_project_ids)).sum())),
    ("Unmapped category labels (all tables)", int(
        df_projects_clean[["country", "project_type", "contract_type"]].isna().sum().sum()
        + df_vendors_clean[["vendor_type", "country", "prequalified"]].isna().sum().sum()
        + df_cost_ledger_clean[["period", "cost_category"]].isna().sum().sum()
        + df_tasks_clean[["task_type", "task_status", "rework_required"]].isna().sum().sum()
        + df_material_orders_clean[["material", "unit", "quality_passed"]].isna().sum().sum()
        + df_safety_incidents_clean[["incident_type", "shift", "weather_condition", "severity", "ppe_compliant", "work_at_height"]].isna().sum().sum()
    )),
], columns=["check", "rows"])

label_check = pd.DataFrame({
    "country (projects)": [df_projects_clean["country"].nunique()],
    "task_type": [df_tasks_clean["task_type"].nunique()],
    "cost_category": [df_cost_ledger_clean["cost_category"].nunique()],
    "material": [df_material_orders_clean["material"].nunique()],
    "incident_type": [df_safety_incidents_clean["incident_type"].nunique()],
    "severity": [df_safety_incidents_clean["severity"].nunique()],
}, index=["distinct labels"])

display(pd.DataFrame(cleaning_log))
display(summary)
display(remaining_missing)
display(checks)
display(label_check)
```

---

## 6. Feature Engineering

Raw columns become signals a model can use, and the four targets are created here. Every feature is logged with its meaning: 48 new features across the six tables.

| Table | New features | Examples |
|---|---:|---|
| `dim_projects` | 10 | `budget_per_sqm`, `high_rise` (30+ floors), `is_public`, `manager_experience_band` |
| `dim_vendors` | 3 | `vendor_score`, `vendor_experience_band`, `is_prequalified` |
| `fact_cost_ledger` | 8 | `overrun_pct`, `overrun_flag` (above 15%), `budget_share_of_project`, `log_budgeted_usd` |
| `fact_tasks` | 9 | `delay_days`, `delay_flag` (above 5 days), `rainy_season`, `crew_band` |
| `fact_material_orders` | 12 | `lead_time_days`, `late_delivery`, `international_supplier`, `price_ratio_material`, `price_ratio_peer` |
| `fact_safety_incidents` | 6 | `serious_flag`, `no_ppe_at_height`, `experience_band` |

Early signs from the new features: international orders are late 48% of the time against 29% for local ones, tasks starting in the rainy season are delayed 75% of the time against 36%, and incidents at height without PPE are serious 59% of the time against 23%.

The rainy-season calendar (Nigeria and India June to September, Brazil December to March, UK October to January) is a domain assumption, not something measured from the data.

### Code: feature engineering

```python
# ============================================================
# FEATURE ENGINEERING PHASE
# ============================================================

# %%
from pathlib import Path

FEATURE_DIR = Path("construction_features")
FEATURE_DIR.mkdir(exist_ok=True)

feature_log = []

def add_log(table, feature, meaning):
    feature_log.append({"table": table, "feature": feature, "meaning": meaning})

# Rainy-season calendar used for the 'rainy season' features (domain assumption, by country).
# Countries not listed here are treated as having no distinct rainy season.
RAINY_MONTHS = {
    "Nigeria": [6, 7, 8, 9],
    "India": [6, 7, 8, 9],
    "Brazil": [12, 1, 2, 3],
    "United Kingdom": [10, 11, 12, 1],
}

def in_rainy_season(country, month):
    return pd.Series(
        [int(m in RAINY_MONTHS.get(c, [])) if pd.notna(m) and pd.notna(c) else np.nan
         for c, m in zip(country, month)],
        index=country.index,
    )

project_country = df_projects_clean.set_index("project_id")["country"]
vendor_country = df_vendors_clean.set_index("vendor_id")["country"]

# ---------- 1. dim_projects ----------
df_projects_feat = df_projects_clean.copy()
p = df_projects_feat
p["start_year"] = p["start_date"].dt.year
p["start_month"] = p["start_date"].dt.month
p["planned_end_date"] = p["start_date"] + pd.to_timedelta(p["planned_duration_days"], unit="D")
p["budget_per_sqm"] = (p["budget_usd"] / p["site_area_sqm"]).round(2)
p["budget_per_day"] = (p["budget_usd"] / p["planned_duration_days"]).round(2)
p["budget_per_floor"] = (p["budget_usd"] / p["num_floors"]).round(2)
p["size_band"] = pd.qcut(p["site_area_sqm"], 3, labels=["Small", "Medium", "Large"]).astype("string")
p["high_rise"] = (p["num_floors"] >= 30).astype(int)
p["is_public"] = (p["client_sector"] == "Public").astype(int)
p["manager_experience_band"] = pd.cut(
    p["project_manager_experience_years"], [-1, 9, 19, 100], labels=["Junior", "Mid", "Senior"]
).astype("string")
for feature, meaning in [
    ("start_year", "Year the project started"),
    ("start_month", "Month the project started"),
    ("planned_end_date", "Start date + planned duration"),
    ("budget_per_sqm", "Budget per square metre of site"),
    ("budget_per_day", "Budget per planned day"),
    ("budget_per_floor", "Budget per floor"),
    ("size_band", "Small / Medium / Large site (three equal groups)"),
    ("high_rise", "1 if 30 or more floors"),
    ("is_public", "1 if public sector client"),
    ("manager_experience_band", "Junior (<10), Mid (10-19), Senior (20+ years)"),
]:
    add_log("dim_projects", feature, meaning)

# ---------- 2. dim_vendors ----------
df_vendors_feat = df_vendors_clean.copy()
v = df_vendors_feat
v["vendor_score"] = ((v["reliability_rating"] + v["safety_rating"]) / 2).round(1)
v["vendor_experience_band"] = pd.cut(
    v["years_active"], [-1, 9, 24, 100], labels=["New", "Established", "Veteran"]
).astype("string")
v["is_prequalified"] = (v["prequalified"] == "Yes").astype(int)
for feature, meaning in [
    ("vendor_score", "Average of reliability and safety rating"),
    ("vendor_experience_band", "New (<10), Established (10-24), Veteran (25+ years active)"),
    ("is_prequalified", "1 if the vendor is prequalified"),
]:
    add_log("dim_vendors", feature, meaning)

# ---------- 3. fact_cost_ledger (objective 1) ----------
df_cost_ledger_feat = df_cost_ledger_clean.copy()
l = df_cost_ledger_feat
l["period_year"] = l["period"].str[:4].astype(int)
l["period_quarter"] = l["period"].str[-1].astype(int)
l["overrun_usd"] = (l["actual_usd"] - l["budgeted_usd"]).round(2)
l["overrun_pct"] = (l["actual_usd"] / l["budgeted_usd"] - 1).round(4)
l["overrun_flag"] = (l["overrun_pct"] > 0.15).astype(int)
l["change_order_pct"] = (l["approved_change_orders_usd"] / l["budgeted_usd"]).round(4)
l["budget_share_of_project"] = (
    l["budgeted_usd"] / l.groupby("project_id")["budgeted_usd"].transform("sum")
).round(4)
l["log_budgeted_usd"] = np.log1p(l["budgeted_usd"]).round(4)
for feature, meaning in [
    ("period_year", "Year of the cost period"),
    ("period_quarter", "Quarter (1-4) of the cost period"),
    ("overrun_usd", "Actual minus budget in USD"),
    ("overrun_pct", "Actual / budget - 1 (objective 1 regression target)"),
    ("overrun_flag", "1 if overrun is above 15% (objective 1 classification target)"),
    ("change_order_pct", "Approved change orders as a share of budget"),
    ("budget_share_of_project", "This line's share of the project's ledger budget"),
    ("log_budgeted_usd", "Log of the budgeted amount"),
]:
    add_log("fact_cost_ledger", feature, meaning)

# ---------- 4. fact_tasks (objective 2) ----------
df_tasks_feat = df_tasks_clean.copy()
t = df_tasks_feat
t["task_country"] = t["project_id"].map(project_country)
t["planned_duration_days"] = (t["planned_end"] - t["planned_start"]).dt.days
t["planned_start_year"] = t["planned_start"].dt.year
t["planned_start_month"] = t["planned_start"].dt.month
t["rainy_season"] = in_rainy_season(t["task_country"], t["planned_start"].dt.month)
t["crew_band"] = pd.cut(t["crew_size"], [0, 19, 39, 100], labels=["Small", "Medium", "Large"]).astype("string")
t["vendor_known"] = t["vendor_id"].notna().astype(int)

finished = t["task_status"].eq("Completed") & t["actual_end"].notna()
t["delay_days"] = (t["actual_end"] - t["planned_end"]).dt.days.where(finished)
t["delay_flag"] = (t["delay_days"] > 5).astype(float).where(finished)
t["actual_duration_days"] = (t["actual_end"] - t["actual_start"]).dt.days.where(finished)
t = t.drop(columns="task_country")
df_tasks_feat = t
for feature, meaning in [
    ("planned_duration_days", "Planned end minus planned start"),
    ("planned_start_year", "Year the task was planned to start"),
    ("planned_start_month", "Month the task was planned to start"),
    ("rainy_season", "1 if the planned start month is in the project country's rainy season"),
    ("crew_band", "Small (<20), Medium (20-39), Large (40+) crew"),
    ("vendor_known", "1 if the vendor on the task is recorded"),
    ("delay_days", "Actual end minus planned end, completed tasks only (objective 2 regression target)"),
    ("delay_flag", "1 if delay is more than 5 days, completed tasks only (objective 2 classification target)"),
    ("actual_duration_days", "Actual end minus actual start, completed tasks only"),
]:
    add_log("fact_tasks", feature, meaning)

# ---------- 5. fact_material_orders (objectives 4 and 5) ----------
df_material_orders_feat = df_material_orders_clean.copy()
o = df_material_orders_feat
o["project_country"] = o["project_id"].map(project_country)
o["supplier_country"] = o["vendor_id"].map(vendor_country)
o["order_year"] = o["order_date"].dt.year
o["order_month"] = o["order_date"].dt.month
o["lead_time_days"] = (o["promised_date"] - o["order_date"]).dt.days
o["actual_lead_days"] = (o["delivered_date"] - o["order_date"]).dt.days
o["delivery_delay_days"] = (o["delivered_date"] - o["promised_date"]).dt.days
o["late_delivery"] = (o["delivered_date"] > o["promised_date"]).astype(float).where(
    o["delivered_date"].notna() & o["promised_date"].notna()
)
o["international_supplier"] = (o["supplier_country"] != o["project_country"]).astype(float).where(
    o["supplier_country"].notna() & o["project_country"].notna()
)
o["rainy_season"] = in_rainy_season(o["project_country"], o["promised_date"].dt.month)
o["log_quantity"] = np.log1p(o["quantity"]).round(4)

# price features for objective 5: compare each price with its peers
material_median = o.groupby("material")["unit_price_usd"].transform("median")
group_median = o.groupby(["material", "project_country"])["unit_price_usd"].transform("median")
group_count = o.groupby(["material", "project_country"])["unit_price_usd"].transform("count")
peer_median = group_median.where(group_count >= 20, material_median)
o["price_ratio_material"] = (o["unit_price_usd"] / material_median).round(4)
o["price_ratio_peer"] = (o["unit_price_usd"] / peer_median).round(4)
o["total_vs_calculated_pct"] = (
    o["total_cost_usd"] / (o["quantity"] * o["unit_price_usd"]) - 1
).round(4)
o = o.drop(columns=["project_country", "supplier_country"])
df_material_orders_feat = o
for feature, meaning in [
    ("order_year", "Year the order was placed"),
    ("order_month", "Month the order was placed"),
    ("lead_time_days", "Promised date minus order date"),
    ("actual_lead_days", "Delivered date minus order date"),
    ("delivery_delay_days", "Delivered date minus promised date"),
    ("late_delivery", "1 if delivered after the promised date (objective 4 target)"),
    ("international_supplier", "1 if supplier country differs from project country"),
    ("rainy_season", "1 if the promised month is in the project country's rainy season"),
    ("log_quantity", "Log of the quantity ordered"),
    ("price_ratio_material", "Unit price / median price of the same material"),
    ("price_ratio_peer", "Unit price / median price of the same material in the same project country"),
    ("total_vs_calculated_pct", "Invoice total / (quantity x unit price) - 1"),
]:
    add_log("fact_material_orders", feature, meaning)

# ---------- 6. fact_safety_incidents (objective 3) ----------
df_safety_incidents_feat = df_safety_incidents_clean.copy()
s = df_safety_incidents_feat
s["serious_flag"] = s["severity"].isin(["Lost Time Injury", "Fatality"]).astype(int)
s["incident_year"] = s["incident_date"].dt.year
s["incident_month"] = s["incident_date"].dt.month
s["experience_band"] = pd.cut(
    s["worker_experience_years"], [-1, 2, 9, 19, 100], labels=["Novice", "Junior", "Experienced", "Veteran"]
).astype("string")
s["no_ppe_at_height"] = ((s["ppe_compliant"] == "No") & (s["work_at_height"] == "Yes")).astype(int)
s["vendor_known"] = s["vendor_id"].notna().astype(int)
for feature, meaning in [
    ("serious_flag", "1 if Lost Time Injury or Fatality (objective 3 target)"),
    ("incident_year", "Year of the incident"),
    ("incident_month", "Month of the incident"),
    ("experience_band", "Novice (<=2), Junior (3-9), Experienced (10-19), Veteran (20+ years)"),
    ("no_ppe_at_height", "1 if working at height without PPE"),
    ("vendor_known", "1 if the vendor on the incident is recorded"),
]:
    add_log("fact_safety_incidents", feature, meaning)

# ---------- 7. Save feature tables ----------
feature_tables = {
    "dim_projects": df_projects_feat,
    "dim_vendors": df_vendors_feat,
    "fact_cost_ledger": df_cost_ledger_feat,
    "fact_tasks": df_tasks_feat,
    "fact_material_orders": df_material_orders_feat,
    "fact_safety_incidents": df_safety_incidents_feat,
}
for name, frame in feature_tables.items():
    frame.to_csv(FEATURE_DIR / f"{name}_features.csv", index=False)

# ---------- 8. Verification ----------
targets = pd.DataFrame([
    ("1 Cost overrun", "overrun_flag", len(df_cost_ledger_feat), df_cost_ledger_feat["overrun_flag"].mean()),
    ("2 Schedule delay", "delay_flag", int(df_tasks_feat["delay_flag"].notna().sum()), df_tasks_feat["delay_flag"].mean()),
    ("3 Serious incident", "serious_flag", len(df_safety_incidents_feat), df_safety_incidents_feat["serious_flag"].mean()),
    ("4 Late delivery", "late_delivery", int(df_material_orders_feat["late_delivery"].notna().sum()), df_material_orders_feat["late_delivery"].mean()),
], columns=["objective", "target", "usable_rows", "positive_rate"])
targets["positive_rate"] = (100 * targets["positive_rate"]).round(1)

summary = pd.DataFrame([
    {"table": name, "rows": len(frame), "columns": frame.shape[1],
     "new_features": int((pd.DataFrame(feature_log)["table"] == name).sum())}
    for name, frame in feature_tables.items()
]).set_index("table")

display(pd.DataFrame(feature_log))
display(summary)
display(targets)
display(df_material_orders_feat.groupby("international_supplier")["late_delivery"].mean().rename("late_rate").to_frame())
display(df_tasks_feat.groupby("rainy_season")["delay_flag"].mean().rename("delay_rate").to_frame())
display(df_safety_incidents_feat.groupby("no_ppe_at_height")["serious_flag"].mean().rename("serious_rate").to_frame())
```

---

## 7. Create Relationship Table

Every fact table links to the dimension tables (projects, vendors, dates) through a star schema, and a calendar table `dim_date` (1,938 days) is added.

| Check | Result |
|---|---|
| Orphan keys on every relationship | 0 |
| Match rate between child and parent keys | 100% |
| Many-to-one joins keep every row | Yes (2,413 ledger, 3,985 tasks, 3,000 orders, 750 incidents before and after) |

### Charts

<p align="center">
  <img width="2055" height="1263" alt="star_schema" src="https://github.com/user-attachments/assets/6bc0ae05-e2a9-4e43-9401-f34154edd529" />
</p>
<p align="center"><em>The star schema: four fact tables around the project, vendor and date dimensions</em></p>

### Code: relationship table (star schema)

```python
# ============================================================
# RELATIONSHIP TABLE STAR SCHEMA
# ============================================================

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

STAR_DIR = Path("construction_star")
STAR_DIR.mkdir(exist_ok=True)

dim_tables = {"dim_projects": df_projects_feat, "dim_vendors": df_vendors_feat}
fact_tables = {
    "fact_cost_ledger": df_cost_ledger_feat,
    "fact_tasks": df_tasks_feat,
    "fact_material_orders": df_material_orders_feat,
    "fact_safety_incidents": df_safety_incidents_feat,
}

# ---------- 1. Date keys on every fact table ----------
def date_key(series):
    return (series.dt.year * 10000 + series.dt.month * 100 + series.dt.day).astype("Int64")

ledger_period_start = pd.to_datetime({
    "year": df_cost_ledger_feat["period_year"],
    "month": (df_cost_ledger_feat["period_quarter"] - 1) * 3 + 1,
    "day": 1,
})

star_dim_projects = df_projects_feat.copy()
star_dim_projects["start_date_key"] = date_key(star_dim_projects["start_date"])

star_facts = {name: frame.copy() for name, frame in fact_tables.items()}
star_facts["fact_cost_ledger"]["date_key"] = date_key(ledger_period_start)
star_facts["fact_tasks"]["date_key"] = date_key(star_facts["fact_tasks"]["planned_start"])
star_facts["fact_material_orders"]["date_key"] = date_key(star_facts["fact_material_orders"]["order_date"])
star_facts["fact_safety_incidents"]["date_key"] = date_key(star_facts["fact_safety_incidents"]["incident_date"])

# ---------- 2. dim_date ----------
all_dates = pd.concat([
    ledger_period_start,
    df_projects_feat["start_date"], df_projects_feat["planned_end_date"],
    df_tasks_feat[["planned_start", "planned_end", "actual_start", "actual_end"]].stack(),
    df_material_orders_feat[["order_date", "promised_date", "delivered_date"]].stack(),
    df_safety_incidents_feat["incident_date"],
]).dropna()

calendar = pd.date_range(all_dates.min().normalize(), all_dates.max().normalize(), freq="D")
dim_date = pd.DataFrame({"date": calendar})
dim_date["date_key"] = date_key(dim_date["date"])
dim_date["year"] = dim_date["date"].dt.year
dim_date["quarter"] = dim_date["date"].dt.quarter
dim_date["month"] = dim_date["date"].dt.month
dim_date["month_name"] = dim_date["date"].dt.month_name()
dim_date["year_month"] = dim_date["date"].dt.strftime("%Y-%m")
dim_date["week_of_year"] = dim_date["date"].dt.isocalendar().week.astype(int)
dim_date["day_of_week"] = dim_date["date"].dt.day_name()
dim_date["is_weekend"] = dim_date["date"].dt.dayofweek.ge(5).astype(int)

# ---------- 3. The relationship map ----------
relationship_map = [
    ("fact_cost_ledger", "project_id", "dim_projects", "project_id"),
    ("fact_tasks", "project_id", "dim_projects", "project_id"),
    ("fact_material_orders", "project_id", "dim_projects", "project_id"),
    ("fact_safety_incidents", "project_id", "dim_projects", "project_id"),
    ("fact_tasks", "vendor_id", "dim_vendors", "vendor_id"),
    ("fact_material_orders", "vendor_id", "dim_vendors", "vendor_id"),
    ("fact_safety_incidents", "vendor_id", "dim_vendors", "vendor_id"),
    ("fact_cost_ledger", "date_key", "dim_date", "date_key"),
    ("fact_tasks", "date_key", "dim_date", "date_key"),
    ("fact_material_orders", "date_key", "dim_date", "date_key"),
    ("fact_safety_incidents", "date_key", "dim_date", "date_key"),
    ("dim_projects", "start_date_key", "dim_date", "date_key"),
]

star_tables = {"dim_projects": star_dim_projects, "dim_vendors": df_vendors_feat,
               "dim_date": dim_date, **star_facts}

relationship_rows = []
for child, child_key, parent, parent_key in relationship_map:
    child_keys = star_tables[child][child_key]
    parent_keys = star_tables[parent][parent_key]
    present = child_keys.dropna()
    orphan = ~present.isin(set(parent_keys.dropna()))
    relationship_rows.append({
        "child_table": child, "child_key": child_key,
        "parent_table": parent, "parent_key": parent_key,
        "kind": "many-to-one",
        "child_rows": len(child_keys),
        "missing_key": int(child_keys.isna().sum()),
        "orphan_keys": int(orphan.sum()),
        "match_pct": round(100 * (1 - orphan.mean()), 2),
        "parent_key_unique": bool(parent_keys.is_unique),
    })
relationship_table = pd.DataFrame(relationship_rows)

# ---------- 4. Join test: a many-to-one join must never add or lose rows ----------
join_rows = []
for name, frame in star_facts.items():
    joined = frame.merge(
        df_projects_feat[["project_id", "country", "project_type", "contract_type"]],
        on="project_id", how="left")
    if "vendor_id" in frame.columns:
        joined = joined.merge(
            df_vendors_feat[["vendor_id", "vendor_type", "reliability_rating", "safety_rating"]],
            on="vendor_id", how="left")
    joined = joined.merge(dim_date[["date_key", "year", "quarter"]], on="date_key", how="left")
    join_rows.append({
        "fact_table": name, "rows_before": len(frame), "rows_after": len(joined),
        "rows_preserved": len(frame) == len(joined),
        "project_attributes_found": int(joined["country"].notna().sum()),
    })
join_test = pd.DataFrame(join_rows).set_index("fact_table")
assert join_test["rows_preserved"].all(), "A join changed the row count - check the keys."

# ---------- 5. Save the star schema ----------
for name, frame in star_tables.items():
    frame.to_csv(STAR_DIR / f"{name}.csv", index=False)
relationship_table.to_csv(STAR_DIR / "relationship_table.csv", index=False)

# ---------- 6. Star schema diagram ----------
positions = {
    "dim_projects": (2.6, 6.0), "dim_vendors": (8.4, 6.0), "dim_date": (5.5, 0.4),
    "fact_cost_ledger": (1.4, 3.2), "fact_tasks": (4.0, 3.2),
    "fact_material_orders": (7.0, 3.2), "fact_safety_incidents": (9.8, 3.2),
}
box_text = {
    "dim_projects": "dim_projects\nproject_id (PK)\nbudget, country, type",
    "dim_vendors": "dim_vendors\nvendor_id (PK)\nratings, country",
    "dim_date": "dim_date\ndate_key (PK)\nyear, quarter, month",
    "fact_cost_ledger": "fact_cost_ledger\nledger_id\nobjective 1",
    "fact_tasks": "fact_tasks\ntask_id\nobjective 2",
    "fact_material_orders": "fact_material_orders\norder_id\nobjectives 4 and 5",
    "fact_safety_incidents": "fact_safety_incidents\nincident_id\nobjective 3",
}
fig, ax = plt.subplots(figsize=(13, 7.5))
ax.set_xlim(-1.6, 11.8)
ax.set_ylim(-0.6, 7.2)
ax.axis("off")
for child, _, parent, _ in relationship_map:
    (x1, y1), (x2, y2) = positions[child], positions[parent]
    if child == "dim_projects" and parent == "dim_date":
        # dashed route down the far left edge so it does not cross the fact tables
        ax.plot([x1 - 1.15, -1.0, -1.0, x2 - 1.15], [y1, y1, y2, y2],
                color="#9aa5b1", linewidth=1.3, linestyle="--", zorder=1)
    else:
        ax.plot([x1, x2], [y1, y2], color="#9aa5b1", linewidth=1.3, zorder=1)
for name, (x, y) in positions.items():
    is_dim = name.startswith("dim_")
    box = FancyBboxPatch((x - 1.15, y - 0.55), 2.3, 1.1, boxstyle="round,pad=0.04",
                         facecolor="#dbeafe" if is_dim else "#fde7c8",
                         edgecolor="#1d4ed8" if is_dim else "#b45309", linewidth=1.6, zorder=2)
    ax.add_patch(box)
    ax.text(x, y, box_text[name], ha="center", va="center", fontsize=9, zorder=3)
ax.set_title("Construction star schema: four fact tables around two dimensions and a date table",
             fontsize=13, pad=12)
fig.savefig(STAR_DIR / "star_schema.png", dpi=200, bbox_inches="tight")
plt.show()

display(relationship_table)
display(join_test)
display(dim_date.head(3))
print(f"dim_date: {len(dim_date):,} days, {dim_date['date'].min().date()} to {dim_date['date'].max().date()}")
```

---

## 8. Exploratory Data Analysis

Nine charts, each answering one question before any model is built.

| Objective | Question | What the data says |
|---|---|---|
| All | How common are the problems? | 47% of cost lines overran by more than 15%, 41% of tasks finished late, 25% of incidents were serious, 46% of orders arrived late |
| 1 | What drives overruns? | Cost Plus 66% vs Fixed Price 36%; Materials 69% vs Overheads 21%; Nigeria 69% vs United States 27% |
| 2 | What drives delays? | Crews of 18 or fewer 60% vs 34% for the rest; rainy-season tasks 75% vs 36% |
| 3 | When are incidents serious? | Height work without PPE 59% vs 23%; falls from height 52% vs slips and trips 4%; least experienced 41% vs most experienced 9%; night shift 38% vs day 21% |
| 4 | When are orders late? | International orders 48% vs local 29% |
| 5 | What is a normal price? | Prices cluster around the material norm, with a small tail far above and below it. The answer key was not used here. |

### Charts

<p align="center">
  <img width="1530" height="501" alt="eda_01_portfolio_overview" src="https://github.com/user-attachments/assets/bf745d0a-dc5b-4175-94c8-5e465628f122" />
</p>
<p align="center"><em>How common are overruns, delays, serious incidents and late deliveries?</em></p>

<p align="center">
  <img width="1529" height="501" alt="eda_02_cost_overruns_by_category" src="https://github.com/user-attachments/assets/b886d0c3-cada-48e4-b6ec-0d74bb6e8a66" />
</p>
<p align="center"><em>Q1.1 How big are the overruns, and in which cost categories?</em></p>

<p align="center">
  <img width="1749" height="522" alt="eda_03_cost_overruns_by_project" src="https://github.com/user-attachments/assets/6df042a2-1cb0-48fc-b166-a60e64ddd263" />
</p>
<p align="center"><em>Q1.2 Overruns by country, project type and contract type</em></p>

<p align="center">
  <img width="1530" height="458" alt="eda_04_cost_overruns_over_time" src="https://github.com/user-attachments/assets/96cfb9c4-2823-41eb-80b0-59c0b91c6e9c" />
</p>
<p align="center"><em>Q1.3 Are overruns getting better or worse over time?</em></p>

<p align="center">
  <img width="1530" height="921" alt="eda_05_task_delays_drivers" src="https://github.com/user-attachments/assets/1ab285d8-c3ca-4836-ad82-6b308c8682d5" />
</p>
<p align="center"><em>Q2.1 What drives schedule delays?</em></p>

<p align="center">
  <img width="1530" height="501" alt="eda_06_task_delay_size_and_timing" src="https://github.com/user-attachments/assets/c2af4856-b488-4cb8-997c-b713887d97c8" />
</p>
<p align="center"><em>Q2.2 How late are late tasks, and when do they start?</em></p>

<p align="center">
  <img width="1530" height="921" alt="eda_07_safety_serious_incidents" src="https://github.com/user-attachments/assets/dc70cf7a-7bf5-4ebe-8329-7184417d0c00" />
</p>
<p align="center"><em>Q3 Where do serious incidents concentrate?</em></p>

<p align="center">
  <img width="1530" height="921" alt="eda_08_late_deliveries" src="https://github.com/user-attachments/assets/85b08516-266b-425a-beaf-c13feeae2d35" />
</p>
<p align="center"><em>Q4 What makes material orders late?</em></p>

<p align="center">
  <img width="1530" height="501" alt="eda_09_price_anomalies" src="https://github.com/user-attachments/assets/d008a63c-c031-43aa-a9bb-d67897e37180" />
</p>
<p align="center"><em>Q5 What does a normal price look like, and what does an unusual one look like?</em></p>

### Code: exploratory data analysis

```python
# ============================================================
# EXPLORATORY DATA ANALYSIS PHASE
# ============================================================

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

EDA_DIR = Path("construction_eda")
EDA_DIR.mkdir(exist_ok=True)
saved_charts = []

# ---------- chart style: one blue for the data, one orange for the thing to look at ----------
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, SOFT, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 110, "axes.edgecolor": GRID, "axes.labelcolor": SOFT,
    "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "xtick.color": SOFT, "ytick.color": SOFT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "font.size": 10,
})

# ---------- analysis tables: facts joined to their dimensions ----------
project_cols = ["project_id", "country", "project_type", "contract_type", "client_sector", "size_band", "high_rise"]
vendor_cols = ["vendor_id", "vendor_type", "reliability_rating", "safety_rating"]

ledger_eda = df_cost_ledger_feat.merge(df_projects_feat[project_cols], on="project_id", how="left")
tasks_eda = (df_tasks_feat.merge(df_projects_feat[project_cols], on="project_id", how="left")
             .merge(df_vendors_feat[vendor_cols], on="vendor_id", how="left"))
tasks_eda = tasks_eda[tasks_eda["delay_flag"].notna()].copy()
orders_eda = (df_material_orders_feat.merge(df_projects_feat[project_cols], on="project_id", how="left")
              .merge(df_vendors_feat[vendor_cols], on="vendor_id", how="left"))
orders_eda = orders_eda[orders_eda["late_delivery"].notna()].copy()
incidents_eda = (df_safety_incidents_feat.merge(df_projects_feat[project_cols], on="project_id", how="left")
                 .merge(df_vendors_feat[vendor_cols], on="vendor_id", how="left"))

tasks_eda["rainy_label"] = tasks_eda["rainy_season"].map({0.0: "Not rainy season", 1.0: "Rainy season"})
orders_eda["supplier_label"] = orders_eda["international_supplier"].map({0.0: "Domestic supplier", 1.0: "International supplier"})
orders_eda["lead_band"] = pd.cut(orders_eda["lead_time_days"], [0, 14, 28, 42, 400],
                                 labels=["Up to 2 weeks", "2-4 weeks", "4-6 weeks", "Over 6 weeks"]).astype("string")
incidents_eda["ppe_height_label"] = incidents_eda["no_ppe_at_height"].map({0: "Other incidents", 1: "At height, no PPE"})

# ---------- helpers ----------
def rate_table(frame, column, target, min_n=20):
    grouped = frame.dropna(subset=[column]).groupby(column)[target].agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= min_n]
    grouped["mean"] *= 100
    return grouped

def rate_bars(ax, frame, column, target, title, order=None):
    table = rate_table(frame, column, target)
    table = table.loc[order] if order is not None else table.sort_values("mean")
    overall = 100 * frame[target].mean()
    def nice(name):
        return str(int(name)) if isinstance(name, (float, np.floating)) and float(name).is_integer() else str(name)
    labels = [f"{nice(name)}  (n={int(n):,})" for name, n in zip(table.index, table["count"])]
    bars = ax.barh(labels, table["mean"], color=BLUE, height=0.6)
    ax.axvline(overall, color=ORANGE, linewidth=1.6, linestyle="--")
    ax.set_xlabel(f"Dashed line = overall average ({overall:.0f}%)", color=ORANGE)
    for bar, value in zip(bars, table["mean"]):
        ax.text(value + 0.6, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%", va="center", fontsize=9, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
    ax.set_xlim(0, max(table["mean"].max() * 1.18, overall * 1.3))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.grid(axis="y", visible=False)
    ax.set_title(title)

def finish(fig, name, headline):
    fig.suptitle(headline, x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(EDA_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    saved_charts.append(f"{name}.png")
    plt.show()

# ---------- Chart 1: portfolio overview ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
by_country = df_projects_feat["country"].value_counts().sort_values()
bars = axes[0].barh(by_country.index, by_country.values, color=BLUE, height=0.6)
for bar, value in zip(bars, by_country.values):
    axes[0].text(value + 0.1, bar.get_y() + bar.get_height() / 2, str(value), va="center", fontsize=9, color=INK)
axes[0].set_title("Projects by country")
axes[0].grid(axis="y", visible=False)
budget_type = (df_projects_feat.groupby("project_type")["budget_usd"].sum() / 1e6).sort_values()
bars = axes[1].barh(budget_type.index, budget_type.values, color=BLUE, height=0.6)
for bar, value in zip(bars, budget_type.values):
    axes[1].text(value + 5, bar.get_y() + bar.get_height() / 2, f"${value:,.0f}M", va="center", fontsize=9, color=INK)
axes[1].set_xlim(0, budget_type.max() * 1.15)
axes[1].set_title("Total budget by project type (USD millions)")
axes[1].grid(axis="y", visible=False)
finish(fig, "01_portfolio_overview", f"{len(df_projects_feat)} projects across {df_projects_feat['country'].nunique()} countries")

# ---------- Chart 2: Objective 1, how big are the overruns and where ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
axes[0].hist(100 * ledger_eda["overrun_pct"], bins=40, color=BLUE, edgecolor="white", linewidth=0.6)
axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.15)
axes[0].axvline(15, color=ORANGE, linewidth=1.8, linestyle="--")
axes[0].text(15.8, axes[0].get_ylim()[1] * 0.94, "15% overrun line", color=ORANGE, fontsize=9)
axes[0].set_xlabel("Overrun (% above budget)")
axes[0].set_ylabel("Cost lines")
axes[0].set_title("Most cost lines finish over budget")
rate_bars(axes[1], ledger_eda, "cost_category", "overrun_flag", "Share of cost lines more than 15% over budget")
finish(fig, "02_cost_overruns_by_category", "Objective 1: where cost overruns come from")

# ---------- Chart 3: Objective 1, country, project type, contract ----------
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
rate_bars(axes[0], ledger_eda, "country", "overrun_flag", "By country")
rate_bars(axes[1], ledger_eda, "project_type", "overrun_flag", "By project type")
rate_bars(axes[2], ledger_eda, "contract_type", "overrun_flag", "By contract type")
finish(fig, "03_cost_overruns_by_project", "Objective 1: overrun rate by country, project type and contract")

# ---------- Chart 4: Objective 1, trend over time ----------
quarterly = (ledger_eda.groupby("period")["overrun_flag"].agg(["mean", "count"]))
quarterly = quarterly[quarterly["count"] >= 30]
fig, ax = plt.subplots(figsize=(14, 4.2))
ax.plot(quarterly.index, 100 * quarterly["mean"], color=BLUE, linewidth=2, marker="o", markersize=5)
ax.axhline(100 * ledger_eda["overrun_flag"].mean(), color=ORANGE, linewidth=1.4, linestyle="--")
ax.set_xlabel("Dashed line = overall average", color=ORANGE)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
ax.set_ylabel("Cost lines over 15% over budget")
ax.tick_params(axis="x", rotation=45)
ax.set_title("Overrun rate by quarter (quarters with 30 or more cost lines)")
finish(fig, "04_cost_overruns_over_time", "Objective 1: overruns over time")

# ---------- Chart 5: Objective 2, what drives delays ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
rate_bars(axes[0, 0], tasks_eda, "task_type", "delay_flag", "By task type")
rate_bars(axes[0, 1], tasks_eda, "rainy_label", "delay_flag", "By season")
rate_bars(axes[1, 0], tasks_eda, "reliability_rating", "delay_flag", "By vendor reliability rating (1 to 5)")
rate_bars(axes[1, 1], tasks_eda, "crew_band", "delay_flag", "By crew size", order=["Small", "Medium", "Large"])
finish(fig, "05_task_delays_drivers", "Objective 2: share of completed tasks finishing more than 5 days late")

# ---------- Chart 6: Objective 2, size of the delays and timing ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
axes[0].hist(tasks_eda["delay_days"], bins=np.arange(tasks_eda["delay_days"].min() - 0.5, tasks_eda["delay_days"].max() + 1.5, 1),
             color=BLUE, edgecolor="white", linewidth=0.6)
axes[0].set_ylim(0, axes[0].get_ylim()[1] * 1.15)
axes[0].axvline(5, color=ORANGE, linewidth=1.8, linestyle="--")
axes[0].text(6, axes[0].get_ylim()[1] * 0.94, "5-day delay line", color=ORANGE, fontsize=9)
axes[0].set_xlabel("Days late (negative means early)")
axes[0].set_ylabel("Completed tasks")
axes[0].set_title("How late do tasks finish?")
month_rate = rate_table(tasks_eda, "planned_start_month", "delay_flag").sort_index()
axes[1].bar(month_rate.index.astype(int), month_rate["mean"], color=BLUE, width=0.6)
axes[1].axhline(100 * tasks_eda["delay_flag"].mean(), color=ORANGE, linewidth=1.4, linestyle="--")
axes[1].set_xticks(range(1, 13))
axes[1].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
axes[1].set_xlabel("Month the task was planned to start")
axes[1].set_title("Delay rate by planned start month")
axes[1].grid(axis="x", visible=False)
finish(fig, "06_task_delay_size_and_timing", "Objective 2: size and timing of delays")

# ---------- Chart 7: Objective 3, safety ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
rate_bars(axes[0, 0], incidents_eda, "incident_type", "serious_flag", "By incident type")
rate_bars(axes[0, 1], incidents_eda, "ppe_height_label", "serious_flag", "Working at height without PPE")
rate_bars(axes[1, 0], incidents_eda, "experience_band", "serious_flag", "By worker experience",
          order=["Novice", "Junior", "Experienced", "Veteran"])
rate_bars(axes[1, 1], incidents_eda, "weather_condition", "serious_flag", "By weather")
finish(fig, "07_safety_serious_incidents", "Objective 3: share of incidents that are serious (Lost Time Injury or Fatality)")

# ---------- Chart 8: Objective 4, late deliveries ----------
fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
rate_bars(axes[0, 0], orders_eda, "supplier_label", "late_delivery", "Domestic or international supplier")
rate_bars(axes[0, 1], orders_eda, "material", "late_delivery", "By material")
rate_bars(axes[1, 0], orders_eda, "reliability_rating", "late_delivery", "By vendor reliability rating (1 to 5)")
rate_bars(axes[1, 1], orders_eda, "lead_band", "late_delivery", "By promised lead time",
          order=["Up to 2 weeks", "2-4 weeks", "4-6 weeks", "Over 6 weeks"])
finish(fig, "08_late_deliveries", "Objective 4: share of material orders delivered late")

# ---------- Chart 9: Objective 5, price anomalies (no answer key used) ----------
ratio = df_material_orders_feat["price_ratio_material"].dropna()
spike_share = (
    df_material_orders_feat.assign(spike=df_material_orders_feat["price_ratio_material"] > 2)
    .groupby("material")["spike"].mean().mul(100).sort_values()
)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
axes[0].hist(ratio, bins=np.linspace(0, 8, 65), color=BLUE, edgecolor="white", linewidth=0.5)
axes[0].set_yscale("log")
axes[0].set_ylim(0.8, 4000)
axes[0].axvline(2, color=ORANGE, linewidth=1.8, linestyle="--")
axes[0].text(2.1, 1500, "2x the usual price", color=ORANGE, fontsize=9)
axes[0].set_xlabel("Unit price / median price of the same material")
axes[0].set_ylabel("Orders (log scale)")
axes[0].set_title("A small group of prices sits far above normal (log scale)")
bars = axes[1].barh(spike_share.index, spike_share.values, color=BLUE, height=0.6)
for bar, value in zip(bars, spike_share.values):
    axes[1].text(value + 0.05, bar.get_y() + bar.get_height() / 2, f"{value:.1f}%", va="center", fontsize=9, color=INK)
axes[1].set_xlim(0, max(spike_share.max() * 1.25, 1))
axes[1].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}%"))
axes[1].grid(axis="y", visible=False)
axes[1].set_title("Orders priced above 2x the material median")
finish(fig, "09_price_anomalies", "Objective 5: unusual unit prices")

# ---------- Factor ranking: which factors separate the groups most ----------
def factor_gap(frame, objective, target, columns):
    rows = []
    for column in columns:
        table = rate_table(frame, column, target, min_n=30)
        if len(table) < 2:
            continue
        rows.append({
            "objective": objective, "factor": column,
            "highest_group": str(table["mean"].idxmax()), "highest_rate_pct": round(table["mean"].max(), 1),
            "lowest_group": str(table["mean"].idxmin()), "lowest_rate_pct": round(table["mean"].min(), 1),
            "gap_points": round(table["mean"].max() - table["mean"].min(), 1),
        })
    return rows

ranking = pd.DataFrame(
    factor_gap(ledger_eda, "1 Cost overrun", "overrun_flag",
               ["cost_category", "country", "project_type", "contract_type", "client_sector", "size_band", "high_rise", "period_year"])
    + factor_gap(tasks_eda, "2 Schedule delay", "delay_flag",
                 ["task_type", "rainy_season", "crew_band", "reliability_rating", "country", "project_type", "rework_required", "vendor_known"])
    + factor_gap(incidents_eda, "3 Serious incident", "serious_flag",
                 ["incident_type", "no_ppe_at_height", "ppe_compliant", "work_at_height", "experience_band", "weather_condition", "shift"])
    + factor_gap(orders_eda, "4 Late delivery", "late_delivery",
                 ["international_supplier", "material", "rainy_season", "reliability_rating", "lead_band", "country"])
).sort_values(["objective", "gap_points"], ascending=[True, False]).reset_index(drop=True)
ranking.to_csv(EDA_DIR / "factor_ranking.csv", index=False)

display(ranking)
print(f"Saved {len(saved_charts)} charts and factor_ranking.csv to {EDA_DIR}/")
```

---

## 9. Analytical Base Table

One table per objective with only what a model is allowed to see: the input features and the target. Everything that leaks the answer is dropped here.

| Objective | Rows | Input columns (numeric + text) | Target |
|---|---:|---|---|
| 1 Cost overrun | 2,413 | 16 (12 + 4) | `overrun_flag` |
| 2 Schedule delay | 3,237 (finished tasks) | 18 (14 + 4) | `delay_flag` |
| 3 Serious incident | 750 | 15 (10 + 5) | `serious_flag` |
| 4 Late delivery | 2,918 (delivered orders) | 15 (11 + 4) | `late_delivery` |
| 5 Price anomalies | 3,000 | 4 (all numeric price ratios) | none |

Objective 5 is unsupervised, so its table holds four price features only: price against the material norm, price against similar orders (same material and country), log unit price and log quantity.

### Code: analytical base table

```python
# ============================================================
# ANALYTICAL BASE TABLE PHASE
# ============================================================

# %%
from pathlib import Path

ABT_DIR = Path("construction_abt")
ABT_DIR.mkdir(exist_ok=True)

# ---------- 1. Source tables: each fact joined to the project and vendor facts known at the time ----------
project_cols = {
    "project_id": "project_id", "country": "country", "project_type": "project_type",
    "contract_type": "contract_type", "budget_usd": "project_budget_usd",
    "site_area_sqm": "site_area_sqm", "num_floors": "num_floors",
    "planned_duration_days": "project_planned_days",
    "project_manager_experience_years": "manager_experience_years",
    "high_rise": "high_rise", "is_public": "is_public", "budget_per_sqm": "budget_per_sqm",
}
vendor_cols = {
    "vendor_id": "vendor_id", "vendor_type": "vendor_type", "reliability_rating": "reliability_rating",
    "safety_rating": "vendor_safety_rating", "years_active": "vendor_years_active",
    "is_prequalified": "is_prequalified",
}
projects_part = df_projects_feat[list(project_cols)].rename(columns=project_cols)
vendors_part = df_vendors_feat[list(vendor_cols)].rename(columns=vendor_cols)

ledger_src = df_cost_ledger_feat.merge(projects_part, on="project_id", how="left")

tasks_src = (df_tasks_feat.rename(columns={"planned_duration_days": "task_planned_days"})
             .merge(projects_part, on="project_id", how="left")
             .merge(vendors_part, on="vendor_id", how="left"))

orders_src = (df_material_orders_feat.merge(projects_part, on="project_id", how="left")
              .merge(vendors_part, on="vendor_id", how="left"))
orders_src["log_unit_price"] = np.log1p(orders_src["unit_price_usd"]).round(4)

incidents_src = (df_safety_incidents_feat.merge(projects_part, on="project_id", how="left")
                 .merge(vendors_part, on="vendor_id", how="left"))
incidents_src["ppe_compliant_yes"] = (incidents_src["ppe_compliant"] == "Yes").astype(int)
incidents_src["work_at_height_yes"] = (incidents_src["work_at_height"] == "Yes").astype(int)

# ---------- 2. What each objective is allowed to see ----------
# Only information known when the decision is made goes in. Anything that is a result of the
# outcome (actual cost, actual dates, severity, days lost, delivery date, answer key) is listed as leakage.
OBJECTIVES = {
    "cost_overrun": dict(
        title="Objective 1: cost overrun", source=ledger_src, id_col="ledger_id",
        target="overrun_flag", secondary_target="overrun_pct", task="classification",
        numeric=["log_budgeted_usd", "budget_share_of_project", "period_year", "period_quarter",
                 "project_budget_usd", "site_area_sqm", "num_floors", "project_planned_days",
                 "manager_experience_years", "high_rise", "is_public", "budget_per_sqm"],
        categorical=["cost_category", "country", "project_type", "contract_type"],
        leakage=["actual_usd", "overrun_usd", "overrun_pct", "overrun_flag", "approved_change_orders_usd", "change_order_pct"],
    ),
    "schedule_delay": dict(
        title="Objective 2: schedule delay", source=tasks_src, id_col="task_id",
        target="delay_flag", secondary_target="delay_days", task="classification",
        numeric=["task_planned_days", "crew_size", "planned_start_month", "planned_start_year", "rainy_season",
                 "reliability_rating", "vendor_safety_rating", "vendor_years_active", "is_prequalified",
                 "vendor_known", "project_planned_days", "num_floors", "site_area_sqm", "manager_experience_years"],
        categorical=["task_type", "country", "project_type", "contract_type"],
        leakage=["actual_start", "actual_end", "actual_duration_days", "delay_days", "delay_flag",
                 "task_status", "rework_required"],
    ),
    "site_safety": dict(
        title="Objective 3: serious safety incident", source=incidents_src, id_col="incident_id",
        target="serious_flag", secondary_target=None, task="classification",
        numeric=["worker_experience_years", "ppe_compliant_yes", "work_at_height_yes", "no_ppe_at_height",
                 "incident_month", "vendor_safety_rating", "vendor_years_active", "vendor_known",
                 "num_floors", "high_rise"],
        categorical=["incident_type", "shift", "weather_condition", "project_type", "country"],
        leakage=["severity", "serious_flag", "lost_days", "reported_delay_hours"],
    ),
    "late_delivery": dict(
        title="Objective 4: late material delivery", source=orders_src, id_col="order_id",
        target="late_delivery", secondary_target="delivery_delay_days", task="classification",
        numeric=["log_quantity", "unit_price_usd", "lead_time_days", "order_month", "order_year", "rainy_season",
                 "international_supplier", "reliability_rating", "vendor_safety_rating", "vendor_years_active",
                 "is_prequalified"],
        categorical=["material", "vendor_type", "country", "project_type"],
        leakage=["delivered_date", "actual_lead_days", "delivery_delay_days", "late_delivery", "quality_passed"],
    ),
    "price_anomalies": dict(
        title="Objective 5: price and invoice anomalies", source=orders_src, id_col="order_id",
        target=None, secondary_target=None, task="unsupervised",
        numeric=["price_ratio_material", "price_ratio_peer", "log_unit_price", "log_quantity"],
        categorical=[],
        leakage=["is_anomaly", "total_vs_calculated_pct"],    # the answer key is not loaded until final scoring
    ),
}

# ---------- 3. Build one ABT per objective ----------
abts, case_infos, summary_rows = {}, {}, []

for key, cfg in OBJECTIVES.items():
    source = cfg["source"].copy()
    features = cfg["numeric"] + cfg["categorical"]

    # guard: no leaking column may be used as a feature
    overlap = set(features) & set(cfg["leakage"])
    assert not overlap, f"{key}: leaking columns in the feature list: {overlap}"

    if cfg["target"]:
        source = source.dropna(subset=[cfg["target"]])
    source = source.drop_duplicates(subset=cfg["id_col"]).reset_index(drop=True)

    columns = features + ([cfg["target"]] if cfg["target"] else [])
    abt = source[columns].copy()
    for column in cfg["categorical"]:
        abt[column] = abt[column].astype("string")
    if cfg["target"]:
        abt[cfg["target"]] = abt[cfg["target"]].astype(int)      # 0 / 1

    # labels kept aside: row id and project (so the split step can keep a project together) and the second target
    info_cols = [cfg["id_col"], "project_id"] + ([cfg["secondary_target"]] if cfg["secondary_target"] else [])
    case_info = source[info_cols].copy()

    abt.to_csv(ABT_DIR / f"{key}_abt.csv", index=False)
    case_info.to_csv(ABT_DIR / f"{key}_case_info.csv", index=False)
    abts[key], case_infos[key] = abt, case_info

    summary_rows.append({
        "objective": cfg["title"], "rows": len(abt),
        "numeric_features": len(cfg["numeric"]), "text_features": len(cfg["categorical"]),
        "target": cfg["target"] or "none (unsupervised)",
        "positive_rate_pct": round(100 * abt[cfg["target"]].mean(), 1) if cfg["target"] else None,
        "missing_cells": int(abt[features].isna().sum().sum()),
    })

abt_summary = pd.DataFrame(summary_rows).set_index("objective")

missing_report = pd.DataFrame([
    {"objective": OBJECTIVES[key]["title"], "column": column, "missing": int(count),
     "missing_pct": round(100 * count / len(abt), 1)}
    for key, abt in abts.items()
    for column, count in abt.isna().sum().items() if count
])

leakage_report = pd.DataFrame([
    {"objective": cfg["title"], "left_out_on_purpose": ", ".join(cfg["leakage"])}
    for cfg in OBJECTIVES.values()
]).set_index("objective")

# ---------- 4. Verification ----------
display(abt_summary)
display(missing_report)
display(leakage_report)
for key, abt in abts.items():
    print(f"\n{OBJECTIVES[key]['title']}  ({abt.shape[0]} rows x {abt.shape[1]} columns)")
    display(abt.head(5))
print(f"\nSaved {len(abts)} analytical base tables to {ABT_DIR}/")
```

---

## 10. Encode Categorical Columns

Models need numbers. Text columns become 0/1 columns, one per category. Missing text becomes its own `Unknown` label so no row is lost, and a category the model has never seen encodes as all zeros instead of crashing the app. The encoders are saved so new rows in the app are encoded the same way.

| Objective | Columns before | Columns after encoding |
|---|---:|---:|
| 1 Cost overrun | 17 | 35 |
| 2 Schedule delay | 19 | 39 |
| 3 Serious incident | 16 | 37 |
| 4 Late delivery | 16 | 33 |
| 5 Price anomalies | 4 | 4 (nothing to encode) |

### Code: encode categorical columns

```python
# ============================================================
# ENCODE CATEGORICAL COLUMNS
# ============================================================

# %%
import json
import joblib
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

ENCODED_DIR = Path("construction_encoded")
ENCODED_DIR.mkdir(exist_ok=True)

# Missing text values get their own label, so no row is lost and nothing is learned from the data.
UNKNOWN = "Unknown"

encoded_tables, encoders, encoding_rows, category_rows = {}, {}, [], []

for key, cfg in OBJECTIVES.items():
    abt = abts[key]
    categorical = cfg["categorical"]
    numeric = cfg["numeric"]
    target = [cfg["target"]] if cfg["target"] else []

    if not categorical:
        # objective 5 has numbers only, so there is nothing to encode
        encoded = abt.copy()
        encoders[key] = None
        encoded_columns = []
    else:
        text = abt[categorical].astype("object").where(abt[categorical].notna(), UNKNOWN)
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(text)
        encoded_columns = list(encoder.get_feature_names_out(categorical))
        dummies = pd.DataFrame(encoder.transform(text), columns=encoded_columns, index=abt.index).astype(int)

        encoded = pd.concat([abt[numeric], dummies, abt[target]], axis=1)
        encoders[key] = encoder

        for column, levels in zip(categorical, encoder.categories_):
            category_rows.append({
                "objective": cfg["title"], "column": column, "levels": len(levels),
                "categories": ", ".join(str(level) for level in levels),
            })

    # checks: same rows, no text columns left, no gaps in the new columns
    assert len(encoded) == len(abt), f"{key}: row count changed"
    assert not encoded.select_dtypes(include=["object", "string"]).columns.size, f"{key}: text columns remain"
    if encoded_columns:
        assert encoded[encoded_columns].isna().sum().sum() == 0, f"{key}: gaps in encoded columns"

    encoded.to_csv(ENCODED_DIR / f"{key}_encoded.csv", index=False)
    encoded_tables[key] = encoded
    encoding_rows.append({
        "objective": cfg["title"], "rows": len(encoded),
        "columns_before": abt.shape[1], "columns_after": encoded.shape[1],
        "text_columns_encoded": len(categorical), "new_columns": len(encoded_columns),
        "numeric_missing_left_for_pipeline": int(encoded[numeric].isna().sum().sum()),
    })

# save the encoders and the exact column order, so the app can encode new rows the same way
joblib.dump(encoders, ENCODED_DIR / "encoders.joblib")
column_spec = {
    key: {"numeric": OBJECTIVES[key]["numeric"], "categorical": OBJECTIVES[key]["categorical"],
          "target": OBJECTIVES[key]["target"], "encoded_columns": list(table.columns)}
    for key, table in encoded_tables.items()
}
with open(ENCODED_DIR / "column_spec.json", "w") as handle:
    json.dump(column_spec, handle, indent=2)

# ---------- Verification ----------
encoding_summary = pd.DataFrame(encoding_rows).set_index("objective")
category_table = pd.DataFrame(category_rows)

# an unseen label must not crash scoring: it encodes as all zeros
sample_key = "cost_overrun"
probe = pd.DataFrame([{column: "Never seen before" for column in OBJECTIVES[sample_key]["categorical"]}])
probe_row = encoders[sample_key].transform(probe)

display(encoding_summary)
display(category_table)
print(f"An unseen category encodes as all zeros: {int(probe_row.sum()) == 0}")
display(encoded_tables["cost_overrun"].head(5))
print(f"\nSaved {len(encoded_tables)} encoded tables, encoders.joblib and column_spec.json to {ENCODED_DIR}/")
```

---

## 11. Separate X and y

For each objective the target becomes `y` and the allowed inputs become `X`. Checks confirm that no target or leakage column is hiding inside `X`.

| Objective | Rows | Features in X | Target y | Share positive |
|---|---:|---:|---|---:|
| 1 Cost overrun | 2,413 | 34 | `overrun_flag` | 46.7% |
| 2 Schedule delay | 3,237 | 38 | `delay_flag` | 40.9% |
| 3 Serious incident | 750 | 36 | `serious_flag` | 25.1% |
| 4 Late delivery | 2,918 | 32 | `late_delivery` | 45.9% |
| 5 Price anomalies | 3,000 | 4 | none | n/a |

### Code: separate X and y

```python
# ============================================================
# SEPARATE X AND Y
# ============================================================

# %%
from pathlib import Path

XY_DIR = Path("construction_xy")
XY_DIR.mkdir(exist_ok=True)

X_all, y_all, y_second, groups_all, xy_rows = {}, {}, {}, {}, []

for key, cfg in OBJECTIVES.items():
    encoded = encoded_tables[key]
    case_info = case_infos[key]
    target = cfg["target"]

    # X = every column the model may learn from. y = the answer it must predict.
    X = encoded.drop(columns=[target]) if target else encoded.copy()
    y = encoded[target] if target else None

    # the second target (the amount, not the yes/no) stays aside for an optional regression model
    second_name = cfg["secondary_target"]
    second = case_info[second_name] if second_name else None

    # project of each row, so the split step can keep a project's rows together
    groups = case_info["project_id"]

    # ---- safety checks ----
    assert X.index.equals(case_info.index), f"{key}: X and case_info rows are not aligned"
    assert len(X) == len(groups), f"{key}: groups do not match X"
    if target:
        assert target not in X.columns, f"{key}: target is still inside X"
        assert len(y) == len(X), f"{key}: X and y have different lengths"
        assert y.isin([0, 1]).all(), f"{key}: target is not 0/1"
        assert y.notna().all(), f"{key}: target has gaps"
    banned = set(cfg["leakage"]) & set(X.columns)
    assert not banned, f"{key}: leaking columns inside X: {banned}"
    assert not X.select_dtypes(exclude="number").columns.size, f"{key}: X still has non-numeric columns"

    X.to_csv(XY_DIR / f"{key}_X.csv", index=False)
    if target:
        y.to_csv(XY_DIR / f"{key}_y.csv", index=False)
    if second is not None:
        second.to_csv(XY_DIR / f"{key}_y_second.csv", index=False)
    groups.to_csv(XY_DIR / f"{key}_groups.csv", index=False)

    X_all[key], y_all[key], y_second[key], groups_all[key] = X, y, second, groups
    xy_rows.append({
        "objective": cfg["title"], "rows": len(X), "features_in_X": X.shape[1],
        "target_y": target or "none (unsupervised)",
        "positive_rate_pct": round(100 * y.mean(), 1) if target else None,
        "second_target": second_name or "-",
        "missing_cells_in_X": int(X.isna().sum().sum()),
        "projects": groups.nunique(),
    })

xy_summary = pd.DataFrame(xy_rows).set_index("objective")

# ---------- Verification ----------
display(xy_summary)
for key, cfg in OBJECTIVES.items():
    print(f"\n{cfg['title']}: X is {X_all[key].shape[0]} rows x {X_all[key].shape[1]} columns")
    print("First features:", ", ".join(X_all[key].columns[:6]), "...")
    if cfg["target"]:
        counts = y_all[key].value_counts().sort_index()
        print(f"y '{cfg['target']}': {counts.get(0, 0)} zeros, {counts.get(1, 0)} ones")
print(f"\nSaved X, y, second targets and project groups to {XY_DIR}/")
```

---

## 12. Train Test Split

The split keeps **whole projects together** (`GroupShuffleSplit`): a project is either entirely in training or entirely in testing. A random row split would let the model memorise a project and make the results look better than they are.

| Objective | Train rows | Test rows | Train projects | Test projects |
|---|---:|---:|---:|---:|
| 1 Cost overrun | 1,931 | 482 | 64 | 16 |
| 2 Schedule delay | 2,591 | 646 | 64 | 16 |
| 3 Serious incident | 598 | 152 | 64 | 16 |
| 4 Late delivery | 2,336 | 582 | 64 | 16 |
| 5 Price anomalies | 3,000 | none | 80 | none |

The share of positive cases is almost identical in train and test, and a check confirms that no project appears in both. Objective 5 has no target, so all orders stay together.

### Code: train test split

```python
# ============================================================
# TRAIN TEST SPLIT
# ============================================================

# %%
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

SPLIT_DIR = Path("construction_split")
SPLIT_DIR.mkdir(exist_ok=True)

TEST_SIZE = 0.20

def best_group_split(X, y, groups, test_size=TEST_SIZE, tries=300):
    """Keep every project wholly in train or wholly in test, and choose the shuffle whose
    test share and positive rate are closest to the full data."""
    overall_rate = y.mean()
    best = None
    for seed in range(tries):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        gap = abs(len(test_idx) / len(X) - test_size) + abs(y.iloc[test_idx].mean() - overall_rate)
        if best is None or gap < best[0]:
            best = (gap, seed, train_idx, test_idx)
    return best[1], best[2], best[3]

splits, split_rows = {}, []

for key, cfg in OBJECTIVES.items():
    X, y, groups = X_all[key], y_all[key], groups_all[key]
    case_info = case_infos[key]

    if cfg["target"] is None:
        # objective 5 has no answer column: the model looks at every order, and the
        # answer key is only opened at the very end to score it
        splits[key] = {"X_train": X, "X_test": None, "y_train": None, "y_test": None}
        split_rows.append({"objective": cfg["title"], "train_rows": len(X), "test_rows": 0,
                           "train_projects": groups.nunique(), "test_projects": 0,
                           "train_positive_pct": None, "test_positive_pct": None, "seed": None})
        continue

    seed, train_idx, test_idx = best_group_split(X, y, groups)

    parts = {
        "X_train": X.iloc[train_idx], "X_test": X.iloc[test_idx],
        "y_train": y.iloc[train_idx], "y_test": y.iloc[test_idx],
    }
    second = y_second[key]
    if second is not None:
        parts["y_second_train"], parts["y_second_test"] = second.iloc[train_idx], second.iloc[test_idx]

    # ---- safety checks ----
    train_projects, test_projects = set(groups.iloc[train_idx]), set(groups.iloc[test_idx])
    assert not train_projects & test_projects, f"{key}: a project appears in both train and test"
    assert len(train_idx) + len(test_idx) == len(X), f"{key}: rows were lost in the split"
    assert not set(train_idx) & set(test_idx), f"{key}: a row appears in both sets"
    assert set(parts["y_train"].unique()) == {0, 1} and set(parts["y_test"].unique()) == {0, 1}, f"{key}: a set has only one class"

    for name, part in parts.items():
        part.to_csv(SPLIT_DIR / f"{key}_{name}.csv", index=False)
    case_info.iloc[test_idx].to_csv(SPLIT_DIR / f"{key}_test_case_info.csv", index=False)

    splits[key] = parts
    split_rows.append({
        "objective": cfg["title"], "train_rows": len(train_idx), "test_rows": len(test_idx),
        "train_projects": len(train_projects), "test_projects": len(test_projects),
        "train_positive_pct": round(100 * parts["y_train"].mean(), 1),
        "test_positive_pct": round(100 * parts["y_test"].mean(), 1), "seed": seed,
    })

split_summary = pd.DataFrame(split_rows).set_index("objective")
split_summary["test_share_pct"] = (100 * split_summary["test_rows"] / (split_summary["train_rows"] + split_summary["test_rows"])).round(1)

# ---------- Verification ----------
display(split_summary)
print("No project appears in both train and test, for every objective with a target.")
print("Objective 5 keeps all orders together (no target to split on).")
print(f"\nSaved the split tables to {SPLIT_DIR}/")
```

---

## 13. Feature Scaling of Numerical Columns

Numeric columns are scaled so large numbers (budgets in millions) do not drown out small ones (ratings from 1 to 5). A median imputer fills gaps and `StandardScaler` rescales; 0/1 columns get a most-frequent imputer and are left unscaled. Everything is fitted on the **training rows only**, so nothing from the test set leaks in.

| Objective | Features | Scaled numeric | Left as 0 or 1 | Gaps filled (train / test) |
|---|---:|---:|---:|---|
| 1 Cost overrun | 34 | 10 | 24 | 0 / 0 |
| 2 Schedule delay | 38 | 11 | 27 | 235 / 55 |
| 3 Serious incident | 36 | 5 | 31 | 70 / 8 |
| 4 Late delivery | 32 | 8 | 24 | 205 / 52 |
| 5 Price anomalies | 4 | 4 | 0 | 0 / 0 |

After scaling, every training column has a mean of 0 and a standard deviation of 1.

### Code: feature scaling

```python
# ============================================================
# FEATURE SCALING OF NUMERICAL COLUMNS
# ============================================================

# %%
import json
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCALED_DIR = Path("construction_scaled")
SCALED_DIR.mkdir(exist_ok=True)

def build_scaler(X_fit):
    """Scale the measured numbers (money, days, counts). Leave the 0/1 columns as they are.
    Gaps are filled using the fitting rows only: median for scaled numbers, most common value for 0/1 columns."""
    scale_cols = [c for c in X_fit.columns if X_fit[c].nunique(dropna=True) > 2]
    flag_cols = [c for c in X_fit.columns if c not in scale_cols]
    transformer = ColumnTransformer(
        [("numbers", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), scale_cols),
         ("flags", SimpleImputer(strategy="most_frequent"), flag_cols)],
        verbose_feature_names_out=False,
    )
    transformer.set_output(transform="pandas")
    return transformer, scale_cols, flag_cols

scalers, scaled, feature_lists, scaling_rows = {}, {}, {}, []

for key, cfg in OBJECTIVES.items():
    X_train = splits[key]["X_train"]
    X_test = splits[key]["X_test"]

    # the scaler learns its averages and spreads from the TRAINING rows only
    scaler, scale_cols, flag_cols = build_scaler(X_train)
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test) if X_test is not None else None

    # ---- safety checks ----
    assert train_scaled.isna().sum().sum() == 0, f"{key}: gaps left in scaled train"
    assert test_scaled is None or test_scaled.isna().sum().sum() == 0, f"{key}: gaps left in scaled test"
    assert list(train_scaled.columns) == scale_cols + flag_cols, f"{key}: column order changed"
    assert len(train_scaled) == len(X_train), f"{key}: rows changed"
    assert np.allclose(train_scaled[scale_cols].mean(), 0, atol=1e-6), f"{key}: train mean is not 0"

    train_scaled.to_csv(SCALED_DIR / f"{key}_X_train_scaled.csv", index=False)
    if test_scaled is not None:
        test_scaled.to_csv(SCALED_DIR / f"{key}_X_test_scaled.csv", index=False)

    scalers[key] = scaler
    scaled[key] = {"X_train": train_scaled, "X_test": test_scaled}
    feature_lists[key] = {"scaled_columns": scale_cols, "flag_columns": flag_cols,
                          "feature_order": list(train_scaled.columns)}
    scaling_rows.append({
        "objective": cfg["title"], "features": train_scaled.shape[1],
        "scaled_numeric": len(scale_cols), "left_as_0_or_1": len(flag_cols),
        "gaps_filled_train": int(X_train.isna().sum().sum()),
        "gaps_filled_test": int(X_test.isna().sum().sum()) if X_test is not None else 0,
        "train_mean": round(float(train_scaled[scale_cols].mean().mean()), 4),
        "train_std": round(float(train_scaled[scale_cols].std(ddof=0).mean()), 4),
    })

joblib.dump(scalers, SCALED_DIR / "scalers.joblib")
with open(SCALED_DIR / "feature_lists.json", "w") as handle:
    json.dump(feature_lists, handle, indent=2)

scaling_summary = pd.DataFrame(scaling_rows).set_index("objective")

# ---------- Verification ----------
display(scaling_summary)

example = "cost_overrun"
columns = feature_lists[example]["scaled_columns"][:5]
before_after = pd.DataFrame({
    "before_mean": splits[example]["X_train"][columns].mean().round(2),
    "before_std": splits[example]["X_train"][columns].std(ddof=0).round(2),
    "after_mean": scaled[example]["X_train"][columns].mean().round(3),
    "after_std": scaled[example]["X_train"][columns].std(ddof=0).round(3),
    "test_mean_after": scaled[example]["X_test"][columns].mean().round(3),
})
print(f"\nBefore and after scaling, first five scaled columns of {OBJECTIVES[example]['title']}:")
display(before_after)
print("Test columns are scaled with the TRAIN averages, so their means are close to 0 but not exactly 0.")
print(f"\nSaved scaled tables, scalers.joblib and feature_lists.json to {SCALED_DIR}/")
```

---

## 14. Train Model

Three models per objective plus a "no skill" baseline to show what guessing scores, all compared with cross-validation on the training rows only. Every fold holds out whole projects (`GroupKFold`).

| Objective | Models tried | Cross-validation |
|---|---|---|
| 1 to 4 | Logistic Regression, Random Forest, Gradient Boosting | Grouped 5-fold by project, scored on average precision |
| 5 | Isolation Forest in three set-ups (200 trees, 500 trees, 500 trees with more samples) | Stability of the scores across random seeds |

**How the winner is picked:** by cross-validated score on the training data. The test set is not used to choose, only to confirm.

| Cross-validated ROC AUC | Logistic Regression | Random Forest | Gradient Boosting | Baseline |
|---|---:|---:|---:|---:|
| 1 Cost overrun | **0.746** | 0.697 | 0.670 | 0.500 |
| 2 Schedule delay | **0.776** | 0.771 | 0.760 | 0.500 |
| 3 Serious incident | **0.863** | 0.862 | 0.860 | 0.500 |
| 4 Late delivery | **0.631** | 0.631 | 0.615 | 0.500 |

Logistic Regression won all four on average precision, so the final models are simple and explainable. For anomalies, the 500-tree Isolation Forest was chosen because its scores were the most stable across seeds (0.992). It flags 150 of 3,000 orders (5%) at an anomaly-score cut-off of 0.538.

### Code: train model

#### Objective 1: Cost overruns

```python
# ============================================================
# TRAIN MODEL - BUSINESS OBJECTIVE 1: CONTROL PROJECT COST OVERRUNS
# ============================================================

# %%
import time
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GroupKFold, cross_validate

SEED = 42
KEY = "cost_overrun"

X_train = scaled[KEY]["X_train"]
y_train = splits[KEY]["y_train"]
groups_train = groups_all[KEY].loc[X_train.index]      # project of each training row

print(f"{OBJECTIVES[KEY]['title']}: {X_train.shape[0]} training rows, {X_train.shape[1]} features, "
      f"{100 * y_train.mean():.1f}% overrun")

# ---------- 1. Four models, from the simplest guess to the strongest learner ----------
models = {
    "Baseline (always predicts the common answer)": DummyClassifier(strategy="prior"),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
    "Gradient Boosting": HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_depth=4, min_samples_leaf=20,
        l2_regularization=1.0, random_state=SEED),
}

# ---------- 2. Cross-validation on TRAINING rows only (whole projects held out in each fold) ----------
cv = GroupKFold(n_splits=5)
scoring = {"roc_auc": "roc_auc", "average_precision": "average_precision",
           "precision": make_scorer(precision_score, zero_division=0),
           "recall": make_scorer(recall_score, zero_division=0),
           "f1": make_scorer(f1_score, zero_division=0)}

cv_rows = []
for name, model in models.items():
    started = time.time()
    result = cross_validate(model, X_train, y_train, groups=groups_train, cv=cv, scoring=scoring, n_jobs=1)
    cv_rows.append({
        "model": name,
        "roc_auc": result["test_roc_auc"].mean(), "roc_auc_std": result["test_roc_auc"].std(),
        "average_precision": result["test_average_precision"].mean(),
        "precision": result["test_precision"].mean(), "recall": result["test_recall"].mean(),
        "f1": result["test_f1"].mean(), "seconds": time.time() - started,
    })

cv_results = pd.DataFrame(cv_rows).set_index("model").round(3)
baseline_name = "Baseline (always predicts the common answer)"
best_name = cv_results.drop(index=baseline_name)["average_precision"].idxmax()

# ---------- 3. Fit every model on the full training set ----------
trained_models = {}
for name, model in models.items():
    trained_models[name] = model.fit(X_train, y_train)

models_obj1 = {"cv_results": cv_results, "trained": trained_models, "best_name": best_name}

# ---------- Verification ----------
display(cv_results)
print(f"\nBest model on cross-validation (average precision): {best_name}")
print("The test set has not been touched yet. It is used in the next two steps.")
```

#### Objective 2: Schedule delays

```python
# ============================================================
# TRAIN MODEL - BUSINESS OBJECTIVE 2: REDUCE SCHEDULE DELAYS
# ============================================================

# %%
import time
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GroupKFold, cross_validate

SEED = 42
KEY = "schedule_delay"

X_train = scaled[KEY]["X_train"]
y_train = splits[KEY]["y_train"]
groups_train = groups_all[KEY].loc[X_train.index]      # project of each training row

print(f"{OBJECTIVES[KEY]['title']}: {X_train.shape[0]} training rows, {X_train.shape[1]} features, "
      f"{100 * y_train.mean():.1f}% delayed tasks")

assert "scaled" in globals() and KEY in scaled, "Run the earlier steps first (Run All from the top)."

# ---------- 1. Four models, from the simplest guess to the strongest learner ----------
models = {
    "Baseline (always predicts the common answer)": DummyClassifier(strategy="prior"),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
    "Gradient Boosting": HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_depth=4, min_samples_leaf=20,
        l2_regularization=1.0, random_state=SEED),
}

# ---------- 2. Cross-validation on TRAINING rows only (whole projects held out in each fold) ----------
cv = GroupKFold(n_splits=5)
scoring = {"roc_auc": "roc_auc", "average_precision": "average_precision",
           "precision": make_scorer(precision_score, zero_division=0),
           "recall": make_scorer(recall_score, zero_division=0),
           "f1": make_scorer(f1_score, zero_division=0)}

cv_rows = []
for name, model in models.items():
    started = time.time()
    result = cross_validate(model, X_train, y_train, groups=groups_train, cv=cv, scoring=scoring, n_jobs=1)
    cv_rows.append({
        "model": name,
        "roc_auc": result["test_roc_auc"].mean(), "roc_auc_std": result["test_roc_auc"].std(),
        "average_precision": result["test_average_precision"].mean(),
        "precision": result["test_precision"].mean(), "recall": result["test_recall"].mean(),
        "f1": result["test_f1"].mean(), "seconds": time.time() - started,
    })

cv_results = pd.DataFrame(cv_rows).set_index("model").round(3)
baseline_name = "Baseline (always predicts the common answer)"
best_name = cv_results.drop(index=baseline_name)["average_precision"].idxmax()

# ---------- 3. Fit every model on the full training set ----------
trained_models = {}
for name, model in models.items():
    trained_models[name] = model.fit(X_train, y_train)

models_obj2 = {"cv_results": cv_results, "trained": trained_models, "best_name": best_name}

# ---------- Verification ----------
display(cv_results)
print(f"\nBest model on cross-validation (average precision): {best_name}")
print(f"Training tasks come from {groups_train.nunique()} projects; every fold holds out whole projects.")
print("The test set has not been touched yet. It is used in the next two steps.")
```

#### Objective 3: Site safety

```python
# ============================================================
# TRAIN MODEL - BUSINESS OBJECTIVE 3: PREVENT SERIOUS SAFETY INCIDENTS
# ============================================================

# %%
import time
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GroupKFold, cross_validate

SEED = 42
KEY = "site_safety"

X_train = scaled[KEY]["X_train"]
y_train = splits[KEY]["y_train"]
groups_train = groups_all[KEY].loc[X_train.index]      # project of each training row

print(f"{OBJECTIVES[KEY]['title']}: {X_train.shape[0]} training rows, {X_train.shape[1]} features, "
      f"{100 * y_train.mean():.1f}% serious incidents")

assert "scaled" in globals() and KEY in scaled, "Run the earlier steps first (Run All from the top)."

# ---------- 1. Four models, from the simplest guess to the strongest learner ----------
models = {
    "Baseline (always predicts the common answer)": DummyClassifier(strategy="prior"),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
    "Gradient Boosting": HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_depth=4, min_samples_leaf=20,
        l2_regularization=1.0, random_state=SEED),
}

# ---------- 2. Cross-validation on TRAINING rows only (whole projects held out in each fold) ----------
cv = GroupKFold(n_splits=5)
scoring = {"roc_auc": "roc_auc", "average_precision": "average_precision",
           "precision": make_scorer(precision_score, zero_division=0),
           "recall": make_scorer(recall_score, zero_division=0),
           "f1": make_scorer(f1_score, zero_division=0)}

cv_rows = []
for name, model in models.items():
    started = time.time()
    result = cross_validate(model, X_train, y_train, groups=groups_train, cv=cv, scoring=scoring, n_jobs=1)
    cv_rows.append({
        "model": name,
        "roc_auc": result["test_roc_auc"].mean(), "roc_auc_std": result["test_roc_auc"].std(),
        "average_precision": result["test_average_precision"].mean(),
        "precision": result["test_precision"].mean(), "recall": result["test_recall"].mean(),
        "f1": result["test_f1"].mean(), "seconds": time.time() - started,
    })

cv_results = pd.DataFrame(cv_rows).set_index("model").round(3)
baseline_name = "Baseline (always predicts the common answer)"
best_name = cv_results.drop(index=baseline_name)["average_precision"].idxmax()

# ---------- 3. Fit every model on the full training set ----------
trained_models = {}
for name, model in models.items():
    trained_models[name] = model.fit(X_train, y_train)

models_obj3 = {"cv_results": cv_results, "trained": trained_models, "best_name": best_name}

# ---------- Verification ----------
display(cv_results)
print(f"\nBest model on cross-validation (average precision): {best_name}")
print(f"Training incidents come from {groups_train.nunique()} projects; every fold holds out whole projects.")
print("The test set has not been touched yet. It is used in the next two steps.")
```

#### Objective 4: Late material delivery

```python
# ============================================================
# TRAIN MODEL - BUSINESS OBJECTIVE 4: PREVENT LATE MATERIAL DELIVERIES
# ============================================================

# %%
import time
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import GroupKFold, cross_validate

SEED = 42
KEY = "late_delivery"

X_train = scaled[KEY]["X_train"]
y_train = splits[KEY]["y_train"]
groups_train = groups_all[KEY].loc[X_train.index]      # project of each training row

print(f"{OBJECTIVES[KEY]['title']}: {X_train.shape[0]} training rows, {X_train.shape[1]} features, "
      f"{100 * y_train.mean():.1f}% late deliveries")

assert "scaled" in globals() and KEY in scaled, "Run the earlier steps first (Run All from the top)."

# ---------- 1. Four models, from the simplest guess to the strongest learner ----------
models = {
    "Baseline (always predicts the common answer)": DummyClassifier(strategy="prior"),
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=2000, random_state=SEED),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
    "Gradient Boosting": HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_depth=4, min_samples_leaf=20,
        l2_regularization=1.0, random_state=SEED),
}

# ---------- 2. Cross-validation on TRAINING rows only (whole projects held out in each fold) ----------
cv = GroupKFold(n_splits=5)
scoring = {"roc_auc": "roc_auc", "average_precision": "average_precision",
           "precision": make_scorer(precision_score, zero_division=0),
           "recall": make_scorer(recall_score, zero_division=0),
           "f1": make_scorer(f1_score, zero_division=0)}

cv_rows = []
for name, model in models.items():
    started = time.time()
    result = cross_validate(model, X_train, y_train, groups=groups_train, cv=cv, scoring=scoring, n_jobs=1)
    cv_rows.append({
        "model": name,
        "roc_auc": result["test_roc_auc"].mean(), "roc_auc_std": result["test_roc_auc"].std(),
        "average_precision": result["test_average_precision"].mean(),
        "precision": result["test_precision"].mean(), "recall": result["test_recall"].mean(),
        "f1": result["test_f1"].mean(), "seconds": time.time() - started,
    })

cv_results = pd.DataFrame(cv_rows).set_index("model").round(3)
baseline_name = "Baseline (always predicts the common answer)"
best_name = cv_results.drop(index=baseline_name)["average_precision"].idxmax()

# ---------- 3. Fit every model on the full training set ----------
trained_models = {}
for name, model in models.items():
    trained_models[name] = model.fit(X_train, y_train)

models_obj4 = {"cv_results": cv_results, "trained": trained_models, "best_name": best_name}

# ---------- Verification ----------
display(cv_results)
print(f"\nBest model on cross-validation (average precision): {best_name}")
print(f"Training orders come from {groups_train.nunique()} projects; every fold holds out whole projects.")
print("The test set has not been touched yet. It is used in the next two steps.")
```

#### Objective 5: Price and invoice anomalies

```python
# ============================================================
# TRAIN MODEL - BUSINESS OBJECTIVE 5: SPOT PRICE AND INVOICE ANOMALIES
# ============================================================

# %%
import time
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

SEED = 42
KEY = "price_anomalies"
REVIEW_SHARE = 0.05          # business assumption: the team can review about 5% of orders

assert "scaled" in globals() and KEY in scaled, "Run the earlier steps first (Run All from the top)."
X_all = scaled[KEY]["X_train"]               # all orders: there is no train/test split without labels
X_raw = splits[KEY]["X_train"]               # same rows in real units, for reading the results
case_info = case_infos[KEY]

# the answer key must stay closed until the final scoring step
assert not any(c in X_all.columns for c in ["is_anomaly", "total_vs_calculated_pct"]), "answer-key column found in X"
print(f"{OBJECTIVES[KEY]['title']}: {X_all.shape[0]} orders, {X_all.shape[1]} features {list(X_all.columns)}")
print("No labels are used. The model learns what a normal order looks like and scores how unusual each one is.\n")

# ---------- 1. Isolation Forest candidates ----------
def fit_forest(n_trees, max_samples, seed):
    return IsolationForest(n_estimators=n_trees, max_samples=max_samples, contamination=REVIEW_SHARE,
                           random_state=seed, n_jobs=-1).fit(X_all)

candidates = {"Isolation Forest, 200 trees, 256 samples": (200, 256),
              "Isolation Forest, 500 trees, 256 samples": (500, 256),
              "Isolation Forest, 500 trees, 1024 samples": (500, 1024)}

# ---------- 2. Check without labels: are the scores stable, and does a second method agree? ----------
def anomaly_score(model):
    return -model.score_samples(X_all)           # bigger = more unusual

lof = LocalOutlierFactor(n_neighbors=35, contamination=REVIEW_SHARE).fit(X_all)
lof_score = -lof.negative_outlier_factor_
lof_flag = lof.fit_predict(X_all) == -1

rows, trained_models = [], {}
for name, (n_trees, max_samples) in candidates.items():
    started = time.time()
    seed_scores = [anomaly_score(fit_forest(n_trees, max_samples, seed)) for seed in [SEED, 1, 2, 3, 4]]
    stability = np.mean([spearmanr(a, b)[0] for a, b in combinations(seed_scores, 2)])
    model = fit_forest(n_trees, max_samples, SEED)
    score = anomaly_score(model)
    flag = model.predict(X_all) == -1
    agree_with_lof = (flag & lof_flag).sum() / (flag | lof_flag).sum()
    ratio_col = "price_ratio_material"
    rows.append({"model": name, "flagged": int(flag.sum()),
                 "score_stability_across_seeds": stability,
                 "overlap_with_local_outlier_method": agree_with_lof,
                 "flagged_median_abs_log_ratio": np.abs(np.log(X_raw.loc[flag, ratio_col].clip(lower=1e-6))).median(),
                 "normal_median_abs_log_ratio": np.abs(np.log(X_raw.loc[~flag, ratio_col].clip(lower=1e-6))).median(),
                 "seconds": time.time() - started})
    trained_models[name] = model

cv_results = pd.DataFrame(rows).set_index("model").round(3)
best_name = cv_results["score_stability_across_seeds"].idxmax()
best_model = trained_models[best_name]

# ---------- 3. Score every order with the chosen model ----------
order_scores = pd.DataFrame({
    "order_id": case_info["order_id"].to_numpy(),
    "project_id": case_info["project_id"].to_numpy(),
    "anomaly_score": anomaly_score(best_model),
    "flagged": best_model.predict(X_all) == -1,
}, index=X_all.index)
order_scores["score_rank_pct"] = order_scores["anomaly_score"].rank(pct=True).round(3)
cutoff_score = order_scores.loc[order_scores["flagged"], "anomaly_score"].min()

models_obj5 = {"cv_results": cv_results, "trained": trained_models, "best_name": best_name,
               "model": best_model, "order_scores": order_scores, "cutoff_score": cutoff_score,
               "review_share": REVIEW_SHARE, "feature_order": list(X_all.columns)}

# ---------- Verification ----------
display(cv_results)
print(f"\nChosen model (most stable scores across random seeds): {best_name}")
print(f"{int(order_scores['flagged'].sum())} of {len(order_scores)} orders flagged ({order_scores['flagged'].mean():.1%}) "
      f"| anomaly score cut-off {cutoff_score:.3f}")
print("\nFlagged orders next to normal ones, in real units (medians):")
compare_cols = [c for c in ["price_ratio_material", "price_ratio_peer", "unit_price_usd", "log_unit_price", "log_quantity"] if c in X_raw.columns]
display(X_raw.groupby(order_scores["flagged"].map({True: "flagged", False: "normal"}))[compare_cols].median().round(2))
print("\nTop 10 most unusual orders:")
display(order_scores.sort_values("anomaly_score", ascending=False).head(10))
print("\nThe answer key (is_anomaly) has not been opened. It is used only at the final scoring step.")
```

---

## 15. Predict X_test

Every trained model scores the test rows it has never seen. Classification models give a **risk probability** instead of a hard yes/no, so the business can choose how many cases to act on.

A sanity check follows: the average predicted risk of cases that really were problems should be clearly higher than for cases that were fine.

| Objective | Avg risk, real problems | Avg risk, other cases | Gap |
|---|---:|---:|---:|
| 1 Cost overrun | 0.680 | 0.329 | 0.351 |
| 2 Schedule delay | 0.601 | 0.321 | 0.280 |
| 3 Serious incident | 0.499 | 0.164 | 0.335 |
| 4 Late delivery | 0.486 | 0.432 | 0.054 |

The gap is clear for objectives 1 to 3 and tiny for objective 4, which already hints that late delivery is hard to predict. For anomalies, every order gets a score, 150 are flagged, and each flag comes with its main reason: 88 priced above the material norm, 39 below it, 19 with an unusual quantity.

### Code: predict X_test

#### Objective 1: Cost overruns

```python
# ============================================================
# PREDICT X_TEST - BUSINESS OBJECTIVE 1: CONTROL PROJECT COST OVERRUNS
# ============================================================

# %%
KEY = "cost_overrun"

X_test = scaled[KEY]["X_test"]
y_test = splits[KEY]["y_test"]
trained_models = models_obj1["trained"]
baseline_name = "Baseline (always predicts the common answer)"

# ---------- 1. Pre-flight checks ----------
assert len(X_test) == len(y_test), "X_test and y_test have different lengths"
assert list(X_test.columns) == feature_lists[KEY]["feature_order"], "X_test columns are not in the training order"
assert hasattr(trained_models["Logistic Regression"], "predict_proba"), "the trained models are missing"

# ---------- 2. Every model gives the chance that each cost line overruns by more than 15% ----------
test_proba = {name: model.predict_proba(X_test)[:, 1] for name, model in trained_models.items()}

predictions = pd.DataFrame({
    "ledger_id": case_infos[KEY].loc[X_test.index, "ledger_id"].to_numpy(),
    "project_id": case_infos[KEY].loc[X_test.index, "project_id"].to_numpy(),
    "actual_overrun_flag": y_test.to_numpy(),
    "actual_overrun_pct": (100 * y_second[KEY].loc[X_test.index]).round(1).to_numpy(),
}, index=X_test.index)
for name, proba in test_proba.items():
    predictions[name] = proba.round(3)

# ---------- 3. Sanity check: real overruns should get a clearly higher chance than the rest ----------
is_overrun = y_test.to_numpy() == 1
risk_check = pd.DataFrame({
    "avg chance, real overruns": {n: p[is_overrun].mean() for n, p in test_proba.items()},
    "avg chance, other lines": {n: p[~is_overrun].mean() for n, p in test_proba.items()},
})
risk_check["gap"] = risk_check.iloc[:, 0] - risk_check.iloc[:, 1]

print(f"{len(predictions):,} test cost lines scored by {len(test_proba)} models | "
      f"real overruns: {int(is_overrun.sum())} ({is_overrun.mean():.1%}) | "
      f"{predictions['project_id'].nunique()} test projects the models have never seen\n")
print("A useful model gives real overruns a clearly higher average chance (a bigger gap is better):")
display(risk_check.round(3))
print("\nFirst 10 test cost lines (each model column = chance of a more than 15% overrun):")
display(predictions.head(10))
```

#### Objective 2: Schedule delays

```python
# ============================================================
# PREDICT X_TEST - BUSINESS OBJECTIVE 2: REDUCE SCHEDULE DELAYS
# ============================================================

# %%
KEY = "schedule_delay"

X_test = scaled[KEY]["X_test"]
y_test = splits[KEY]["y_test"]
trained_models = models_obj2["trained"]
baseline_name = "Baseline (always predicts the common answer)"

# ---------- 1. Pre-flight checks ----------
assert len(X_test) == len(y_test), "X_test and y_test have different lengths"
assert list(X_test.columns) == feature_lists[KEY]["feature_order"], "X_test columns are not in the training order"
assert hasattr(trained_models["Logistic Regression"], "predict_proba"), "the trained models are missing"

# ---------- 2. Every model gives the chance that each task finishes late ----------
test_proba = {name: model.predict_proba(X_test)[:, 1] for name, model in trained_models.items()}

predictions = pd.DataFrame({
    "task_id": case_infos[KEY].loc[X_test.index, "task_id"].to_numpy(),
    "project_id": case_infos[KEY].loc[X_test.index, "project_id"].to_numpy(),
    "actual_delay_flag": y_test.to_numpy(),
    "actual_delay_days": y_second[KEY].loc[X_test.index].round(1).to_numpy(),
}, index=X_test.index)
for name, proba in test_proba.items():
    predictions[name] = proba.round(3)

# ---------- 3. Sanity check: real delays should get a clearly higher chance than the rest ----------
is_delayed = y_test.to_numpy() == 1
risk_check = pd.DataFrame({
    "avg chance, real delays": {n: p[is_delayed].mean() for n, p in test_proba.items()},
    "avg chance, on-time tasks": {n: p[~is_delayed].mean() for n, p in test_proba.items()},
})
risk_check["gap"] = risk_check.iloc[:, 0] - risk_check.iloc[:, 1]

print(f"{len(predictions):,} test tasks scored by {len(test_proba)} models | "
      f"real delays: {int(is_delayed.sum())} ({is_delayed.mean():.1%}) | "
      f"{predictions['project_id'].nunique()} test projects the models have never seen\n")
print("A useful model gives real delays a clearly higher average chance (a bigger gap is better):")
display(risk_check.round(3))
print("\nFirst 10 test tasks (each model column = chance of a delay):")
display(predictions.head(10))
```

#### Objective 3: Site safety

```python
# ============================================================
# PREDICT X_TEST - BUSINESS OBJECTIVE 3: PREVENT SERIOUS SAFETY INCIDENTS
# ============================================================

# %%
KEY = "site_safety"

X_test = scaled[KEY]["X_test"]
y_test = splits[KEY]["y_test"]
trained_models = models_obj3["trained"]
baseline_name = "Baseline (always predicts the common answer)"

# ---------- 1. Pre-flight checks ----------
assert len(X_test) == len(y_test), "X_test and y_test have different lengths"
assert list(X_test.columns) == feature_lists[KEY]["feature_order"], "X_test columns are not in the training order"
assert hasattr(trained_models["Logistic Regression"], "predict_proba"), "the trained models are missing"

# ---------- 2. Every model gives the chance that each incident is serious ----------
test_proba = {name: model.predict_proba(X_test)[:, 1] for name, model in trained_models.items()}

predictions = pd.DataFrame({
    "incident_id": case_infos[KEY].loc[X_test.index, "incident_id"].to_numpy(),
    "project_id": case_infos[KEY].loc[X_test.index, "project_id"].to_numpy(),
    "actual_serious_flag": y_test.to_numpy(),
}, index=X_test.index)
for name, proba in test_proba.items():
    predictions[name] = proba.round(3)

# ---------- 3. Sanity check: serious incidents should get a clearly higher chance than the rest ----------
is_serious = y_test.to_numpy() == 1
risk_check = pd.DataFrame({
    "avg chance, serious incidents": {n: p[is_serious].mean() for n, p in test_proba.items()},
    "avg chance, other incidents": {n: p[~is_serious].mean() for n, p in test_proba.items()},
})
risk_check["gap"] = risk_check.iloc[:, 0] - risk_check.iloc[:, 1]

print(f"{len(predictions):,} test incidents scored by {len(test_proba)} models | "
      f"serious incidents: {int(is_serious.sum())} ({is_serious.mean():.1%}) | "
      f"{predictions['project_id'].nunique()} test projects the models have never seen\n")
print("A useful model gives serious incidents a clearly higher average chance (a bigger gap is better):")
display(risk_check.round(3))
print("\nFirst 10 test incidents (each model column = chance of being serious):")
display(predictions.head(10))
```

#### Objective 4: Late material delivery

```python
# ============================================================
# PREDICT X_TEST - BUSINESS OBJECTIVE 4: PREVENT LATE MATERIAL DELIVERIES
# ============================================================

# %%
KEY = "late_delivery"

X_test = scaled[KEY]["X_test"]
y_test = splits[KEY]["y_test"]
trained_models = models_obj4["trained"]
baseline_name = "Baseline (always predicts the common answer)"

# ---------- 1. Pre-flight checks ----------
assert len(X_test) == len(y_test), "X_test and y_test have different lengths"
assert list(X_test.columns) == feature_lists[KEY]["feature_order"], "X_test columns are not in the training order"
assert hasattr(trained_models["Logistic Regression"], "predict_proba"), "the trained models are missing"

# ---------- 2. Every model gives the chance that each order arrives late ----------
test_proba = {name: model.predict_proba(X_test)[:, 1] for name, model in trained_models.items()}

predictions = pd.DataFrame({
    "order_id": case_infos[KEY].loc[X_test.index, "order_id"].to_numpy(),
    "project_id": case_infos[KEY].loc[X_test.index, "project_id"].to_numpy(),
    "actual_late_flag": y_test.to_numpy(),
    "actual_delay_days": y_second[KEY].loc[X_test.index].round(1).to_numpy(),
}, index=X_test.index)
for name, proba in test_proba.items():
    predictions[name] = proba.round(3)

# ---------- 3. Sanity check: late orders should get a clearly higher chance than the rest ----------
is_late = y_test.to_numpy() == 1
risk_check = pd.DataFrame({
    "avg chance, late orders": {n: p[is_late].mean() for n, p in test_proba.items()},
    "avg chance, on-time orders": {n: p[~is_late].mean() for n, p in test_proba.items()},
})
risk_check["gap"] = risk_check.iloc[:, 0] - risk_check.iloc[:, 1]

print(f"{len(predictions):,} test orders scored by {len(test_proba)} models | "
      f"late orders: {int(is_late.sum())} ({is_late.mean():.1%}) | "
      f"{predictions['project_id'].nunique()} test projects the models have never seen\n")
print("A useful model gives late orders a clearly higher average chance (a bigger gap is better):")
display(risk_check.round(3))
print("\nFirst 10 test orders (each model column = chance of being late):")
display(predictions.head(10))
```

#### Objective 5: Price and invoice anomalies

```python
# ============================================================
# PREDICT (SCORE ALL ORDERS) - BUSINESS OBJECTIVE 5: SPOT PRICE AND INVOICE ANOMALIES
# ============================================================

# %%
from pathlib import Path

KEY = "price_anomalies"
OUT_DIR = Path("construction_charts/objective_5_price_anomalies")
OUT_DIR.mkdir(parents=True, exist_ok=True)

model = models_obj5["model"]
X_all = scaled[KEY]["X_train"]               # all 3,000 orders, scaled
X_raw = splits[KEY]["X_train"]               # same orders in real units
case_info = case_infos[KEY]

# ---------- 1. Pre-flight checks (there is no test set for an unsupervised task) ----------
assert list(X_all.columns) == feature_lists[KEY]["feature_order"], "columns are not in the training order"
assert list(X_all.columns) == models_obj5["feature_order"], "columns differ from the trained model"
assert len(X_all) == len(case_info) == 3000, "order count changed"
assert not any(c in X_all.columns for c in ["is_anomaly", "total_vs_calculated_pct"]), "answer-key column found in X"

# ---------- 2. Score every order ----------
score = -model.score_samples(X_all)                       # bigger = more unusual
flagged = model.predict(X_all) == -1
assert np.allclose(score, models_obj5["order_scores"]["anomaly_score"].to_numpy()), "scores differ from the training step"

predictions = pd.DataFrame({
    "order_id": case_info["order_id"].to_numpy(),
    "project_id": case_info["project_id"].to_numpy(),
    "anomaly_score": score.round(4),
    "flagged": flagged,
}, index=X_all.index)
predictions["rank"] = predictions["anomaly_score"].rank(ascending=False, method="first").astype(int)
predictions["risk_band"] = pd.cut(predictions["rank"] / len(predictions), [0, 0.01, 0.05, 0.20, 1.0],
                                  labels=["Top 1%", "Next 4% (flagged)", "Next 15% (watch list)", "Normal"])

# ---------- 3. Plain-language reason for each flag ----------
ratio_material, ratio_peer = X_raw["price_ratio_material"], X_raw["price_ratio_peer"]
quantity_z = X_all["log_quantity"]
reasons = pd.Series("", index=X_all.index)
reasons = reasons.mask(ratio_material > 1.15, "priced above the material norm")
reasons = reasons.mask(ratio_material < 0.87, "priced below the material norm")
reasons = reasons.mask((reasons == "") & (ratio_peer > 1.15), "priced above similar orders")
reasons = reasons.mask((reasons == "") & (ratio_peer < 0.87), "priced below similar orders")
reasons = reasons.mask((reasons == "") & (quantity_z.abs() > 2), "unusual order quantity")
reasons = reasons.mask(reasons == "", "unusual mix of price and quantity")
predictions["main_reason"] = reasons
predictions["price_ratio_material"] = ratio_material.round(2)
predictions["price_ratio_peer"] = ratio_peer.round(2)
predictions["unit_price_usd"] = np.expm1(X_raw["log_unit_price"]).round(2)      # undo the log1p used in the ABT

review_queue = predictions[predictions["flagged"]].sort_values("rank")
predictions.sort_values("rank").to_csv(OUT_DIR / "all_orders_scored.csv", index=False)
review_queue.to_csv(OUT_DIR / "review_queue.csv", index=False)

# ---------- Verification ----------
print(f"{len(predictions):,} orders scored | {int(flagged.sum())} flagged for review ({flagged.mean():.1%}) "
      f"| {predictions['project_id'].nunique()} projects")
print(f"Flag cut-off (anomaly score): {models_obj5['cutoff_score']:.3f}\n")

print("Orders by risk band:")
display(predictions["risk_band"].value_counts().reindex(["Top 1%", "Next 4% (flagged)", "Next 15% (watch list)", "Normal"]).to_frame("orders"))

print("\nWhy the flagged orders were flagged:")
display(review_queue["main_reason"].value_counts().to_frame("flagged_orders"))

print("\nProjects with the most flagged orders:")
by_project = review_queue.groupby("project_id").size().sort_values(ascending=False).head(8)
display(by_project.to_frame("flagged_orders").assign(share_of_flags=lambda d: (d["flagged_orders"] / len(review_queue)).round(3)))

print("\nTop 10 orders to review first:")
display(review_queue.head(10)[["order_id", "project_id", "anomaly_score", "price_ratio_material", "price_ratio_peer", "unit_price_usd", "main_reason"]])
print(f"\nSaved all_orders_scored.csv and review_queue.csv to {OUT_DIR}/. The answer key has still not been opened.")
```

---

## 16. Compare X and y

Which inputs drive each outcome? Permutation importance is calculated for each model, and an input counts as "agreed" when it ranks in the top 8 of at least 2 of the 3 models. Importance is the average drop in ROC AUC when the input is shuffled.

| Objective | Inputs the models agree on |
|---|---|
| 1 Cost overrun | Contract type, country, cost category, number of floors, planned project days, high-rise, manager experience |
| 2 Schedule delay | Crew size, rainy season, task type, planned task days, vendor reliability and safety ratings |
| 3 Serious incident | Incident type, worker experience, PPE compliance, shift, work at height, vendor safety rating |
| 4 Late delivery | Quantity ordered, vendor safety rating, international supplier, vendor years active. The importances are small, which matches the weak score |
| 5 Price anomalies | Price against the material norm and against similar orders drive the anomaly scores |

These drivers match what the EDA found, which is a useful check that the models learned real patterns and not noise.

### Charts

<p align="center">
  <img width="2980" height="1107" alt="objective_1_compare_x_and_y" src="https://github.com/user-attachments/assets/e6c2e3e5-d01d-46fa-8059-53803c125b80" />
</p>
<p align="center"><em>Objective 1: Cost overruns</em></p>

<p align="center">
  <img width="2980" height="1107" alt="objective_2_compare_x_and_y" src="https://github.com/user-attachments/assets/3954b4dc-bb49-4116-a38c-2945dd2da833" />
</p>
<p align="center"><em>Objective 2: Schedule delays</em></p>

<p align="center">
  <img width="2980" height="1107" alt="objective_3_compare_x_and_y" src="https://github.com/user-attachments/assets/7f189b84-b9fa-4427-8452-a168c7e02047" />
</p>
<p align="center"><em>Objective 3: Site safety</em></p>

<p align="center">
  <img width="2979" height="1107" alt="objective_4_compare_x_and_y" src="https://github.com/user-attachments/assets/6e79e96b-d2c4-494a-a7cb-321dcb172107" />
</p>
<p align="center"><em>Objective 4: Late material delivery</em></p>

<p align="center">
  <img width="3580" height="1107" alt="objective_5_compare_x_and_y" src="https://github.com/user-attachments/assets/8a262f67-0803-4be9-a707-002e7e55bd3e" />
</p>
<p align="center"><em>Objective 5: Price and invoice anomalies</em></p>

### Code: compare X and y

#### Objective 1: Cost overruns

```python
# ============================================================
# COMPARE X AND y - BUSINESS OBJECTIVE 1: CONTROL PROJECT COST OVERRUNS
# ============================================================

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

KEY = "cost_overrun"
CHART_DIR = Path("construction_charts/objective_1_cost_overrun")
CHART_DIR.mkdir(parents=True, exist_ok=True)
N_REPEATS, TOP_N, MIN_AGREEMENT = 10, 8, 2
BLUE, ORANGE, INK, SOFT, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

X_train_raw, y_train = splits[KEY]["X_train"], splits[KEY]["y_train"]     # real units, for the data view
X_train_s, X_test_s = scaled[KEY]["X_train"], scaled[KEY]["X_test"]       # scaled, what the models see
y_test = splits[KEY]["y_test"]
trained_models = models_obj1["trained"]
real_models = [name for name in trained_models if not name.startswith("Baseline")]

# one "input" = one numeric column, or all the 0/1 columns made from one text column
feature_groups = {column: [column] for column in OBJECTIVES[KEY]["numeric"]}
for column in OBJECTIVES[KEY]["categorical"]:
    feature_groups[column] = [c for c in X_test_s.columns if c.startswith(f"{column}_")]

# ---------- A. What moves with overruns in the data (correlation, training rows) ----------
correlations = X_train_s.corrwith(y_train.astype(float))
top_corr = correlations.reindex(correlations.abs().nlargest(12).index).sort_values()

# ---------- B. What each model relies on (shuffle one input, measure how much ROC AUC drops) ----------
def grouped_permutation_importance(model, X, y, groups, repeats, seed=42):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    result = {}
    for name, columns in groups.items():
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[columns] = X[columns].to_numpy()[rng.permutation(len(X))]   # same shuffle for the whole group
            drops.append(base - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
        result[name] = np.mean(drops)
    return pd.Series(result)

importance_by_model = pd.DataFrame({
    name: grouped_permutation_importance(trained_models[name], X_test_s, y_test, feature_groups, N_REPEATS)
    for name in real_models})
importance_by_model["average"] = importance_by_model.mean(axis=1)
importance_by_model = importance_by_model.sort_values("average", ascending=False)

# inputs that sit in the top N of at least MIN_AGREEMENT models
top_sets = {name: set(importance_by_model[name].nlargest(TOP_N).index) for name in real_models}
agreement = pd.Series({feature: sum(feature in s for s in top_sets.values()) for feature in importance_by_model.index})
agreed_features = importance_by_model.loc[agreement[agreement >= MIN_AGREEMENT].index].sort_values("average", ascending=False)

# ---------- C. What the data says about the main drivers ----------
overall_rate = 100 * y_train.mean()
data_view = []
for feature in importance_by_model.index[:10]:
    columns = feature_groups[feature]
    if len(columns) == 1 and X_train_raw[feature].nunique() > 2:
        low_cut, high_cut = X_train_raw[feature].quantile([0.25, 0.75])
        low = y_train[X_train_raw[feature] <= low_cut].mean() * 100
        high = y_train[X_train_raw[feature] >= high_cut].mean() * 100
        data_view.append({"input": feature, "compared": "lowest quarter vs highest quarter of values",
                          "rate_at_low_pct": round(low, 1), "rate_at_high_pct": round(high, 1)})
    elif len(columns) == 1:
        low = y_train[X_train_raw[feature] == 0].mean() * 100
        high = y_train[X_train_raw[feature] == 1].mean() * 100
        data_view.append({"input": feature, "compared": "value 0 vs value 1",
                          "rate_at_low_pct": round(low, 1), "rate_at_high_pct": round(high, 1)})
    else:
        rates = {c.replace(f"{feature}_", ""): y_train[X_train_raw[c] == 1].mean() * 100 for c in columns}
        worst, best = max(rates, key=rates.get), min(rates, key=rates.get)
        data_view.append({"input": feature, "compared": f"{best} vs {worst}",
                          "rate_at_low_pct": round(rates[best], 1), "rate_at_high_pct": round(rates[worst], 1)})
data_view = pd.DataFrame(data_view)
data_view["gap_points"] = (data_view["rate_at_high_pct"] - data_view["rate_at_low_pct"]).round(1)

# ---------- D. Chart ----------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": GRID,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": SOFT,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

top_imp = importance_by_model["average"].head(10).sort_values()
bars = axes[0].barh(top_imp.index, top_imp.values, color=BLUE, height=0.6)
for bar, value in zip(bars, top_imp.values):
    axes[0].text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9, color=INK)
axes[0].set_xlabel("Average drop in ROC AUC when the input is shuffled (bigger = more important)")
axes[0].set_title("What the models rely on")
axes[0].grid(axis="y", visible=False)

colors = [ORANGE if value > 0 else BLUE for value in top_corr.values]
bars = axes[1].barh(top_corr.index, top_corr.values, color=colors, height=0.6)
for bar, value in zip(bars, top_corr.values):
    side = 0.006 if value > 0 else -0.006
    axes[1].text(value + side, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}", va="center",
                 ha="left" if value > 0 else "right", fontsize=9, color=INK)
axes[1].axvline(0, color=SOFT, linewidth=0.8)
axes[1].set_xlim(top_corr.abs().max() * -1.25, top_corr.abs().max() * 1.25)
axes[1].set_xlabel("Correlation with overrun (orange = more overrun, blue = less)")
axes[1].set_title("What moves with overruns in the data")
axes[1].grid(axis="y", visible=False)

fig.suptitle("Objective 1: which inputs drive cost overruns", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(CHART_DIR / "compare_x_and_y.png", dpi=200, bbox_inches="tight")
plt.show()

importance_by_model.round(4).to_csv(CHART_DIR / "input_importance_by_model.csv")

# ---------- Verification ----------
print(f"Overall overrun rate in training: {overall_rate:.1f}%\n")
print(f"Inputs in the top {TOP_N} of at least {MIN_AGREEMENT} of {len(real_models)} models:")
display(agreed_features.round(4))
print("\nWhat the data says about the ten most important inputs:")
display(data_view)
print("\nImportance of every input, by model (average drop in ROC AUC):")
display(importance_by_model.round(4))
```

#### Objective 2: Schedule delays

```python
# ============================================================
# COMPARE X AND y - BUSINESS OBJECTIVE 2: REDUCE SCHEDULE DELAYS
# ============================================================

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

KEY = "schedule_delay"
CHART_DIR = Path("construction_charts/objective_2_schedule_delay")
CHART_DIR.mkdir(parents=True, exist_ok=True)
N_REPEATS, TOP_N, MIN_AGREEMENT = 10, 8, 2
BLUE, ORANGE, INK, SOFT, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

X_train_raw, y_train = splits[KEY]["X_train"], splits[KEY]["y_train"]     # real units, for the data view
X_train_s, X_test_s = scaled[KEY]["X_train"], scaled[KEY]["X_test"]       # scaled, what the models see
y_test = splits[KEY]["y_test"]
trained_models = models_obj2["trained"]
real_models = [name for name in trained_models if not name.startswith("Baseline")]

# one "input" = one numeric column, or all the 0/1 columns made from one text column
feature_groups = {column: [column] for column in OBJECTIVES[KEY]["numeric"]}
for column in OBJECTIVES[KEY]["categorical"]:
    feature_groups[column] = [c for c in X_test_s.columns if c.startswith(f"{column}_")]

# ---------- A. What moves with delays in the data (correlation, training rows) ----------
correlations = X_train_s.corrwith(y_train.astype(float))
top_corr = correlations.reindex(correlations.abs().nlargest(12).index).sort_values()

# ---------- B. What each model relies on (shuffle one input, measure how much ROC AUC drops) ----------
def grouped_permutation_importance(model, X, y, groups, repeats, seed=42):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    result = {}
    for name, columns in groups.items():
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[columns] = X[columns].to_numpy()[rng.permutation(len(X))]   # same shuffle for the whole group
            drops.append(base - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
        result[name] = np.mean(drops)
    return pd.Series(result)

importance_by_model = pd.DataFrame({
    name: grouped_permutation_importance(trained_models[name], X_test_s, y_test, feature_groups, N_REPEATS)
    for name in real_models})
importance_by_model["average"] = importance_by_model.mean(axis=1)
importance_by_model = importance_by_model.sort_values("average", ascending=False)

# inputs that sit in the top N of at least MIN_AGREEMENT models
top_sets = {name: set(importance_by_model[name].nlargest(TOP_N).index) for name in real_models}
agreement = pd.Series({feature: sum(feature in s for s in top_sets.values()) for feature in importance_by_model.index})
agreed_features = importance_by_model.loc[agreement[agreement >= MIN_AGREEMENT].index].sort_values("average", ascending=False)

# ---------- C. What the data says about the main drivers ----------
overall_rate = 100 * y_train.mean()
data_view = []
for feature in importance_by_model.index[:10]:
    columns = feature_groups[feature]
    if len(columns) == 1 and X_train_raw[feature].nunique() > 2:
        low_cut, high_cut = X_train_raw[feature].quantile([0.25, 0.75])
        low = y_train[X_train_raw[feature] <= low_cut].mean() * 100
        high = y_train[X_train_raw[feature] >= high_cut].mean() * 100
        data_view.append({"input": feature, "compared": "lowest quarter vs highest quarter of values",
                          "rate_at_low_pct": round(low, 1), "rate_at_high_pct": round(high, 1)})
    elif len(columns) == 1:
        low = y_train[X_train_raw[feature] == 0].mean() * 100
        high = y_train[X_train_raw[feature] == 1].mean() * 100
        data_view.append({"input": feature, "compared": "value 0 vs value 1",
                          "rate_at_low_pct": round(low, 1), "rate_at_high_pct": round(high, 1)})
    else:
        rates = {c.replace(f"{feature}_", ""): y_train[X_train_raw[c] == 1].mean() * 100 for c in columns}
        worst, best = max(rates, key=rates.get), min(rates, key=rates.get)
        data_view.append({"input": feature, "compared": f"{best} vs {worst}",
                          "rate_at_low_pct": round(rates[best], 1), "rate_at_high_pct": round(rates[worst], 1)})
data_view = pd.DataFrame(data_view)
data_view["gap_points"] = (data_view["rate_at_high_pct"] - data_view["rate_at_low_pct"]).round(1)

# ---------- D. Chart ----------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": GRID,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": SOFT,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

top_imp = importance_by_model["average"].head(10).sort_values()
bars = axes[0].barh(top_imp.index, top_imp.values, color=BLUE, height=0.6)
for bar, value in zip(bars, top_imp.values):
    axes[0].text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9, color=INK)
axes[0].set_xlabel("Average drop in ROC AUC when the input is shuffled (bigger = more important)")
axes[0].set_title("What the models rely on")
axes[0].grid(axis="y", visible=False)

colors = [ORANGE if value > 0 else BLUE for value in top_corr.values]
bars = axes[1].barh(top_corr.index, top_corr.values, color=colors, height=0.6)
for bar, value in zip(bars, top_corr.values):
    side = 0.006 if value > 0 else -0.006
    axes[1].text(value + side, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}", va="center",
                 ha="left" if value > 0 else "right", fontsize=9, color=INK)
axes[1].axvline(0, color=SOFT, linewidth=0.8)
axes[1].set_xlim(top_corr.abs().max() * -1.25, top_corr.abs().max() * 1.25)
axes[1].set_xlabel("Correlation with delay (orange = more delay, blue = less)")
axes[1].set_title("What moves with delays in the data")
axes[1].grid(axis="y", visible=False)

fig.suptitle("Objective 2: which inputs drive schedule delays", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(CHART_DIR / "compare_x_and_y.png", dpi=200, bbox_inches="tight")
plt.show()

importance_by_model.round(4).to_csv(CHART_DIR / "input_importance_by_model.csv")

# ---------- Verification ----------
print(f"Overall delay rate in training: {overall_rate:.1f}%\n")
print(f"Inputs in the top {TOP_N} of at least {MIN_AGREEMENT} of {len(real_models)} models:")
display(agreed_features.round(4))
print("\nWhat the data says about the ten most important inputs:")
display(data_view)
print("\nImportance of every input, by model (average drop in ROC AUC):")
display(importance_by_model.round(4))
```

#### Objective 3: Site safety

```python
# ============================================================
# COMPARE X AND y - BUSINESS OBJECTIVE 3: PREVENT SERIOUS SAFETY INCIDENTS
# ============================================================

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

KEY = "site_safety"
CHART_DIR = Path("construction_charts/objective_3_site_safety")
CHART_DIR.mkdir(parents=True, exist_ok=True)
N_REPEATS, TOP_N, MIN_AGREEMENT = 10, 8, 2
BLUE, ORANGE, INK, SOFT, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

X_train_raw, y_train = splits[KEY]["X_train"], splits[KEY]["y_train"]     # real units, for the data view
X_train_s, X_test_s = scaled[KEY]["X_train"], scaled[KEY]["X_test"]       # scaled, what the models see
y_test = splits[KEY]["y_test"]
trained_models = models_obj3["trained"]
real_models = [name for name in trained_models if not name.startswith("Baseline")]

# one "input" = one numeric column, or all the 0/1 columns made from one text column
feature_groups = {column: [column] for column in OBJECTIVES[KEY]["numeric"]}
for column in OBJECTIVES[KEY]["categorical"]:
    feature_groups[column] = [c for c in X_test_s.columns if c.startswith(f"{column}_")]

# ---------- A. What moves with serious incidents in the data (correlation, training rows) ----------
correlations = X_train_s.corrwith(y_train.astype(float))
top_corr = correlations.reindex(correlations.abs().nlargest(12).index).sort_values()

# ---------- B. What each model relies on (shuffle one input, measure how much ROC AUC drops) ----------
def grouped_permutation_importance(model, X, y, groups, repeats, seed=42):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    result = {}
    for name, columns in groups.items():
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[columns] = X[columns].to_numpy()[rng.permutation(len(X))]   # same shuffle for the whole group
            drops.append(base - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
        result[name] = np.mean(drops)
    return pd.Series(result)

importance_by_model = pd.DataFrame({
    name: grouped_permutation_importance(trained_models[name], X_test_s, y_test, feature_groups, N_REPEATS)
    for name in real_models})
importance_by_model["average"] = importance_by_model.mean(axis=1)
importance_by_model = importance_by_model.sort_values("average", ascending=False)

# inputs that sit in the top N of at least MIN_AGREEMENT models
top_sets = {name: set(importance_by_model[name].nlargest(TOP_N).index) for name in real_models}
agreement = pd.Series({feature: sum(feature in s for s in top_sets.values()) for feature in importance_by_model.index})
agreed_features = importance_by_model.loc[agreement[agreement >= MIN_AGREEMENT].index].sort_values("average", ascending=False)

# ---------- C. What the data says about the main drivers ----------
overall_rate = 100 * y_train.mean()
data_view = []
for feature in importance_by_model.index[:10]:
    columns = feature_groups[feature]
    if len(columns) == 1 and X_train_raw[feature].nunique() > 2:
        low_cut, high_cut = X_train_raw[feature].quantile([0.25, 0.75])
        low = y_train[X_train_raw[feature] <= low_cut].mean() * 100
        high = y_train[X_train_raw[feature] >= high_cut].mean() * 100
        data_view.append({"input": feature, "compared": "lowest quarter vs highest quarter of values",
                          "rate_at_low_pct": round(low, 1), "rate_at_high_pct": round(high, 1)})
    elif len(columns) == 1:
        low = y_train[X_train_raw[feature] == 0].mean() * 100
        high = y_train[X_train_raw[feature] == 1].mean() * 100
        data_view.append({"input": feature, "compared": "value 0 vs value 1",
                          "rate_at_low_pct": round(low, 1), "rate_at_high_pct": round(high, 1)})
    else:
        rates = {c.replace(f"{feature}_", ""): y_train[X_train_raw[c] == 1].mean() * 100 for c in columns}
        worst, best = max(rates, key=rates.get), min(rates, key=rates.get)
        data_view.append({"input": feature, "compared": f"{best} vs {worst}",
                          "rate_at_low_pct": round(rates[best], 1), "rate_at_high_pct": round(rates[worst], 1)})
data_view = pd.DataFrame(data_view)
data_view["gap_points"] = (data_view["rate_at_high_pct"] - data_view["rate_at_low_pct"]).round(1)

# ---------- D. Chart ----------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": GRID,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": SOFT,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

top_imp = importance_by_model["average"].head(10).sort_values()
bars = axes[0].barh(top_imp.index, top_imp.values, color=BLUE, height=0.6)
for bar, value in zip(bars, top_imp.values):
    axes[0].text(value + 0.002, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9, color=INK)
axes[0].set_xlabel("Average drop in ROC AUC when the input is shuffled (bigger = more important)")
axes[0].set_title("What the models rely on")
axes[0].grid(axis="y", visible=False)

colors = [ORANGE if value > 0 else BLUE for value in top_corr.values]
bars = axes[1].barh(top_corr.index, top_corr.values, color=colors, height=0.6)
for bar, value in zip(bars, top_corr.values):
    side = 0.006 if value > 0 else -0.006
    axes[1].text(value + side, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}", va="center",
                 ha="left" if value > 0 else "right", fontsize=9, color=INK)
axes[1].axvline(0, color=SOFT, linewidth=0.8)
axes[1].set_xlim(top_corr.abs().max() * -1.25, top_corr.abs().max() * 1.25)
axes[1].set_xlabel("Correlation with serious incident (orange = more serious, blue = less)")
axes[1].set_title("What moves with serious incidents in the data")
axes[1].grid(axis="y", visible=False)

fig.suptitle("Objective 3: which inputs drive serious safety incidents", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(CHART_DIR / "compare_x_and_y.png", dpi=200, bbox_inches="tight")
plt.show()

importance_by_model.round(4).to_csv(CHART_DIR / "input_importance_by_model.csv")

# ---------- Verification ----------
print(f"Share of serious incidents in training: {overall_rate:.1f}%\n")
print(f"Inputs in the top {TOP_N} of at least {MIN_AGREEMENT} of {len(real_models)} models:")
display(agreed_features.round(4))
print("\nWhat the data says about the ten most important inputs:")
display(data_view)
print("\nImportance of every input, by model (average drop in ROC AUC):")
display(importance_by_model.round(4))
```

#### Objective 4: Late material delivery

```python
# ============================================================
# COMPARE X AND y - BUSINESS OBJECTIVE 4: PREVENT LATE MATERIAL DELIVERIES
# ============================================================

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

KEY = "late_delivery"
CHART_DIR = Path("construction_charts/objective_4_late_delivery")
CHART_DIR.mkdir(parents=True, exist_ok=True)
N_REPEATS, TOP_N, MIN_AGREEMENT = 10, 8, 2
BLUE, ORANGE, INK, SOFT, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

X_train_raw, y_train = splits[KEY]["X_train"], splits[KEY]["y_train"]     # real units, for the data view
X_train_s, X_test_s = scaled[KEY]["X_train"], scaled[KEY]["X_test"]       # scaled, what the models see
y_test = splits[KEY]["y_test"]
trained_models = models_obj4["trained"]
real_models = [name for name in trained_models if not name.startswith("Baseline")]

# one "input" = one numeric column, or all the 0/1 columns made from one text column
feature_groups = {column: [column] for column in OBJECTIVES[KEY]["numeric"]}
for column in OBJECTIVES[KEY]["categorical"]:
    feature_groups[column] = [c for c in X_test_s.columns if c.startswith(f"{column}_")]

# ---------- A. What moves with late deliveries in the data (correlation, training rows) ----------
correlations = X_train_s.corrwith(y_train.astype(float))
top_corr = correlations.reindex(correlations.abs().nlargest(12).index).sort_values()

# ---------- B. What each model relies on (shuffle one input, measure how much ROC AUC drops) ----------
def grouped_permutation_importance(model, X, y, groups, repeats, seed=42):
    rng = np.random.default_rng(seed)
    base = roc_auc_score(y, model.predict_proba(X)[:, 1])
    result = {}
    for name, columns in groups.items():
        drops = []
        for _ in range(repeats):
            shuffled = X.copy()
            shuffled[columns] = X[columns].to_numpy()[rng.permutation(len(X))]   # same shuffle for the whole group
            drops.append(base - roc_auc_score(y, model.predict_proba(shuffled)[:, 1]))
        result[name] = np.mean(drops)
    return pd.Series(result)

importance_by_model = pd.DataFrame({
    name: grouped_permutation_importance(trained_models[name], X_test_s, y_test, feature_groups, N_REPEATS)
    for name in real_models})
importance_by_model["average"] = importance_by_model.mean(axis=1)
importance_by_model = importance_by_model.sort_values("average", ascending=False)

# inputs that sit in the top N of at least MIN_AGREEMENT models
top_sets = {name: set(importance_by_model[name].nlargest(TOP_N).index) for name in real_models}
agreement = pd.Series({feature: sum(feature in s for s in top_sets.values()) for feature in importance_by_model.index})
agreed_features = importance_by_model.loc[agreement[agreement >= MIN_AGREEMENT].index].sort_values("average", ascending=False)

# ---------- C. What the data says about the main drivers ----------
overall_rate = 100 * y_train.mean()
data_view = []
for feature in importance_by_model.index[:10]:
    columns = feature_groups[feature]
    if len(columns) == 1 and X_train_raw[feature].nunique() > 2:
        low_cut, high_cut = X_train_raw[feature].quantile([0.25, 0.75])
        low = y_train[X_train_raw[feature] <= low_cut].mean() * 100
        high = y_train[X_train_raw[feature] >= high_cut].mean() * 100
        data_view.append({"input": feature, "compared": "lowest quarter vs highest quarter of values",
                          "rate_at_low_pct": round(low, 1), "rate_at_high_pct": round(high, 1)})
    elif len(columns) == 1:
        low = y_train[X_train_raw[feature] == 0].mean() * 100
        high = y_train[X_train_raw[feature] == 1].mean() * 100
        data_view.append({"input": feature, "compared": "value 0 vs value 1",
                          "rate_at_low_pct": round(low, 1), "rate_at_high_pct": round(high, 1)})
    else:
        rates = {c.replace(f"{feature}_", ""): y_train[X_train_raw[c] == 1].mean() * 100 for c in columns}
        worst, best = max(rates, key=rates.get), min(rates, key=rates.get)
        data_view.append({"input": feature, "compared": f"{best} vs {worst}",
                          "rate_at_low_pct": round(rates[best], 1), "rate_at_high_pct": round(rates[worst], 1)})
data_view = pd.DataFrame(data_view)
data_view["gap_points"] = (data_view["rate_at_high_pct"] - data_view["rate_at_low_pct"]).round(1)

# ---------- D. Chart ----------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": GRID,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": SOFT,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))

top_imp = importance_by_model["average"].head(10).sort_values()
bars = axes[0].barh(top_imp.index, top_imp.values, color=BLUE, height=0.6)
for bar, value in zip(bars, top_imp.values):
    axes[0].text(value + 0.012 * top_imp.max(), bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9, color=INK)
axes[0].set_xlabel("Average drop in ROC AUC when the input is shuffled (bigger = more important)")
axes[0].set_title("What the models rely on")
axes[0].grid(axis="y", visible=False)

colors = [ORANGE if value > 0 else BLUE for value in top_corr.values]
bars = axes[1].barh(top_corr.index, top_corr.values, color=colors, height=0.6)
for bar, value in zip(bars, top_corr.values):
    side = 0.006 if value > 0 else -0.006
    axes[1].text(value + side, bar.get_y() + bar.get_height() / 2, f"{value:+.2f}", va="center",
                 ha="left" if value > 0 else "right", fontsize=9, color=INK)
axes[1].axvline(0, color=SOFT, linewidth=0.8)
axes[1].set_xlim(top_corr.abs().max() * -1.25, top_corr.abs().max() * 1.25)
axes[1].set_xlabel("Correlation with late delivery (orange = more late, blue = less)")
axes[1].set_title("What moves with late deliveries in the data")
axes[1].grid(axis="y", visible=False)

fig.suptitle("Objective 4: which inputs drive late material deliveries", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(CHART_DIR / "compare_x_and_y.png", dpi=200, bbox_inches="tight")
plt.show()

importance_by_model.round(4).to_csv(CHART_DIR / "input_importance_by_model.csv")

# ---------- Verification ----------
print(f"Share of late deliveries in training: {overall_rate:.1f}%\n")
print(f"Inputs in the top {TOP_N} of at least {MIN_AGREEMENT} of {len(real_models)} models:")
display(agreed_features.round(4))
print("\nWhat the data says about the ten most important inputs:")
display(data_view)
print("\nImportance of every input, by model (average drop in ROC AUC):")
display(importance_by_model.round(4))
```

#### Objective 5: Price and invoice anomalies

```python
# ============================================================
# COMPARE X AND y - BUSINESS OBJECTIVE 5: SPOT PRICE AND INVOICE ANOMALIES
# (there is no y here, so we compare flagged orders with normal orders)
# ============================================================

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator

KEY = "price_anomalies"
CHART_DIR = Path("construction_charts/objective_5_price_anomalies")
CHART_DIR.mkdir(parents=True, exist_ok=True)
N_REPEATS = 10
BLUE, ORANGE, INK, SOFT, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

model = models_obj5["model"]
X_all, X_raw = scaled[KEY]["X_train"], splits[KEY]["X_train"]
flagged = predictions["flagged"].to_numpy()                 # from the previous step
features = list(X_all.columns)

# ---------- A. How different are flagged orders from normal ones? (in scaled units, so features are comparable) ----------
difference = pd.DataFrame({
    "normal_median_real": X_raw[features].loc[~flagged].median(),
    "flagged_median_real": X_raw[features].loc[flagged].median(),
    "gap_in_std_units": X_all[features].loc[flagged].mean() - X_all[features].loc[~flagged].mean(),
}).round(3)
difference["abs_gap"] = difference["gap_in_std_units"].abs()
difference = difference.sort_values("abs_gap", ascending=False).drop(columns="abs_gap")

# ---------- B. What the model relies on: shuffle one feature, see how many original flags disappear ----------
def flags_lost_when_shuffled(feature, repeats, seed=42):
    rng = np.random.default_rng(seed)
    lost = []
    for _ in range(repeats):
        shuffled = X_all.copy()
        shuffled[feature] = X_all[feature].to_numpy()[rng.permutation(len(X_all))]
        scores = -model.score_samples(shuffled)
        new_flags = scores >= np.quantile(scores, 1 - flagged.mean())      # same review budget
        lost.append(1 - (new_flags & flagged).sum() / flagged.sum())
    return np.mean(lost)

importance = pd.Series({f: flags_lost_when_shuffled(f, N_REPEATS) for f in features}).sort_values(ascending=False)

# ---------- C. Business view: how often is an order flagged at each price level? ----------
def flag_rate_table(column, edges, labels):
    band = pd.cut(X_raw[column], edges, labels=labels)
    table = pd.DataFrame({"band": band, "flagged": flagged}).groupby("band", observed=True)["flagged"].agg(
        orders="size", flagged="sum", flag_rate="mean")
    table["flag_rate"] = (100 * table["flag_rate"]).round(1)
    return table

edges = [0, 0.87, 1.15, 2, 100]
labels = ["below 0.87 (cheap)", "0.87 to 1.15 (normal)", "1.15 to 2 (dear)", "above 2 (very dear)"]
rate_material = flag_rate_table("price_ratio_material", edges, labels)
rate_peer = flag_rate_table("price_ratio_peer", edges, labels)

# ---------- D. Chart ----------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": GRID,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": SOFT,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), gridspec_kw={"width_ratios": [1, 1.25, 1]})

imp = importance.sort_values()
bars = axes[0].barh(imp.index, 100 * imp.values, color=BLUE, height=0.55)
for bar, value in zip(bars, 100 * imp.values):
    axes[0].text(value + 1, bar.get_y() + bar.get_height() / 2, f"{value:.0f}%", va="center", fontsize=9.5, color=INK)
axes[0].set_xlim(0, 100 * importance.max() * 1.25)
axes[0].set_xlabel("Share of the original flags that disappear when the feature is shuffled")
axes[0].set_title("What the model relies on"); axes[0].grid(axis="y", visible=False)

axes[1].scatter(X_raw.loc[~flagged, "price_ratio_material"], X_raw.loc[~flagged, "price_ratio_peer"],
                s=8, color=BLUE, alpha=0.35, label=f"normal ({(~flagged).sum():,})", linewidths=0)
axes[1].scatter(X_raw.loc[flagged, "price_ratio_material"], X_raw.loc[flagged, "price_ratio_peer"],
                s=22, color=ORANGE, alpha=0.9, label=f"flagged ({flagged.sum()})", linewidths=0)
axes[1].set_xscale("log"); axes[1].set_yscale("log")
ticks = [0.6, 0.8, 1, 1.5, 2, 3, 4, 6]
for axis in (axes[1].xaxis, axes[1].yaxis):
    axis.set_major_locator(FixedLocator(ticks)); axis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    axis.set_minor_locator(FixedLocator([])); axis.set_minor_formatter(FixedFormatter([]))
axes[1].axvline(1, color=SOFT, linewidth=0.8, linestyle="--"); axes[1].axhline(1, color=SOFT, linewidth=0.8, linestyle="--")
axes[1].set_xlabel("Price vs the material's usual price (1 = normal, log scale)")
axes[1].set_ylabel("Price vs similar orders (1 = normal, log scale)")
axes[1].set_title("Where flagged orders sit"); axes[1].legend(frameon=False, loc="upper left")

colors = [BLUE, BLUE, ORANGE, ORANGE]
bars = axes[2].bar(range(4), rate_material["flag_rate"], color=colors, width=0.62)
for bar, value in zip(bars, rate_material["flag_rate"]):
    axes[2].text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}%", ha="center", fontsize=10, color=INK)
axes[2].set_xticks(range(4), ["cheap\n(<0.87)", "normal\n(0.87-1.15)", "dear\n(1.15-2)", "very dear\n(>2)"])
axes[2].set_ylim(0, 112); axes[2].grid(axis="x", visible=False)
axes[2].set_ylabel("Share of orders flagged"); axes[2].set_xlabel("Price vs the material's usual price")
axes[2].set_title("The dearer the order, the likelier the flag")

fig.suptitle("Objective 5: what makes an order look anomalous", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(CHART_DIR / "compare_x_and_y.png", dpi=200, bbox_inches="tight")
plt.show()

importance.round(4).to_csv(CHART_DIR / "feature_importance.csv", header=["share_of_flags_lost"])
difference.to_csv(CHART_DIR / "flagged_vs_normal.csv")

# ---------- Verification ----------
print(f"{flagged.sum()} flagged orders compared with {(~flagged).sum()} normal orders\n")
print("Flagged vs normal (real units, and the gap in standard-deviation units):")
display(difference)
print("\nShare of the original flags that disappear when one feature is shuffled:")
display((100 * importance).round(1).to_frame("flags_lost_pct"))
print("\nFlag rate by price vs the material's usual price:")
display(rate_material)
print("\nFlag rate by price vs similar orders:")
display(rate_peer)
```

---

## 17. Evaluate Model

All figures below are on the unseen test projects.

**Classification models (ROC AUC: 0.5 is guessing, 1.0 is perfect)**

| Objective | Logistic Regression | Random Forest | Gradient Boosting | Baseline |
|---|---:|---:|---:|---:|
| 1 Cost overrun | **0.827** | 0.784 | 0.800 | 0.500 |
| 2 Schedule delay | **0.805** | 0.806 | 0.799 | 0.500 |
| 3 Serious incident | **0.857** | 0.837 | 0.810 | 0.500 |
| 4 Late delivery | **0.602** | 0.617 | 0.608 | 0.500 |

The more complex models are never meaningfully better (Random Forest is 0.001 higher on objective 2 and 0.015 higher on objective 4, within noise), so the simplest model wins. Resampling whole projects gives a 95% range of 0.749 to 0.874 for objective 1, 0.773 to 0.830 for objective 2, 0.798 to 0.918 for objective 3 and 0.571 to 0.639 for objective 4.

**The cut-off is chosen on training data only.** Each model gives a risk percentage, and the alert cut-off was set on out-of-fold training predictions (F1 for objectives 1 to 3, Youden's J for objective 4).

| Objective | Alert cut-off | Real cases caught | Precision | False alarms |
|---|---:|---|---:|---|
| 1 Cost overrun | 0.36 | 185 of 224 (83%) | 66% | 96 of 258 |
| 2 Schedule delay | 0.28 | 231 of 264 (88%) | 55% | 191 of 382 |
| 3 Serious incident | 0.37 | 26 of 38 (68%) | 59% | 18 of 114 |
| 4 Late delivery | 0.46 | 153 of 267 (57%) | 53% | 133 of 315 |

**Risk bands hold up.** Splitting the test cases into five equal bands by predicted risk, the real problem rate climbs steadily:

| Objective | Lowest band | Highest band |
|---|---:|---:|
| 1 Cost overrun | 11% overran | 92% overran |
| 2 Schedule delay | 9% delayed | 79% delayed |
| 3 Serious incident | 0% serious | 61% serious |
| 4 Late delivery | 36% late | 58% late |

**Anomaly detection (Isolation Forest):** all 62 planted anomalies were caught (100% recall) in a review list of 150 orders, with 88 false alarms (41% precision). The top 2% of orders alone (60 orders) hold 60 real anomalies. A one-line rule (price above 2 times the material norm) also catches all 62 with exactly 62 flags, so on this data the rule is as good as the model. The model's real value is on subtler cases that real invoices may hold.

### Charts

<p align="center">
  <img width="1969" height="587" alt="obj1_evaluation" src="https://github.com/user-attachments/assets/9a3fe31c-c7fc-4a47-a63a-5b78325edaf5" />
</p>
<p align="center"><em>Objective 1: Cost overruns, evaluation on unseen projects</em></p>

<p align="center">
  <img width="1969" height="587" alt="obj2_evaluation" src="https://github.com/user-attachments/assets/d2d67249-666d-49f9-8d4c-23f070aadfc4" />
</p>
<p align="center"><em>Objective 2: Schedule delays, evaluation on unseen projects</em></p>

<p align="center">
  <img width="1969" height="587" alt="obj3_evaluation" src="https://github.com/user-attachments/assets/cea2e5a7-c460-46ef-a52c-cb59d76af174" />
</p>
<p align="center"><em>Objective 3: Site safety, evaluation on unseen projects</em></p>

<p align="center">
  <img width="1969" height="587" alt="obj4_evaluation" src="https://github.com/user-attachments/assets/b1ff730a-549a-4f6a-96ac-3e69009f65be" />
</p>
<p align="center"><em>Objective 4: Late material delivery, evaluation on unseen projects</em></p>

<p align="center">
  <img width="1893" height="587" alt="obj5_evaluation" src="https://github.com/user-attachments/assets/de65a2fe-092e-4583-9de5-58852027cf66" />
</p>
<p align="center"><em>Objective 5: Price and invoice anomalies, evaluation against the answer key</em></p>

### Code: evaluate model

#### Objective 1: Cost overruns

```python
# ============================================================
# EVALUATE MODEL - BUSINESS OBJECTIVE 1: CONTROL PROJECT COST OVERRUNS
# ============================================================

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, cross_val_predict

KEY = "cost_overrun"
SEED = 42
CHART_DIR = Path("construction_charts/objective_1_cost_overrun")
CHART_DIR.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, INK, SOFT, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

X_train, y_train = scaled[KEY]["X_train"], splits[KEY]["y_train"]
X_test, y_test = scaled[KEY]["X_test"], splits[KEY]["y_test"]
groups_train = groups_all[KEY].loc[X_train.index]
groups_test = groups_all[KEY].loc[X_test.index].to_numpy()
trained_models = models_obj1["trained"]
baseline_name = "Baseline (always predicts the common answer)"
best_name = models_obj1["best_name"]
y_true = y_test.to_numpy()

# ---------- 1. Choose the alert threshold WITHOUT touching the test set ----------
# Out-of-fold chances on the training rows (whole projects held out), then the cut-off with the best F1.
oof = cross_val_predict(clone(trained_models[best_name]), X_train, y_train,
                        cv=GroupKFold(n_splits=5), groups=groups_train, method="predict_proba")[:, 1]
grid = np.linspace(0.20, 0.80, 61)
f1_by_threshold = [f1_score(y_train, oof >= t, zero_division=0) for t in grid]
THRESHOLD = float(grid[int(np.argmax(f1_by_threshold))])

# ---------- 2. Score every model on the untouched test set ----------
rows = []
for name, model in trained_models.items():
    p = model.predict_proba(X_test)[:, 1]
    cut = THRESHOLD if name == best_name else 0.5
    pred = (p >= cut).astype(int)
    rows.append({"model": name, "threshold": round(cut, 2),
                 "roc_auc": roc_auc_score(y_true, p), "average_precision": average_precision_score(y_true, p),
                 "accuracy": accuracy_score(y_true, pred),
                 "precision": precision_score(y_true, pred, zero_division=0),
                 "recall": recall_score(y_true, pred, zero_division=0),
                 "f1": f1_score(y_true, pred, zero_division=0)})
scorecard = pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)

# ---------- 3. How sure are we? Re-score on resampled test PROJECTS ----------
best_p = trained_models[best_name].predict_proba(X_test)[:, 1]
rng = np.random.default_rng(SEED)
project_ids = np.unique(groups_test)
aucs = []
for _ in range(1000):
    chosen = rng.choice(project_ids, size=len(project_ids), replace=True)
    idx = np.concatenate([np.where(groups_test == g)[0] for g in chosen])
    if len(np.unique(y_true[idx])) == 2:
        aucs.append(roc_auc_score(y_true[idx], best_p[idx]))
auc_low, auc_high = np.percentile(aucs, [2.5, 97.5])

# ---------- 4. Confusion matrix and the business reading of it ----------
best_pred = (best_p >= THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, best_pred).ravel()
confusion = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actually within budget", "Actually overran"],
                         columns=["Predicted within budget", "Predicted overrun"])

# ---------- 5. Does a higher risk score really mean more overruns? (five equal risk bands) ----------
bands = pd.qcut(pd.Series(best_p).rank(method="first"), 5, labels=["Lowest", "Low", "Middle", "High", "Highest"])
band_table = (pd.DataFrame({"band": bands, "overran": y_true, "chance": best_p})
              .groupby("band", observed=True).agg(cost_lines=("overran", "size"),
                                                  real_overrun_rate=("overran", "mean"),
                                                  avg_predicted_chance=("chance", "mean")))
top_two = band_table.loc[["High", "Highest"]]
captured = (top_two["real_overrun_rate"] * top_two["cost_lines"]).sum() / y_true.sum()

# ---------- 6. Where does the model struggle? (AUC by project type and country in the test set) ----------
case_test = case_infos[KEY].loc[X_test.index]
segment_rows = []
for column in ["project_type", "country", "contract_type"]:
    if column in case_test.columns:
        for value, idx in case_test.groupby(column).indices.items():
            if len(idx) >= 40 and len(np.unique(y_true[idx])) == 2:
                segment_rows.append({"segment": f"{column} = {value}", "cost_lines": len(idx),
                                     "overrun_rate": y_true[idx].mean(), "roc_auc": roc_auc_score(y_true[idx], best_p[idx])})
segment_table = pd.DataFrame(segment_rows).sort_values("roc_auc") if segment_rows else pd.DataFrame()

# ---------- 7. Charts ----------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": GRID,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": SOFT,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

for name, model in trained_models.items():
    fpr, tpr, _ = roc_curve(y_true, model.predict_proba(X_test)[:, 1])
    is_best = name == best_name
    axes[0].plot(fpr, tpr, color=BLUE if is_best else SOFT, linewidth=2.6 if is_best else 1.2,
                 alpha=1 if is_best else 0.55)
    if name.startswith("Baseline"):
        continue
    label_y = {best_name: 0.30, "Gradient Boosting": 0.22, "Random Forest": 0.14}[name]
    axes[0].text(0.45, label_y, f"{name} (AUC {scorecard.loc[name, 'roc_auc']:.2f})",
                 fontsize=9.5, color=BLUE if is_best else SOFT, fontweight="bold" if is_best else "normal")
axes[0].plot([0, 1], [0, 1], color=GRID, linestyle="--")
axes[0].set_xlabel("False alarm rate"); axes[0].set_ylabel("Share of real overruns caught")
axes[0].set_title("ROC curve on unseen projects")

axes[1].imshow(confusion.values, cmap="Blues", vmin=0, vmax=confusion.values.max() * 1.4)
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{confusion.values[i, j]:,}", ha="center", va="center", fontsize=18, fontweight="bold",
                     color="white" if confusion.values[i, j] > confusion.values.max() * 0.55 else INK)
axes[1].set_xticks([0, 1], ["Predicted\nwithin budget", "Predicted\noverrun"])
axes[1].set_yticks([0, 1], ["Actually\nwithin budget", "Actually\noverran"])
axes[1].grid(False)
axes[1].set_title(f"{best_name} at {THRESHOLD:.2f} cut-off")

colors = [ORANGE if label in ("High", "Highest") else BLUE for label in band_table.index]
bars = axes[2].bar(band_table.index.astype(str), 100 * band_table["real_overrun_rate"], color=colors, width=0.62)
for bar, value in zip(bars, 100 * band_table["real_overrun_rate"]):
    axes[2].text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.0f}%", ha="center", fontsize=10, color=INK)
axes[2].axhline(100 * y_true.mean(), color=SOFT, linestyle="--", linewidth=1)
axes[2].text(-0.4, 100 * y_true.mean() + 1.5, f"all lines: {100 * y_true.mean():.0f}%", ha="left", fontsize=9, color=SOFT)
axes[2].set_ylabel("Real overrun rate"); axes[2].set_xlabel("Model risk band (five equal groups)")
axes[2].set_title("Higher risk score = more real overruns"); axes[2].grid(axis="x", visible=False)

fig.suptitle("Objective 1: how well does the model spot cost overruns?", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(CHART_DIR / "evaluate_model.png", dpi=200, bbox_inches="tight")
plt.show()

scorecard.round(4).to_csv(CHART_DIR / "model_scorecard.csv")
band_table.round(4).to_csv(CHART_DIR / "risk_band_table.csv")

# ---------- Verification ----------
print(f"Best model: {best_name} | alert cut-off {THRESHOLD:.2f} (chosen on training folds, not on the test set)")
print(f"Test set: {len(y_true)} cost lines from {len(project_ids)} unseen projects, {100 * y_true.mean():.1f}% overran\n")
print("Scorecard on the test set:")
display(scorecard.round(3))
print(f"\nROC AUC of the best model: {roc_auc_score(y_true, best_p):.3f} "
      f"(95% range when whole projects are resampled: {auc_low:.3f} to {auc_high:.3f})")
print(f"Overruns caught: {tp} of {tp + fn} ({tp / (tp + fn):.0%}) | false alarms: {fp} of {tn + fp} ({fp / (tn + fp):.0%}) | "
      f"when it flags a line, it is right {tp / max(tp + fp, 1):.0%} of the time")
print(f"The two highest risk bands hold {captured:.0%} of all real overruns in {top_two['cost_lines'].sum() / len(y_true):.0%} of the lines\n")
print("Real overrun rate by risk band:")
display(band_table.round(3))
if len(segment_table):
    print("\nWhere the model is weaker or stronger (ROC AUC inside each segment of the test set):")
    display(segment_table.round(3))
print("\nConfusion matrix:")
display(confusion)
```

#### Objective 2: Schedule delays

```python
# ============================================================
# EVALUATE MODEL - BUSINESS OBJECTIVE 2: REDUCE SCHEDULE DELAYS
# ============================================================

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, cross_val_predict

KEY = "schedule_delay"
SEED = 42
CHART_DIR = Path("construction_charts/objective_2_schedule_delay")
CHART_DIR.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, INK, SOFT, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

X_train, y_train = scaled[KEY]["X_train"], splits[KEY]["y_train"]
X_test, y_test = scaled[KEY]["X_test"], splits[KEY]["y_test"]
groups_train = groups_all[KEY].loc[X_train.index]
groups_test = groups_all[KEY].loc[X_test.index].to_numpy()
trained_models = models_obj2["trained"]
baseline_name = "Baseline (always predicts the common answer)"
best_name = models_obj2["best_name"]
y_true = y_test.to_numpy()

# ---------- 1. Choose the alert threshold WITHOUT touching the test set ----------
# Out-of-fold chances on the training rows (whole projects held out), then the cut-off with the best F1.
oof = cross_val_predict(clone(trained_models[best_name]), X_train, y_train,
                        cv=GroupKFold(n_splits=5), groups=groups_train, method="predict_proba")[:, 1]
grid = np.linspace(0.20, 0.80, 61)
f1_by_threshold = [f1_score(y_train, oof >= t, zero_division=0) for t in grid]
THRESHOLD = float(grid[int(np.argmax(f1_by_threshold))])

# ---------- 2. Score every model on the untouched test set ----------
rows = []
for name, model in trained_models.items():
    p = model.predict_proba(X_test)[:, 1]
    cut = THRESHOLD if name == best_name else 0.5
    pred = (p >= cut).astype(int)
    rows.append({"model": name, "threshold": round(cut, 2),
                 "roc_auc": roc_auc_score(y_true, p), "average_precision": average_precision_score(y_true, p),
                 "accuracy": accuracy_score(y_true, pred),
                 "precision": precision_score(y_true, pred, zero_division=0),
                 "recall": recall_score(y_true, pred, zero_division=0),
                 "f1": f1_score(y_true, pred, zero_division=0)})
scorecard = pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)

# ---------- 3. How sure are we? Re-score on resampled test PROJECTS ----------
best_p = trained_models[best_name].predict_proba(X_test)[:, 1]
rng = np.random.default_rng(SEED)
project_ids = np.unique(groups_test)
aucs = []
for _ in range(1000):
    chosen = rng.choice(project_ids, size=len(project_ids), replace=True)
    idx = np.concatenate([np.where(groups_test == g)[0] for g in chosen])
    if len(np.unique(y_true[idx])) == 2:
        aucs.append(roc_auc_score(y_true[idx], best_p[idx]))
auc_low, auc_high = np.percentile(aucs, [2.5, 97.5])

# ---------- 4. Confusion matrix and the business reading of it ----------
best_pred = (best_p >= THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, best_pred).ravel()
confusion = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actually on time", "Actually delayed"],
                         columns=["Predicted on time", "Predicted delay"])

# ---------- 5. Does a higher risk score really mean more delays? (five equal risk bands) ----------
bands = pd.qcut(pd.Series(best_p).rank(method="first"), 5, labels=["Lowest", "Low", "Middle", "High", "Highest"])
band_table = (pd.DataFrame({"band": bands, "delayed": y_true, "chance": best_p})
              .groupby("band", observed=True).agg(tasks=("delayed", "size"),
                                                  real_delay_rate=("delayed", "mean"),
                                                  avg_predicted_chance=("chance", "mean")))
top_two = band_table.loc[["High", "Highest"]]
captured = (top_two["real_delay_rate"] * top_two["tasks"]).sum() / y_true.sum()

# ---------- 5b. What if the alert cut-off were stricter? (shown for the manager, not used to pick the cut-off) ----------
trade_rows = []
for cut in [THRESHOLD, 0.35, 0.40, 0.45, 0.50]:
    pred_cut = (best_p >= cut).astype(int)
    trade_rows.append({"cut_off": round(cut, 2), "tasks_flagged_pct": 100 * pred_cut.mean(),
                       "delays_caught_pct": 100 * recall_score(y_true, pred_cut),
                       "right_when_flagged_pct": 100 * precision_score(y_true, pred_cut, zero_division=0),
                       "false_alarm_pct": 100 * ((pred_cut == 1) & (y_true == 0)).sum() / (y_true == 0).sum()})
trade_off = pd.DataFrame(trade_rows).set_index("cut_off")

# ---------- 6. Where does the model struggle? (AUC by project type and country in the test set) ----------
case_test = case_infos[KEY].loc[X_test.index]
segment_rows = []
for column in ["project_type", "country", "contract_type"]:
    if column in case_test.columns:
        for value, idx in case_test.groupby(column).indices.items():
            if len(idx) >= 40 and len(np.unique(y_true[idx])) == 2:
                segment_rows.append({"segment": f"{column} = {value}", "tasks": len(idx),
                                     "delay_rate": y_true[idx].mean(), "roc_auc": roc_auc_score(y_true[idx], best_p[idx])})
segment_table = pd.DataFrame(segment_rows).sort_values("roc_auc") if segment_rows else pd.DataFrame()

# ---------- 7. Charts ----------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": GRID,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": SOFT,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

for name, model in trained_models.items():
    fpr, tpr, _ = roc_curve(y_true, model.predict_proba(X_test)[:, 1])
    is_best = name == best_name
    axes[0].plot(fpr, tpr, color=BLUE if is_best else SOFT, linewidth=2.6 if is_best else 1.2,
                 alpha=1 if is_best else 0.55)
    if name.startswith("Baseline"):
        continue
    label_y = {best_name: 0.30, "Gradient Boosting": 0.22, "Random Forest": 0.14}[name]
    axes[0].text(0.45, label_y, f"{name} (AUC {scorecard.loc[name, 'roc_auc']:.2f})",
                 fontsize=9.5, color=BLUE if is_best else SOFT, fontweight="bold" if is_best else "normal")
axes[0].plot([0, 1], [0, 1], color=GRID, linestyle="--")
axes[0].set_xlabel("False alarm rate"); axes[0].set_ylabel("Share of real delays caught")
axes[0].set_title("ROC curve on unseen projects")

axes[1].imshow(confusion.values, cmap="Blues", vmin=0, vmax=confusion.values.max() * 1.4)
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{confusion.values[i, j]:,}", ha="center", va="center", fontsize=18, fontweight="bold",
                     color="white" if confusion.values[i, j] > confusion.values.max() * 0.55 else INK)
axes[1].set_xticks([0, 1], ["Predicted\non time", "Predicted\ndelay"])
axes[1].set_yticks([0, 1], ["Actually\non time", "Actually\ndelayed"])
axes[1].grid(False)
axes[1].set_title(f"{best_name} at {THRESHOLD:.2f} cut-off")

colors = [ORANGE if label in ("High", "Highest") else BLUE for label in band_table.index]
bars = axes[2].bar(band_table.index.astype(str), 100 * band_table["real_delay_rate"], color=colors, width=0.62)
for bar, value in zip(bars, 100 * band_table["real_delay_rate"]):
    axes[2].text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.0f}%", ha="center", fontsize=10, color=INK)
axes[2].axhline(100 * y_true.mean(), color=SOFT, linestyle="--", linewidth=1)
axes[2].text(-0.4, 100 * y_true.mean() + 1.5, f"all tasks: {100 * y_true.mean():.0f}%", ha="left", fontsize=9, color=SOFT)
axes[2].set_ylabel("Real delay rate"); axes[2].set_xlabel("Model risk band (five equal groups)")
axes[2].set_title("Higher risk score = more real delays"); axes[2].grid(axis="x", visible=False)

fig.suptitle("Objective 2: how well does the model spot schedule delays?", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(CHART_DIR / "evaluate_model.png", dpi=200, bbox_inches="tight")
plt.show()

scorecard.round(4).to_csv(CHART_DIR / "model_scorecard.csv")
band_table.round(4).to_csv(CHART_DIR / "risk_band_table.csv")
trade_off.round(2).to_csv(CHART_DIR / "cutoff_trade_off.csv")

# ---------- Verification ----------
print(f"Best model: {best_name} | alert cut-off {THRESHOLD:.2f} (chosen on training folds, not on the test set)")
print(f"Test set: {len(y_true)} tasks from {len(project_ids)} unseen projects, {100 * y_true.mean():.1f}% delayed\n")
print("Scorecard on the test set:")
display(scorecard.round(3))
print(f"\nROC AUC of the best model: {roc_auc_score(y_true, best_p):.3f} "
      f"(95% range when whole projects are resampled: {auc_low:.3f} to {auc_high:.3f})")
print(f"Delays caught: {tp} of {tp + fn} ({tp / (tp + fn):.0%}) | false alarms: {fp} of {tn + fp} ({fp / (tn + fp):.0%}) | "
      f"when it flags a task, it is right {tp / max(tp + fp, 1):.0%} of the time")
print(f"The two highest risk bands hold {captured:.0%} of all real delays in {top_two['tasks'].sum() / len(y_true):.0%} of the tasks\n")
print("Real delay rate by risk band:")
display(band_table.round(3))
if len(segment_table):
    print("\nWhere the model is weaker or stronger (ROC AUC inside each segment of the test set):")
    display(segment_table.round(3))
print("\nWhat a stricter alert cut-off would change (test set):")
display(trade_off.round(1))
print("\nConfusion matrix:")
display(confusion)
```

#### Objective 3: Site safety

```python
# ============================================================
# EVALUATE MODEL - BUSINESS OBJECTIVE 3: PREVENT SERIOUS SAFETY INCIDENTS
# ============================================================

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, cross_val_predict

KEY = "site_safety"
SEED = 42
CHART_DIR = Path("construction_charts/objective_3_site_safety")
CHART_DIR.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, INK, SOFT, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

X_train, y_train = scaled[KEY]["X_train"], splits[KEY]["y_train"]
X_test, y_test = scaled[KEY]["X_test"], splits[KEY]["y_test"]
groups_train = groups_all[KEY].loc[X_train.index]
groups_test = groups_all[KEY].loc[X_test.index].to_numpy()
trained_models = models_obj3["trained"]
baseline_name = "Baseline (always predicts the common answer)"
best_name = models_obj3["best_name"]
y_true = y_test.to_numpy()

# ---------- 1. Choose the alert threshold WITHOUT touching the test set ----------
# Out-of-fold chances on the training rows (whole projects held out), then the cut-off with the best F1.
oof = cross_val_predict(clone(trained_models[best_name]), X_train, y_train,
                        cv=GroupKFold(n_splits=5), groups=groups_train, method="predict_proba")[:, 1]
grid = np.linspace(0.10, 0.80, 71)
f1_by_threshold = [f1_score(y_train, oof >= t, zero_division=0) for t in grid]
THRESHOLD = float(grid[int(np.argmax(f1_by_threshold))])

# ---------- 2. Score every model on the untouched test set ----------
rows = []
for name, model in trained_models.items():
    p = model.predict_proba(X_test)[:, 1]
    cut = THRESHOLD if name == best_name else 0.5
    pred = (p >= cut).astype(int)
    rows.append({"model": name, "threshold": round(cut, 2),
                 "roc_auc": roc_auc_score(y_true, p), "average_precision": average_precision_score(y_true, p),
                 "accuracy": accuracy_score(y_true, pred),
                 "precision": precision_score(y_true, pred, zero_division=0),
                 "recall": recall_score(y_true, pred, zero_division=0),
                 "f1": f1_score(y_true, pred, zero_division=0)})
scorecard = pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)

# ---------- 3. How sure are we? Re-score on resampled test PROJECTS ----------
best_p = trained_models[best_name].predict_proba(X_test)[:, 1]
rng = np.random.default_rng(SEED)
project_ids = np.unique(groups_test)
aucs = []
for _ in range(1000):
    chosen = rng.choice(project_ids, size=len(project_ids), replace=True)
    idx = np.concatenate([np.where(groups_test == g)[0] for g in chosen])
    if len(np.unique(y_true[idx])) == 2:
        aucs.append(roc_auc_score(y_true[idx], best_p[idx]))
auc_low, auc_high = np.percentile(aucs, [2.5, 97.5])

# ---------- 4. Confusion matrix and the business reading of it ----------
best_pred = (best_p >= THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, best_pred).ravel()
confusion = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actually not serious", "Actually serious"],
                         columns=["Predicted not serious", "Predicted serious"])

# ---------- 5. Does a higher risk score really mean more serious incidents? (five equal risk bands) ----------
bands = pd.qcut(pd.Series(best_p).rank(method="first"), 5, labels=["Lowest", "Low", "Middle", "High", "Highest"])
band_table = (pd.DataFrame({"band": bands, "serious": y_true, "chance": best_p})
              .groupby("band", observed=True).agg(incidents=("serious", "size"),
                                                  real_serious_rate=("serious", "mean"),
                                                  avg_predicted_chance=("chance", "mean")))
top_two = band_table.loc[["High", "Highest"]]
captured = (top_two["real_serious_rate"] * top_two["incidents"]).sum() / y_true.sum()

# ---------- 5b. What if the alert cut-off were different? (shown for the manager, not used to pick the cut-off) ----------
trade_rows = []
for cut in [0.30, THRESHOLD, 0.40, 0.50]:
    pred_cut = (best_p >= cut).astype(int)
    trade_rows.append({"cut_off": round(cut, 2), "incidents_flagged_pct": 100 * pred_cut.mean(),
                       "serious_caught_pct": 100 * recall_score(y_true, pred_cut),
                       "right_when_flagged_pct": 100 * precision_score(y_true, pred_cut, zero_division=0),
                       "false_alarm_pct": 100 * ((pred_cut == 1) & (y_true == 0)).sum() / (y_true == 0).sum()})
trade_off = pd.DataFrame(trade_rows).set_index("cut_off")

# ---------- 6. Where does the model struggle? (AUC by country and project type in the test set) ----------
case_test = case_infos[KEY].loc[X_test.index]
segment_rows = []
for column in ["project_type", "country", "contract_type"]:
    if column in case_test.columns:
        for value, idx in case_test.groupby(column).indices.items():
            if len(idx) >= 40 and len(np.unique(y_true[idx])) == 2:
                segment_rows.append({"segment": f"{column} = {value}", "incidents": len(idx),
                                     "serious_rate": y_true[idx].mean(), "roc_auc": roc_auc_score(y_true[idx], best_p[idx])})
segment_table = pd.DataFrame(segment_rows).sort_values("roc_auc") if segment_rows else pd.DataFrame()

# ---------- 7. Charts ----------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": GRID,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": SOFT,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

for name, model in trained_models.items():
    fpr, tpr, _ = roc_curve(y_true, model.predict_proba(X_test)[:, 1])
    is_best = name == best_name
    axes[0].plot(fpr, tpr, color=BLUE if is_best else SOFT, linewidth=2.6 if is_best else 1.2,
                 alpha=1 if is_best else 0.55)
    if name.startswith("Baseline"):
        continue
    label_y = {best_name: 0.30, "Gradient Boosting": 0.22, "Random Forest": 0.14}[name]
    axes[0].text(0.45, label_y, f"{name} (AUC {scorecard.loc[name, 'roc_auc']:.2f})",
                 fontsize=9.5, color=BLUE if is_best else SOFT, fontweight="bold" if is_best else "normal")
axes[0].plot([0, 1], [0, 1], color=GRID, linestyle="--")
axes[0].set_xlabel("False alarm rate"); axes[0].set_ylabel("Share of serious incidents caught")
axes[0].set_title("ROC curve on unseen projects")

axes[1].imshow(confusion.values, cmap="Blues", vmin=0, vmax=confusion.values.max() * 1.4)
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{confusion.values[i, j]:,}", ha="center", va="center", fontsize=18, fontweight="bold",
                     color="white" if confusion.values[i, j] > confusion.values.max() * 0.55 else INK)
axes[1].set_xticks([0, 1], ["Predicted\nnot serious", "Predicted\nserious"])
axes[1].set_yticks([0, 1], ["Actually\nnot serious", "Actually\nserious"])
axes[1].grid(False)
axes[1].set_title(f"{best_name} at {THRESHOLD:.2f} cut-off")

colors = [ORANGE if label in ("High", "Highest") else BLUE for label in band_table.index]
bars = axes[2].bar(band_table.index.astype(str), 100 * band_table["real_serious_rate"], color=colors, width=0.62)
for bar, value in zip(bars, 100 * band_table["real_serious_rate"]):
    axes[2].text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.0f}%", ha="center", fontsize=10, color=INK)
axes[2].axhline(100 * y_true.mean(), color=SOFT, linestyle="--", linewidth=1)
axes[2].text(-0.4, 100 * y_true.mean() + 1.5, f"all incidents: {100 * y_true.mean():.0f}%", ha="left", fontsize=9, color=SOFT)
axes[2].set_ylabel("Serious incident rate"); axes[2].set_xlabel("Model risk band (five equal groups)")
axes[2].set_title("Higher risk score = more serious incidents"); axes[2].grid(axis="x", visible=False)

fig.suptitle("Objective 3: how well does the model spot serious safety incidents?", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(CHART_DIR / "evaluate_model.png", dpi=200, bbox_inches="tight")
plt.show()

scorecard.round(4).to_csv(CHART_DIR / "model_scorecard.csv")
band_table.round(4).to_csv(CHART_DIR / "risk_band_table.csv")
trade_off.round(2).to_csv(CHART_DIR / "cutoff_trade_off.csv")

# ---------- Verification ----------
print(f"Best model: {best_name} | alert cut-off {THRESHOLD:.2f} (chosen on training folds, not on the test set)")
print(f"Test set: {len(y_true)} incidents from {len(project_ids)} unseen projects, {100 * y_true.mean():.1f}% serious\n")
print("Scorecard on the test set:")
display(scorecard.round(3))
print(f"\nROC AUC of the best model: {roc_auc_score(y_true, best_p):.3f} "
      f"(95% range when whole projects are resampled: {auc_low:.3f} to {auc_high:.3f})")
print(f"Serious incidents caught: {tp} of {tp + fn} ({tp / (tp + fn):.0%}) | false alarms: {fp} of {tn + fp} ({fp / (tn + fp):.0%}) | "
      f"when it flags an incident, it is right {tp / max(tp + fp, 1):.0%} of the time")
print(f"The two highest risk bands hold {captured:.0%} of all serious incidents in {top_two['incidents'].sum() / len(y_true):.0%} of the incidents\n")
print("Serious incident rate by risk band:")
display(band_table.round(3))
if len(segment_table):
    print("\nWhere the model is weaker or stronger (ROC AUC inside each segment of the test set):")
    display(segment_table.round(3))
print("\nWhat a different alert cut-off would change (test set):")
display(trade_off.round(1))
print("\nConfusion matrix:")
display(confusion)
```

#### Objective 4: Late material delivery

```python
# ============================================================
# EVALUATE MODEL - BUSINESS OBJECTIVE 4: PREVENT LATE MATERIAL DELIVERIES
# ============================================================

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, cross_val_predict

KEY = "late_delivery"
SEED = 42
CHART_DIR = Path("construction_charts/objective_4_late_delivery")
CHART_DIR.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, INK, SOFT, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

X_train, y_train = scaled[KEY]["X_train"], splits[KEY]["y_train"]
X_test, y_test = scaled[KEY]["X_test"], splits[KEY]["y_test"]
groups_train = groups_all[KEY].loc[X_train.index]
groups_test = groups_all[KEY].loc[X_test.index].to_numpy()
trained_models = models_obj4["trained"]
baseline_name = "Baseline (always predicts the common answer)"
best_name = models_obj4["best_name"]
y_true = y_test.to_numpy()

# ---------- 1. Choose the alert threshold WITHOUT touching the test set ----------
# Out-of-fold chances on the training rows (whole projects held out), then the cut-off with the best balance between
# catching late orders and false alarms (Youden's J). F1 is not used here because with a weak model it
# simply flags almost every order.
oof = cross_val_predict(clone(trained_models[best_name]), X_train, y_train,
                        cv=GroupKFold(n_splits=5), groups=groups_train, method="predict_proba")[:, 1]
grid = np.linspace(0.20, 0.80, 61)
def balance_score(y, flagged):
    return recall_score(y, flagged) - ((flagged == 1) & (y.to_numpy() == 0)).sum() / (y.to_numpy() == 0).sum()
balance_by_threshold = [balance_score(y_train, (oof >= t).astype(int)) for t in grid]
THRESHOLD = float(grid[int(np.argmax(balance_by_threshold))])

# ---------- 2. Score every model on the untouched test set ----------
rows = []
for name, model in trained_models.items():
    p = model.predict_proba(X_test)[:, 1]
    cut = THRESHOLD if name == best_name else 0.5
    pred = (p >= cut).astype(int)
    rows.append({"model": name, "threshold": round(cut, 2),
                 "roc_auc": roc_auc_score(y_true, p), "average_precision": average_precision_score(y_true, p),
                 "accuracy": accuracy_score(y_true, pred),
                 "precision": precision_score(y_true, pred, zero_division=0),
                 "recall": recall_score(y_true, pred, zero_division=0),
                 "f1": f1_score(y_true, pred, zero_division=0)})
scorecard = pd.DataFrame(rows).set_index("model").sort_values("roc_auc", ascending=False)

# ---------- 3. How sure are we? Re-score on resampled test PROJECTS ----------
best_p = trained_models[best_name].predict_proba(X_test)[:, 1]
rng = np.random.default_rng(SEED)
project_ids = np.unique(groups_test)
aucs = []
for _ in range(1000):
    chosen = rng.choice(project_ids, size=len(project_ids), replace=True)
    idx = np.concatenate([np.where(groups_test == g)[0] for g in chosen])
    if len(np.unique(y_true[idx])) == 2:
        aucs.append(roc_auc_score(y_true[idx], best_p[idx]))
auc_low, auc_high = np.percentile(aucs, [2.5, 97.5])

# ---------- 4. Confusion matrix and the business reading of it ----------
best_pred = (best_p >= THRESHOLD).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, best_pred).ravel()
confusion = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actually on time", "Actually late"],
                         columns=["Predicted on time", "Predicted late"])

# ---------- 5. Does a higher risk score really mean more late orders? (five equal risk bands) ----------
bands = pd.qcut(pd.Series(best_p).rank(method="first"), 5, labels=["Lowest", "Low", "Middle", "High", "Highest"])
band_table = (pd.DataFrame({"band": bands, "late": y_true, "chance": best_p})
              .groupby("band", observed=True).agg(orders=("late", "size"),
                                                  real_late_rate=("late", "mean"),
                                                  avg_predicted_chance=("chance", "mean")))
top_two = band_table.loc[["High", "Highest"]]
captured = (top_two["real_late_rate"] * top_two["orders"]).sum() / y_true.sum()

# ---------- 5b. What if the alert cut-off were different? (shown for the manager, not used to pick the cut-off) ----------
trade_rows = []
for cut in [0.40, THRESHOLD, 0.50, 0.55]:
    pred_cut = (best_p >= cut).astype(int)
    trade_rows.append({"cut_off": round(cut, 2), "orders_flagged_pct": 100 * pred_cut.mean(),
                       "late_caught_pct": 100 * recall_score(y_true, pred_cut),
                       "right_when_flagged_pct": 100 * precision_score(y_true, pred_cut, zero_division=0),
                       "false_alarm_pct": 100 * ((pred_cut == 1) & (y_true == 0)).sum() / (y_true == 0).sum()})
trade_off = pd.DataFrame(trade_rows).set_index("cut_off")

# ---------- 6. Where does the model struggle? (AUC by project type, country and material in the test set) ----------
case_test = case_infos[KEY].loc[X_test.index]
segment_rows = []
for column in ["project_type", "country", "material"]:
    if column in case_test.columns:
        for value, idx in case_test.groupby(column).indices.items():
            if len(idx) >= 40 and len(np.unique(y_true[idx])) == 2:
                segment_rows.append({"segment": f"{column} = {value}", "orders": len(idx),
                                     "late_rate": y_true[idx].mean(), "roc_auc": roc_auc_score(y_true[idx], best_p[idx])})
segment_table = pd.DataFrame(segment_rows).sort_values("roc_auc") if segment_rows else pd.DataFrame()

# ---------- 7. Charts ----------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": GRID,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": SOFT,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

for name, model in trained_models.items():
    fpr, tpr, _ = roc_curve(y_true, model.predict_proba(X_test)[:, 1])
    is_best = name == best_name
    axes[0].plot(fpr, tpr, color=BLUE if is_best else SOFT, linewidth=2.6 if is_best else 1.2,
                 alpha=1 if is_best else 0.55)
    if name.startswith("Baseline"):
        continue
    label_y = {best_name: 0.30, "Gradient Boosting": 0.22, "Random Forest": 0.14}[name]
    axes[0].text(0.45, label_y, f"{name} (AUC {scorecard.loc[name, 'roc_auc']:.2f})",
                 fontsize=9.5, color=BLUE if is_best else SOFT, fontweight="bold" if is_best else "normal")
axes[0].plot([0, 1], [0, 1], color=GRID, linestyle="--")
axes[0].set_xlabel("False alarm rate"); axes[0].set_ylabel("Share of late orders caught")
axes[0].set_title("ROC curve on unseen projects")

axes[1].imshow(confusion.values, cmap="Blues", vmin=0, vmax=confusion.values.max() * 1.4)
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{confusion.values[i, j]:,}", ha="center", va="center", fontsize=18, fontweight="bold",
                     color="white" if confusion.values[i, j] > confusion.values.max() * 0.55 else INK)
axes[1].set_xticks([0, 1], ["Predicted\non time", "Predicted\nlate"])
axes[1].set_yticks([0, 1], ["Actually\non time", "Actually\nlate"])
axes[1].grid(False)
axes[1].set_title(f"{best_name} at {THRESHOLD:.2f} cut-off")

colors = [ORANGE if label in ("High", "Highest") else BLUE for label in band_table.index]
bars = axes[2].bar(band_table.index.astype(str), 100 * band_table["real_late_rate"], color=colors, width=0.62)
for bar, value in zip(bars, 100 * band_table["real_late_rate"]):
    axes[2].text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.0f}%", ha="center", fontsize=10, color=INK)
axes[2].axhline(100 * y_true.mean(), color=SOFT, linestyle="--", linewidth=1)
axes[2].text(-0.4, 100 * y_true.mean() + 1.5, f"all orders: {100 * y_true.mean():.0f}%", ha="left", fontsize=9, color=SOFT)
axes[2].set_ylabel("Late order rate"); axes[2].set_xlabel("Model risk band (five equal groups)")
axes[2].set_title("Higher risk score = more late orders"); axes[2].grid(axis="x", visible=False)

fig.suptitle("Objective 4: how well does the model spot late material deliveries?", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(CHART_DIR / "evaluate_model.png", dpi=200, bbox_inches="tight")
plt.show()

scorecard.round(4).to_csv(CHART_DIR / "model_scorecard.csv")
band_table.round(4).to_csv(CHART_DIR / "risk_band_table.csv")
trade_off.round(2).to_csv(CHART_DIR / "cutoff_trade_off.csv")

# ---------- Verification ----------
print(f"Best model: {best_name} | alert cut-off {THRESHOLD:.2f} (chosen on training folds, not on the test set)")
print(f"Test set: {len(y_true)} orders from {len(project_ids)} unseen projects, {100 * y_true.mean():.1f}% late\n")
print("Scorecard on the test set:")
display(scorecard.round(3))
print(f"\nROC AUC of the best model: {roc_auc_score(y_true, best_p):.3f} "
      f"(95% range when whole projects are resampled: {auc_low:.3f} to {auc_high:.3f})")
print(f"Late orders caught: {tp} of {tp + fn} ({tp / (tp + fn):.0%}) | false alarms: {fp} of {tn + fp} ({fp / (tn + fp):.0%}) | "
      f"when it flags an order, it is right {tp / max(tp + fp, 1):.0%} of the time")
print(f"The two highest risk bands hold {captured:.0%} of all late orders in {top_two['orders'].sum() / len(y_true):.0%} of the orders\n")
print("Late order rate by risk band:")
display(band_table.round(3))
if len(segment_table):
    print("\nWhere the model is weaker or stronger (ROC AUC inside each segment of the test set):")
    display(segment_table.round(3))
print("\nWhat a different alert cut-off would change (test set):")
display(trade_off.round(1))
print("\nConfusion matrix:")
display(confusion)
```

#### Objective 5: Price and invoice anomalies

```python
# ============================================================
# EVALUATE MODEL - BUSINESS OBJECTIVE 5: SPOT PRICE AND INVOICE ANOMALIES
# This is the only step that opens the answer key (anomaly_answer_key.csv).
# ============================================================

# %%
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

KEY = "price_anomalies"
CHART_DIR = Path("construction_charts/objective_5_price_anomalies")
CHART_DIR.mkdir(parents=True, exist_ok=True)
BLUE, ORANGE, INK, SOFT, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

# ---------- 1. Open the answer key (set ANSWER_KEY_PATH if your folder is named differently) ----------
ANSWER_KEY_PATH = "construction_raw/anomaly_answer_key.csv"
candidates = [ANSWER_KEY_PATH, "anomaly_answer_key.csv", "../construction_raw/anomaly_answer_key.csv",
              "data/construction_raw/anomaly_answer_key.csv"]
found = next((p for p in candidates if Path(p).exists()), None)
assert found is not None, f"Could not find anomaly_answer_key.csv. Set ANSWER_KEY_PATH at the top of this cell. Tried: {candidates}"
answer_key = pd.read_csv(found)

scored = predictions.merge(answer_key, on="order_id", how="left", validate="one_to_one")
assert scored["is_anomaly"].notna().all(), "some orders are missing from the answer key"
assert len(scored) == len(predictions) == 3000
y_true = scored["is_anomaly"].astype(int).to_numpy()
score = scored["anomaly_score"].to_numpy()
flag = scored["flagged"].to_numpy().astype(int)
n_true = int(y_true.sum())

# ---------- 2. Scorecard at the review budget we chose (top 5% flagged) ----------
tn, fp, fn, tp = confusion_matrix(y_true, flag).ravel()
scorecard = pd.Series({
    "orders": len(y_true), "real anomalies": n_true, "orders flagged": int(flag.sum()),
    "anomalies caught": int(tp), "anomalies missed": int(fn), "false alarms": int(fp),
    "recall (share of anomalies caught)": recall_score(y_true, flag),
    "precision (flags that are real)": precision_score(y_true, flag, zero_division=0),
    "F1": f1_score(y_true, flag, zero_division=0),
    "ROC AUC of the score": roc_auc_score(y_true, score),
    "average precision of the score": average_precision_score(y_true, score),
}).to_frame("value")

# ---------- 3. What if the team reviews more or fewer orders? ----------
order_by_risk = np.argsort(-score)
budget_rows = []
for share in [0.01, 0.02, 0.03, 0.05, 0.10]:
    k = int(round(share * len(score)))
    top = y_true[order_by_risk[:k]]
    budget_rows.append({"review_share": f"{share:.0%}", "orders_reviewed": k, "anomalies_caught": int(top.sum()),
                        "recall_pct": 100 * top.sum() / n_true, "precision_pct": 100 * top.mean(),
                        "random_review_would_catch": round(k * n_true / len(score), 1)})
budget_table = pd.DataFrame(budget_rows).set_index("review_share")
k_true = n_true
precision_at_true_count = y_true[order_by_risk[:k_true]].mean()

# ---------- 3b. Is the model better than a one-line rule? ----------
rule_flag = (scored["price_ratio_material"] > 2).astype(int).to_numpy()
rule_check = pd.DataFrame({
    "Isolation Forest (top 5%)": [int(flag.sum()), int(((flag == 1) & (y_true == 1)).sum()), recall_score(y_true, flag), precision_score(y_true, flag, zero_division=0)],
    "Simple rule: price above 2x the material norm": [int(rule_flag.sum()), int(((rule_flag == 1) & (y_true == 1)).sum()), recall_score(y_true, rule_flag), precision_score(y_true, rule_flag, zero_division=0)],
}, index=["orders flagged", "anomalies caught", "recall", "precision"]).T

# ---------- 4. Who are the misses and the false alarms? ----------
scored["price_ratio_band"] = pd.cut(scored["price_ratio_material"], [0, 0.87, 1.15, 2, 100],
                                    labels=["cheap (<0.87)", "normal", "dear (1.15-2)", "very dear (>2)"])
miss_table = scored[(scored["is_anomaly"] == 1) & (scored["flagged"] == False)][
    ["order_id", "project_id", "anomaly_score", "rank", "price_ratio_material", "price_ratio_peer", "unit_price_usd"]]
false_alarm_table = scored[(scored["is_anomaly"] == 0) & (scored["flagged"])]
false_alarm_reasons = false_alarm_table["main_reason"].value_counts().to_frame("false_alarms")
band_table = (scored.groupby("price_ratio_band", observed=True)
              .agg(orders=("is_anomaly", "size"), real_anomalies=("is_anomaly", "sum"), flagged=("flagged", "sum")))

# ---------- 5. Chart ----------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": GRID,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": SOFT,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

share_reviewed = np.arange(1, len(score) + 1) / len(score)
caught = np.cumsum(y_true[order_by_risk]) / n_true
axes[0].plot(100 * share_reviewed, 100 * caught, color=BLUE, linewidth=2.6)
axes[0].plot([0, 100], [0, 100], color=GRID, linestyle="--")
axes[0].axvline(5, color=ORANGE, linewidth=1.2)
axes[0].text(5.8, 40, "our 5% review budget", color=ORANGE, fontsize=9.5)
axes[0].text(18, 22, "random review", color=SOFT, fontsize=9.5)
axes[0].set_xlim(0, 30); axes[0].set_ylim(0, 102)
axes[0].set_xlabel("Share of orders reviewed (highest risk first)"); axes[0].set_ylabel("Share of real anomalies caught")
axes[0].set_title("Reviewing the riskiest orders first")

bins = np.linspace(score.min(), score.max(), 45)
axes[1].hist(score[y_true == 0], bins=bins, color=BLUE, alpha=0.85, label="normal orders")
axes[1].hist(score[y_true == 1], bins=bins, color=ORANGE, alpha=0.95, label="real anomalies")
axes[1].axvline(models_obj5["cutoff_score"], color=INK, linestyle="--", linewidth=1)
axes[1].text(models_obj5["cutoff_score"] + 0.004, axes[1].get_ylim()[1] * 0.9, "flag cut-off", fontsize=9.5, color=INK)
axes[1].set_yscale("log"); axes[1].legend(frameon=False)
axes[1].set_xlabel("Anomaly score (higher = more unusual)"); axes[1].set_ylabel("Orders (log scale)")
axes[1].set_title("Real anomalies score far above normal orders")

axes[2].imshow([[tn, fp], [fn, tp]], cmap="Blues", vmin=0, vmax=tn * 1.6)
for (i, j), value in {(0, 0): tn, (0, 1): fp, (1, 0): fn, (1, 1): tp}.items():
    axes[2].text(j, i, f"{value:,}", ha="center", va="center", fontsize=18, fontweight="bold",
                 color="white" if value > tn * 0.5 else INK)
axes[2].set_xticks([0, 1], ["Not flagged", "Flagged"]); axes[2].set_yticks([0, 1], ["Normal", "Real anomaly"])
axes[2].grid(False); axes[2].set_title("Flagged orders vs the answer key")

fig.suptitle("Objective 5: how well does the model find planted price anomalies?", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(CHART_DIR / "evaluate_model.png", dpi=200, bbox_inches="tight")
plt.show()

scorecard.to_csv(CHART_DIR / "scorecard.csv")
budget_table.round(2).to_csv(CHART_DIR / "review_budget_table.csv")
rule_check.round(3).to_csv(CHART_DIR / "model_vs_simple_rule.csv")
scored.sort_values("rank").to_csv(CHART_DIR / "all_orders_scored_with_answer_key.csv", index=False)

# ---------- Verification ----------
print(f"Answer key opened from {found}: {n_true} real anomalies in {len(y_true):,} orders ({n_true / len(y_true):.1%})\n")
display(scorecard.round(3))
print(f"\nIf we review exactly {k_true} orders (the true anomaly count), {precision_at_true_count:.0%} of them are real anomalies.")
print("\nCatch rate at different review budgets:")
display(budget_table.round(1))
print("\nIsolation Forest against a one-line rule:")
display(rule_check.round(3))
print("\nPrice level of every order, real anomalies and flags:")
display(band_table)
print("\nWhy the false alarms were flagged:")
display(false_alarm_reasons)
print(f"\nReal anomalies the model missed: {len(miss_table)}")
if len(miss_table):
    display(miss_table)
```

---

## 18. Business Analysis

A model score is only useful if it changes a decision. For each objective, this step asks: **if we review only the riskiest slice, how much of the loss do we find?** Review budgets are 40% of items for objectives 1 to 4 and 5% of orders for objective 5. Losses are measured in the dollars and days recorded in the tables.

| # | Policy | Work created | Share of loss found | Random review would find |
|---|---|---|---:|---:|
| 1 | Review the riskiest cost lines | 40% of lines | 58% of overrun dollars ($35.2M of $60.8M) | 40% |
| 2 | Add buffer and crew to the riskiest tasks | 40% of tasks | 63% of delay days | 40% |
| 3 | Supervise the riskiest incident conditions | 40% of incidents | 83% of lost worker days | 40% |
| 4 | Add buffer to the riskiest orders | 40% of orders | 50% of late days | 40% |
| 5 | Review the highest-scoring orders for price | 5% of orders | 88% of money paid beyond tolerance ($5.34M of $6.10M) | 5% |

**One thing the numbers say about objective 1:** overrun dollars sit in big budget lines. Reviewing the largest 40% of budgets finds 76% of overrun dollars, and model risk alone finds 58%. Use the model to flag risky small lines, and watch the large lines regardless.

**Where to look first (share of loss against share of items)**
- **Cost:** Materials lines are 20% of lines but 41% of overrun dollars.
- **Schedule:** rainy-season tasks are 13% of tasks but 24% of delay days; small crews are 28% of tasks and 39% of delay days.
- **Safety:** height work without PPE is 6% of incidents but 22% of lost days, 3.5 times the average. Falls from height are 20% of incidents and 45% of lost days.
- **Procurement:** Glass Panels are 25% of orders and 30% of late days.

**What-if scenarios (upper bounds, because the data shows a link and cannot prove cause)**

| Scenario | Loss avoided |
|---|---|
| Cost Plus lines overran as rarely as the rest | up to $22.7M of overrun |
| Crews of 18 or fewer slipped as rarely as the rest | up to 2,279 task days |
| Height work without PPE was as safe as the rest | up to 1,023 worker days |
| International orders arrived as reliably as local ones | up to 2,625 delay days |

### Charts

<p align="center">
  <img width="1750" height="598" alt="business_all_objectives" src="https://github.com/user-attachments/assets/6cb29b6a-d07f-49ab-aab6-b309832c857c" />
</p>
<p align="center"><em>All five objectives: share of the loss found by reviewing the riskiest slice, against a random review</em></p>

### Code: business analysis (one cell for all five objectives)

```python
# BUSINESS ANALYSIS - WHAT THE FIVE MODELS ARE WORTH
# Turns model results into money, days and workload. Every assumption is written next to the number it affects.

# %%
from pathlib import Path
import matplotlib.pyplot as plt

OUT_DIR = Path("construction_business")
OUT_DIR.mkdir(exist_ok=True)
BLUE, ORANGE, INK, SOFT, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"

REVIEW_SHARE = 0.40            # assumption: managers can closely review the riskiest 40% of cost lines, tasks, incidents and orders
ANOMALY_REVIEW_SHARE = 0.05    # assumption: procurement can check 5% of orders (the budget used in Objective 5)

models_by_key = {"cost_overrun": models_obj1, "schedule_delay": models_obj2,
                 "site_safety": models_obj3, "late_delivery": models_obj4}

# what "loss" means for each objective, and which column holds it
LOSS = {
    "cost_overrun":   dict(column="overrun_usd", clip=True, unit="USD", what="money spent above budget", item="cost lines"),
    "schedule_delay": dict(column="delay_days", clip=True, unit="days", what="task days lost to delay", item="tasks"),
    "site_safety":    dict(column="lost_days", clip=False, unit="days", what="worker days lost to incidents", item="incidents"),
    "late_delivery":  dict(column="delivery_delay_days", clip=True, unit="days", what="days materials arrived late", item="orders"),
}

def attach_loss(key, ids, loss_column, clip):
    """Look up the loss of each test row from the cleaned source table."""
    cfg = OBJECTIVES[key]
    source = cfg["source"].drop_duplicates(subset=cfg["id_col"]).set_index(cfg["id_col"])
    loss = pd.to_numeric(source.loc[ids, loss_column], errors="coerce").fillna(0).to_numpy()
    return np.clip(loss, 0, None) if clip else loss

# ---------- 1. Objectives 1-4: how much of the loss sits in the riskiest slice of the TEST set? ----------
rows, test_frames = [], {}
for key, spec in LOSS.items():
    pack = models_by_key[key]
    X_test = scaled[key]["X_test"]
    risk = pack["trained"][pack["best_name"]].predict_proba(X_test)[:, 1]
    ids = case_infos[key].loc[X_test.index, OBJECTIVES[key]["id_col"]].to_numpy()
    loss = attach_loss(key, ids, spec["column"], spec["clip"])
    y = splits[key]["y_test"].to_numpy()
    k = int(round(REVIEW_SHARE * len(risk)))
    top = np.argsort(-risk)[:k]
    rows.append({
        "objective": OBJECTIVES[key]["title"], "model": pack["best_name"], "loss measured": f"{spec['what']} ({spec['unit']})",
        "items in test set": len(risk), "total loss in test set": loss.sum(),
        "items reviewed (riskiest 40%)": k,
        "loss inside reviewed items": loss[top].sum(),
        "share of loss captured": loss[top].sum() / loss.sum(),
        "share a random review would capture": REVIEW_SHARE,
        "events caught (share)": y[top].sum() / y.sum(),
    })
    test_frames[key] = pd.DataFrame({"risk": risk, "loss": loss, "event": y})

# ---------- 2. Objective 5: the money at stake in anomalous orders ----------
orders = OBJECTIVES["price_anomalies"]["source"].drop_duplicates(subset="order_id").set_index("order_id")
scores5 = models_obj5["order_scores"].set_index("order_id")
orders = orders.loc[scores5.index]
TOLERANCE = 1.15      # assumption: up to 15% above the material norm is ordinary price variation, not an overcharge
fair_price = orders["unit_price_usd"] / orders["price_ratio_material"]                 # assumption: the material norm is the fair price
excess_paid = (orders["quantity"] * (orders["unit_price_usd"] - TOLERANCE * fair_price)).clip(lower=0).to_numpy()
flagged5 = scores5["flagged"].to_numpy()
k5 = int(round(ANOMALY_REVIEW_SHARE * len(orders)))
rows.append({
    "objective": OBJECTIVES["price_anomalies"]["title"], "model": "Isolation Forest", "loss measured": "money paid beyond 15% above the material norm (USD)",
    "items in test set": len(orders), "total loss in test set": excess_paid.sum(),
    "items reviewed (riskiest 40%)": k5,
    "loss inside reviewed items": excess_paid[flagged5].sum(),
    "share of loss captured": excess_paid[flagged5].sum() / excess_paid.sum(),
    "share a random review would capture": ANOMALY_REVIEW_SHARE,
    "events caught (share)": np.nan,
})
# Objective 1 check: is the model better than simply reviewing the biggest budgets?
o1 = test_frames["cost_overrun"]
o1_source = OBJECTIVES["cost_overrun"]["source"].drop_duplicates(subset="ledger_id").set_index("ledger_id")
o1_budget = o1_source.loc[case_infos["cost_overrun"].loc[scaled["cost_overrun"]["X_test"].index, "ledger_id"], "budgeted_usd"].to_numpy()
k1 = int(round(REVIEW_SHARE * len(o1)))
ranking_rows = []
for label, priority in [("Random review", None), ("Model risk only", o1["risk"].to_numpy()),
                        ("Largest budgets only", o1_budget), ("Model risk x budget", o1["risk"].to_numpy() * o1_budget)]:
    share = REVIEW_SHARE if priority is None else o1["loss"].to_numpy()[np.argsort(-priority)[:k1]].sum() / o1["loss"].sum()
    ranking_rows.append({"how the 40% are chosen": label, "share of overrun dollars found": share})
objective_1_ranking = pd.DataFrame(ranking_rows).set_index("how the 40% are chosen")

summary = pd.DataFrame(rows).set_index("objective")
summary = summary.rename(columns={"items reviewed (riskiest 40%)": "items reviewed"})
summary["capture vs random (times)"] = summary["share of loss captured"] / summary["share a random review would capture"]

# ---------- 3. Where does the loss concentrate? (whole cleaned data, not only the test set) ----------
HOTSPOT_COLUMNS = {
    "cost_overrun": ["contract_type", "country", "cost_category", "project_type"],
    "schedule_delay": ["task_type", "country", "crew_band", "rainy_season"],
    "site_safety": ["incident_type", "shift", "experience_band", "no_ppe_at_height"],
    "late_delivery": ["material", "international_supplier", "country", "rainy_season"],
}
hotspot_rows = []
for key, columns in HOTSPOT_COLUMNS.items():
    cfg, spec = OBJECTIVES[key], LOSS[key]
    frame = cfg["source"].drop_duplicates(subset=cfg["id_col"]).copy()
    frame["_loss"] = pd.to_numeric(frame[spec["column"]], errors="coerce").fillna(0)
    frame["_loss"] = frame["_loss"].clip(lower=0) if spec["clip"] else frame["_loss"]
    average = frame["_loss"].mean()
    for column in columns:
        if column not in frame.columns:
            continue
        grouped = frame.groupby(frame[column].astype("string"), observed=True)["_loss"].agg(["size", "sum", "mean"])
        grouped = grouped[grouped["size"] >= 0.05 * len(frame)]
        for value, row in grouped.iterrows():
            hotspot_rows.append({"objective": cfg["title"], "segment": f"{column} = {value}",
                                 "share of items": row["size"] / len(frame), "share of loss": row["sum"] / frame["_loss"].sum(),
                                 "loss per item": row["mean"], "average loss per item": average,
                                 "times the average": row["mean"] / average if average else np.nan})
hotspots = pd.DataFrame(hotspot_rows)
top_hotspots = (hotspots.sort_values("times the average", ascending=False)
                .groupby("objective", sort=False).head(3).sort_values(["objective", "times the average"], ascending=[True, False]))

# ---------- 4. What-if: what if the worst group behaved like everyone else? (upper bound, correlation is not proof) ----------
def what_if(key, mask_fn, label):
    cfg, spec = OBJECTIVES[key], LOSS[key]
    frame = cfg["source"].drop_duplicates(subset=cfg["id_col"]).dropna(subset=[cfg["target"]]).copy()
    frame["_loss"] = pd.to_numeric(frame[spec["column"]], errors="coerce").fillna(0)
    frame["_loss"] = frame["_loss"].clip(lower=0) if spec["clip"] else frame["_loss"]
    in_group = mask_fn(frame)
    rate_in, rate_out = frame.loc[in_group, cfg["target"]].mean(), frame.loc[~in_group, cfg["target"]].mean()
    events_avoided = max(rate_in - rate_out, 0) * in_group.sum()
    loss_per_event = frame.loc[in_group & (frame[cfg["target"]] == 1), "_loss"].mean()
    return {"objective": cfg["title"], "scenario": label, "group size": int(in_group.sum()),
            "rate in group": rate_in, "rate elsewhere": rate_out, "events avoided": events_avoided,
            "loss avoided": events_avoided * loss_per_event, "loss unit": spec["unit"],
            "share of all events": events_avoided / frame[cfg["target"]].sum()}

crew_cut = OBJECTIVES["schedule_delay"]["source"]["crew_size"].quantile(0.25)
scenarios = pd.DataFrame([
    what_if("cost_overrun", lambda f: f["contract_type"] == "Cost Plus", "Cost Plus lines overrun as rarely as the rest"),
    what_if("schedule_delay", lambda f: f["crew_size"] <= crew_cut, f"Tasks with a crew of {crew_cut:.0f} or fewer are delayed as rarely as the rest"),
    what_if("site_safety", lambda f: f["no_ppe_at_height"] == 1, "Work at height without PPE is as safe as the rest"),
    what_if("late_delivery", lambda f: f["international_supplier"] == 1, "International orders arrive late as rarely as local ones"),
]).set_index("objective")

# ---------- 5. Chart ----------
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.edgecolor": GRID,
                     "xtick.color": SOFT, "ytick.color": SOFT, "axes.labelcolor": SOFT,
                     "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": [1.1, 1]})

labels = [f"Obj {i + 1}: {name.split(': ')[1]}\n(review {'5%' if i == 4 else '40%'} of {'orders' if i >= 3 else ('cost lines' if i == 0 else ('tasks' if i == 1 else 'incidents'))})"
          for i, name in enumerate(summary.index)]
y_pos = np.arange(len(summary))[::-1]
axes[0].barh(y_pos + 0.18, 100 * summary["share of loss captured"], height=0.34, color=BLUE, label="captured by the model's riskiest slice")
axes[0].barh(y_pos - 0.18, 100 * summary["share a random review would capture"], height=0.34, color=GRID, label="a random review would capture")
for pos, value in zip(y_pos + 0.18, 100 * summary["share of loss captured"]):
    axes[0].text(value + 1, pos, f"{value:.0f}%", va="center", fontsize=10, color=INK)
for pos, value in zip(y_pos - 0.18, 100 * summary["share a random review would capture"]):
    axes[0].text(value + 1, pos, f"{value:.0f}%", va="center", fontsize=9, color=SOFT)
axes[0].set_yticks(y_pos, labels); axes[0].set_xlim(0, 112); axes[0].grid(axis="y", visible=False)
axes[0].set_xlabel("Share of the total loss found"); axes[0].set_title("Reviewing only the riskiest slice finds most of the loss")
axes[0].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.45, -0.12), ncol=2, fontsize=9)

best = top_hotspots.groupby("objective", sort=False).head(2).copy()
best["label"] = [f"Obj {int(name.split(':')[0].split()[-1])}: {seg.split(' = ')[0]} = {seg.split(' = ')[1].replace('.0', '')}"
                 for name, seg in zip(best["objective"], best["segment"])]
best = best.iloc[::-1].reset_index(drop=True)
positions = np.arange(len(best))
colors = [ORANGE if v >= 1.5 else BLUE for v in best["times the average"]]
axes[1].barh(positions, best["times the average"], color=colors, height=0.6)
for pos, value in zip(positions, best["times the average"]):
    axes[1].text(value + 0.03, pos, f"{value:.1f}x", va="center", fontsize=9.5, color=INK)
axes[1].set_yticks(positions, best["label"])
axes[1].axvline(1, color=SOFT, linewidth=0.8, linestyle="--")
axes[1].set_xlabel("Loss per item, as a multiple of the average"); axes[1].set_title("Where the loss is concentrated")
axes[1].grid(axis="y", visible=False)

fig.suptitle("What the five models are worth to the business", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT_DIR / "business_analysis.png", dpi=200, bbox_inches="tight")
plt.show()

summary.round(3).to_csv(OUT_DIR / "business_summary.csv")
top_hotspots.round(3).to_csv(OUT_DIR / "loss_hotspots.csv", index=False)
scenarios.round(3).to_csv(OUT_DIR / "what_if_scenarios.csv")
objective_1_ranking.round(3).to_csv(OUT_DIR / "objective_1_ranking_check.csv")

# ---------- Verification ----------
pd.options.display.float_format = "{:,.2f}".format
print("Loss captured by reviewing the riskiest slice (test set for Objectives 1-4, all orders for Objective 5):")
display(summary[["loss measured", "total loss in test set", "items reviewed", "loss inside reviewed items",
                 "share of loss captured", "capture vs random (times)"]])
print("\nObjective 1 check: dollars are concentrated in big budget lines, so what ranking finds the most overrun dollars?")
display((100 * objective_1_ranking).round(1))
print("\nThe three worst segments for each objective (whole cleaned data):")
display(top_hotspots.set_index("objective")[["segment", "share of items", "share of loss", "times the average"]])
print("\nWhat-if scenarios (upper bounds: they assume the gap is caused by the factor, which the data cannot prove):")
display(scenarios[["scenario", "rate in group", "rate elsewhere", "events avoided", "loss avoided", "loss unit", "share of all events"]])
flagged_loss = excess_paid[flagged5].sum()
print(f"\nObjective 5: {int(flagged5.sum())} orders flagged out of {len(orders):,} hold ${flagged_loss:,.0f} of the ${excess_paid.sum():,.0f} "
      f"paid beyond the 15% tolerance ({flagged_loss / excess_paid.sum():.0%}).")
print("Assumptions: the material norm is the fair price; a 40% review budget for Objectives 1-4 and 5% for Objective 5; "
      "losses are measured in the dollars or days recorded in the cleaned tables.")
```

---

## 19. Save Model, Scaler and Features

Each winning model is saved with **four `.joblib` files** (the price-anomaly model has its own set), enough to score new data without the notebook:

| File | What it holds |
|---|---|
| `<objective>_model.joblib` | The trained model |
| `<objective>_scaler.joblib` | The scaler fitted on the training data |
| `<objective>_encoder.joblib` | The encoder that turns text into 0/1 columns (objectives 1 to 4) |
| `<objective>_features.joblib` | Exact input columns and their order, allowed category values, alert cut-off, training size and test results |

| Objective key | Model | Files |
|---|---|---|
| `cost_overrun` | Logistic Regression | model, scaler, encoder, features |
| `schedule_delay` | Logistic Regression | model, scaler, encoder, features |
| `site_safety` | Logistic Regression | model, scaler, encoder, features |
| `late_delivery` | Logistic Regression | model, scaler, encoder, features |
| `price_anomalies` | Isolation Forest | model, scaler, features, price reference tables |

Plus `model_card.joblib`, which records the library versions and test results. In total 21 files.

**Proof that it works:** each objective is reloaded from its saved files alone and has to reproduce the notebook's own predictions. Objectives 1 to 4 match exactly, and objective 5 matches to about 0.001 (training used price ratios rounded to 4 decimals) with the same orders flagged.

### Code: save model, scaler and features

```python
# SAVE MODEL, SCALER AND FEATURES

# %%
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.metrics import roc_auc_score

MODEL_DIR = Path("construction_models")
MODEL_DIR.mkdir(exist_ok=True)

models_by_key = {"cost_overrun": models_obj1, "schedule_delay": models_obj2,
                 "site_safety": models_obj3, "late_delivery": models_obj4}

# Alert cut-offs chosen in each Evaluate step (on training folds only, never on the test set).
ALERT_THRESHOLDS = {"cost_overrun": 0.36, "schedule_delay": 0.28, "site_safety": 0.37, "late_delivery": 0.46}
SAVE_ALL_MODELS = False               # True also saves the Random Forest and Gradient Boosting (about 30 MB more)
UNKNOWN = "Unknown"                    # label given to missing text, same as the encoding step
PRICE_TOLERANCE = 1.15                 # business assumption used in the business analysis
PEER_MIN_ORDERS = 20                   # peer group needs at least 20 orders, else the material median is used

metadata = {"created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
            "libraries": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                          "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
            "objectives": {}}

# ---------- 1. Objectives 1-4: model, scaler, encoder and feature list ----------
for key, pack in models_by_key.items():
    cfg = OBJECTIVES[key]
    best_name = pack["best_name"]
    joblib.dump(pack["trained"][best_name], MODEL_DIR / f"{key}_model.joblib")
    if SAVE_ALL_MODELS:
        joblib.dump(pack["trained"], MODEL_DIR / f"{key}_all_models.joblib")    # every candidate, for comparison
    joblib.dump(scalers[key], MODEL_DIR / f"{key}_scaler.joblib")
    joblib.dump(encoders[key], MODEL_DIR / f"{key}_encoder.joblib")

    X_test, y_test = scaled[key]["X_test"], splits[key]["y_test"]
    test_auc = roc_auc_score(y_test, pack["trained"][best_name].predict_proba(X_test)[:, 1])
    spec = {
        "objective": cfg["title"], "target": cfg["target"], "task": cfg["task"], "best_model": best_name,
        "alert_threshold": ALERT_THRESHOLDS[key],
        "numeric_columns": cfg["numeric"], "categorical_columns": cfg["categorical"],
        "encoded_columns": column_spec[key]["encoded_columns"],
        "scaled_columns": feature_lists[key]["scaled_columns"], "flag_columns": feature_lists[key]["flag_columns"],
        "feature_order": feature_lists[key]["feature_order"],
        "categories": {c: [str(v) for v in levels] for c, levels in zip(cfg["categorical"], encoders[key].categories_)},
        "unknown_text_label": UNKNOWN,
        "training_rows": int(len(splits[key]["X_train"])), "training_positive_rate": round(float(splits[key]["y_train"].mean()), 4),
        "test_roc_auc": round(float(test_auc), 4),
    }
    joblib.dump(spec, MODEL_DIR / f"{key}_features.joblib")
    metadata["objectives"][key] = {k: spec[k] for k in ["objective", "best_model", "alert_threshold", "training_rows",
                                                        "training_positive_rate", "test_roc_auc"]}

# ---------- 2. Objective 5: Isolation Forest, scaler and the price reference tables ----------
key5 = "price_anomalies"
orders = OBJECTIVES[key5]["source"].drop_duplicates(subset="order_id")
material_median = orders.groupby("material")["unit_price_usd"].median()
peer_group = orders.groupby(["material", "country"])["unit_price_usd"].agg(["median", "count"]).reset_index()
peer_reference = peer_group[peer_group["count"] >= PEER_MIN_ORDERS][["material", "country", "median"]]

joblib.dump(models_obj5["model"], MODEL_DIR / f"{key5}_model.joblib")
joblib.dump(scalers[key5], MODEL_DIR / f"{key5}_scaler.joblib")
price_reference = {"material_median": material_median.to_dict(),
                   "peer_median": {(m, c): v for m, c, v in peer_reference.itertuples(index=False)}}
joblib.dump(price_reference, MODEL_DIR / f"{key5}_price_reference.joblib")
spec5 = {
    "objective": OBJECTIVES[key5]["title"], "task": "unsupervised anomaly detection", "best_model": "Isolation Forest",
    "feature_order": models_obj5["feature_order"], "review_share": models_obj5["review_share"],
    "anomaly_score_cutoff": round(float(models_obj5["cutoff_score"]), 6), "price_tolerance": PRICE_TOLERANCE,
    "peer_min_orders": PEER_MIN_ORDERS,
    "how_to_score_new_orders": ["price_ratio_material = unit_price_usd / material median",
                                "price_ratio_peer = unit_price_usd / (material, country) median, else the material median",
                                "log_unit_price = log1p(unit_price_usd), log_quantity = log1p(quantity)",
                                "scale with the saved scaler, then score = -score_samples(); flag when score >= cutoff"],
}
joblib.dump(spec5, MODEL_DIR / f"{key5}_features.joblib")
metadata["objectives"][key5] = {k: spec5[k] for k in ["objective", "best_model", "review_share", "anomaly_score_cutoff"]}
joblib.dump(metadata, MODEL_DIR / "model_card.joblib")

# ---------- 3. Reload everything from disk and prove the saved files score exactly like the notebook ----------
def encode_raw(key, raw, encoder, spec):
    """Raw columns -> encoded columns, in the saved order (what the app will do with a new row)."""
    text = raw[spec["categorical_columns"]].astype("object")
    text = text.where(text.notna(), UNKNOWN)
    dummies = pd.DataFrame(encoder.transform(text), columns=list(encoder.get_feature_names_out(spec["categorical_columns"])), index=raw.index)
    return pd.concat([raw[spec["numeric_columns"]], dummies], axis=1)[[c for c in spec["encoded_columns"] if c in spec["numeric_columns"] or c in dummies.columns]]

check_rows = []
for key, pack in models_by_key.items():
    model = joblib.load(MODEL_DIR / f"{key}_model.joblib")
    scaler = joblib.load(MODEL_DIR / f"{key}_scaler.joblib")
    encoder = joblib.load(MODEL_DIR / f"{key}_encoder.joblib")
    spec = joblib.load(MODEL_DIR / f"{key}_features.joblib")

    X_test = scaled[key]["X_test"]
    raw = abts[key].loc[X_test.index]                                  # the un-encoded rows of the test set
    rebuilt = scaler.transform(encode_raw(key, raw, encoder, spec))   # raw -> encode -> scale, from the saved files
    assert list(rebuilt.columns) == spec["feature_order"], f"{key}: feature order differs"
    same_features = np.allclose(rebuilt.to_numpy(float), X_test.to_numpy(float), atol=1e-9)
    p_saved = model.predict_proba(rebuilt)[:, 1]
    p_notebook = pack["trained"][pack["best_name"]].predict_proba(X_test)[:, 1]
    assert same_features and np.allclose(p_saved, p_notebook), f"{key}: saved pipeline does not reproduce the notebook"
    unseen = raw.head(3).copy()
    unseen[spec["categorical_columns"]] = "Never seen before"
    assert np.isfinite(model.predict_proba(scaler.transform(encode_raw(key, unseen, encoder, spec)))[:, 1]).all(), f"{key}: unseen labels crash"
    check_rows.append({"objective": spec["objective"], "model": spec["best_model"], "rows re-scored": len(raw),
                       "features identical": same_features, "max prediction difference": float(np.abs(p_saved - p_notebook).max()),
                       "unseen labels handled": True})

model5 = joblib.load(MODEL_DIR / f"{key5}_model.joblib")
scaler5 = joblib.load(MODEL_DIR / f"{key5}_scaler.joblib")
spec5_loaded = joblib.load(MODEL_DIR / f"{key5}_features.joblib")
price_ref = joblib.load(MODEL_DIR / f"{key5}_price_reference.joblib")
mat_ref, peer_ref = price_ref["material_median"], price_ref["peer_median"]

def build_price_features(frame):
    """New orders (material, country, unit_price_usd, quantity) -> the four model inputs."""
    material_norm = frame["material"].map(mat_ref)
    peer_norm = pd.Series([peer_ref.get((m, c), np.nan) for m, c in zip(frame["material"], frame["country"])], index=frame.index)
    peer_norm = peer_norm.fillna(material_norm)
    return pd.DataFrame({"price_ratio_material": frame["unit_price_usd"] / material_norm,
                         "price_ratio_peer": frame["unit_price_usd"] / peer_norm,
                         "log_unit_price": np.log1p(frame["unit_price_usd"]),
                         "log_quantity": np.log1p(frame["quantity"])}, index=frame.index)[spec5_loaded["feature_order"]]

new_features = build_price_features(orders.set_index("order_id", drop=False).loc[models_obj5["order_scores"]["order_id"]])
reference = abts[key5].set_index(models_obj5["order_scores"]["order_id"].to_numpy())[spec5_loaded["feature_order"]]
assert np.allclose(new_features.to_numpy(float), reference.to_numpy(float), atol=2e-3, equal_nan=True), "price features differ from the notebook"
scores_saved = -model5.score_samples(scaler5.transform(new_features))
flags_saved = scores_saved >= spec5_loaded["anomaly_score_cutoff"] - 1e-6
flags_notebook = models_obj5["order_scores"]["flagged"].to_numpy()
check_rows.append({"objective": spec5_loaded["objective"], "model": "Isolation Forest", "rows re-scored": len(new_features),
                   "features identical": True, "max prediction difference": float(np.abs(scores_saved - models_obj5["order_scores"]["anomaly_score"].to_numpy()).max()),
                   "unseen labels handled": True})
assert flags_saved.sum() == flags_notebook.sum() and (flags_saved == flags_notebook).mean() > 0.999, "saved Isolation Forest flags differ"

# ---------- Verification ----------
display(pd.DataFrame(check_rows).set_index("objective"))
files = pd.DataFrame([{"file": p.name, "size_kb": round(p.stat().st_size / 1024, 1)} for p in sorted(MODEL_DIR.glob("*"))]).set_index("file")
print(f"\nSaved {len(files)} files to {MODEL_DIR}/:")
display(files)
print("\nObjectives 1-4: the saved pipeline (raw row -> encode -> scale -> model) reproduces the notebook's predictions exactly.")
print("Objective 5: scores match to about 0.001, because training used price ratios rounded to 4 decimals. The same orders are flagged.")
print("Download them in the next step.")
```

---

## 20. Download Model, Scaler and Features

One cell packages all 21 files into `construction_models.zip`, copies it to the Downloads folder, reopens the zip, checks it is not corrupt and loads every file again, so you know nothing was damaged.

```
construction_models.zip
    cost_overrun_model / scaler / encoder / features .joblib
    schedule_delay_...
    site_safety_...
    late_delivery_...
    price_anomalies_model / scaler / features / price_reference .joblib
    model_card.joblib
```

### Code: download model, scaler and features

```python
# ============================================================
# DOWNLOAD MODEL, SCALER AND FEATURES
# ============================================================

# %%
import shutil
import tempfile
import zipfile
from pathlib import Path

import joblib

MODEL_DIR = Path("construction_models")
ZIP_PATH = Path("construction_models.zip")
DOWNLOADS = Path.home() / "Downloads"

assert MODEL_DIR.exists() and any(MODEL_DIR.glob("*.joblib")), "Run the Save step first: construction_models/ is empty."

# ---------- 1. Zip the folder (one file to download, to commit to GitHub, or to upload with the app) ----------
if ZIP_PATH.exists():
    ZIP_PATH.unlink()
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(MODEL_DIR.glob("*.joblib")):
        archive.write(path, arcname=f"{MODEL_DIR.name}/{path.name}")

# ---------- 2. Copy it to your Downloads folder as well ----------
copied_to = None
if DOWNLOADS.exists():
    copied_to = Path(shutil.copy2(ZIP_PATH, DOWNLOADS / ZIP_PATH.name))

# ---------- 3. Check the zip: not corrupt, every file inside, and every file loads again ----------
with zipfile.ZipFile(ZIP_PATH) as archive:
    assert archive.testzip() is None, "the zip file is corrupt"
    names = sorted(Path(n).name for n in archive.namelist())
    expected = sorted(p.name for p in MODEL_DIR.glob("*.joblib"))
    assert names == expected, "the zip does not contain every saved file"
    with tempfile.TemporaryDirectory() as temp:
        archive.extractall(temp)
        loaded = {n: joblib.load(Path(temp) / MODEL_DIR.name / n) for n in names}

OBJECTIVES_SAVED = ["cost_overrun", "schedule_delay", "site_safety", "late_delivery", "price_anomalies"]
contents = pd.DataFrame([{
    "objective": key,
    "model": type(loaded[f"{key}_model.joblib"]).__name__,
    "scaler": type(loaded[f"{key}_scaler.joblib"]).__name__,
    "features": len(loaded[f"{key}_features.joblib"]["feature_order"]),
    "encoder": type(loaded[f"{key}_encoder.joblib"]).__name__ if f"{key}_encoder.joblib" in loaded else "none (numbers only)",
} for key in OBJECTIVES_SAVED]).set_index("objective")

# ---------- 4. A download link inside Jupyter (VS Code ignores it, use the path below instead) ----------
try:
    from IPython.display import FileLink, display as show
    show(FileLink(str(ZIP_PATH), result_html_prefix="Download: "))
except Exception:
    pass

# ---------- Verification ----------
display(contents)
print(f"\n{ZIP_PATH.name}: {len(names)} files, {ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MB, all files load again after unzipping")
print(f"Zip saved here:        {ZIP_PATH.resolve()}")
print(f"Copy in Downloads:     {copied_to if copied_to else 'Downloads folder not found, use the path above'}")
print("\nTo use the models later, unzip and call joblib.load('construction_models/<objective>_model.joblib'), the same for _scaler and _features.")
```

### The Streamlit app

The saved files power the app, [`app.py`](app.py). It has a Home page and one page for each model: **Cost Overruns, Schedule Delays, Site Safety, Material Delivery and Price Anomalies.**

- **Single case:** fill in a short form. The app returns the risk percentage, a Low, Medium or High risk level, a suggested action and the main factors pushing the score up or down.
- **Many cases:** the Price Anomalies page scores a whole CSV of orders, with a template to download and results to download.
- The forms ask plain questions. The app works out the model's derived inputs itself, such as high-rise, budget per sqm, lead time and rainy season.
- The late-delivery page carries a warning that this is the weakest model, and the safety page says it scores an incident that has already happened.


Live app: [Open the Streamlit app](https://construction-project-business-operations-predictive-support-sy.streamlit.app/)

### Code: app.py

```python
"""
CONSTRUCTION BUSINESS OPERATIONS PREDICTIVE SUPPORT SYSTEM
-----------------------------------------------------------
A simple Streamlit app with five models:
  1. Cost overruns         - how likely is a cost line to overrun its budget by more than 15%?
  2. Schedule delays       - how likely is a task to finish more than 5 days late?
  3. Site safety           - how likely is an incident to be serious (lost-time injury or fatality)?
  4. Material procurement  - how likely is a material order to arrive late?
  5. Price anomalies       - is the unit price on an order suspiciously far from the norm?


from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# 1. PAGE SETUP AND SETTINGS
st.set_page_config(page_title="Construction Predictive Support System", layout="wide")

# Find the folder that holds the model files, wherever it sits in the repo
_here = Path(__file__).parent
_found = next(_here.rglob("cost_overrun_model.joblib"), None)
MODEL_DIR = _found.parent if _found else _here / "construction_models"

CLASSIFIERS = ["cost_overrun", "schedule_delay", "site_safety", "late_delivery"]
PAGES = ["Home", "1. Cost overruns", "2. Schedule delays", "3. Site safety",
         "4. Material delivery", "5. Price anomalies"]

# Rainy months by country. This is a business ASSUMPTION used when the models were built.
RAINY_MONTHS = {"Nigeria": [6, 7, 8, 9], "India": [6, 7, 8, 9],
                "Brazil": [12, 1, 2, 3], "United Kingdom": [10, 11, 12, 1]}

PRICE_TOLERANCE = 1.15      # up to 15% above the material norm counts as normal price variation

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# 2. LOAD THE SAVED MODELS (done once, then remembered by Streamlit)
# ==========================================================================

@st.cache_resource
def load_artifacts():
    """Load every model, scaler, encoder and feature file from construction_models/."""
    artifacts = {}
    for key in CLASSIFIERS:
        artifacts[key] = {
            "model": joblib.load(MODEL_DIR / f"{key}_model.joblib"),
            "scaler": joblib.load(MODEL_DIR / f"{key}_scaler.joblib"),
            "encoder": joblib.load(MODEL_DIR / f"{key}_encoder.joblib"),
            "spec": joblib.load(MODEL_DIR / f"{key}_features.joblib"),
        }
    artifacts["price_anomalies"] = {
        "model": joblib.load(MODEL_DIR / "price_anomalies_model.joblib"),
        "scaler": joblib.load(MODEL_DIR / "price_anomalies_scaler.joblib"),
        "spec": joblib.load(MODEL_DIR / "price_anomalies_features.joblib"),
        "reference": joblib.load(MODEL_DIR / "price_anomalies_price_reference.joblib"),
    }
    return artifacts


if not MODEL_DIR.exists():
    st.error(f"Could not find the folder: {MODEL_DIR}\n\n"
             "Put the construction_models folder (the unzipped download) next to app.py.")
    st.stop()

ART = load_artifacts()

# Drop-down lists come straight from what the models learned.
CATEGORIES = ART["cost_overrun"]["spec"]["categories"]
COUNTRIES = CATEGORIES["country"]
PROJECT_TYPES = CATEGORIES["project_type"]
CONTRACT_TYPES = CATEGORIES["contract_type"]
COST_CATEGORIES = CATEGORIES["cost_category"]
TASK_TYPES = ART["schedule_delay"]["spec"]["categories"]["task_type"]
INCIDENT_TYPES = ART["site_safety"]["spec"]["categories"]["incident_type"]
SHIFTS = ART["site_safety"]["spec"]["categories"]["shift"]
WEATHER = ART["site_safety"]["spec"]["categories"]["weather_condition"]
MATERIALS = ART["late_delivery"]["spec"]["categories"]["material"]

# ============================================
# 3. SHARED HELPER FUNCTIONS
# ============================================

def is_rainy(country, month):
    """1 if the month is in the country's rainy season, else 0."""
    return int(month in RAINY_MONTHS.get(country, []))


def readable(column_name):
    """Turn 'contract_type_Cost Plus' into 'contract type = Cost Plus'."""
    for prefix in ["cost_category", "country", "project_type", "contract_type", "task_type",
                   "incident_type", "shift", "weather_condition", "material", "vendor_type"]:
        if column_name.startswith(prefix + "_"):
            return f"{prefix.replace('_', ' ')} = {column_name[len(prefix) + 1:]}"
    return column_name.replace("_", " ")


def score_classifier(key, row):
    """Raw inputs -> encode text -> scale -> chance (0 to 1) and the reasons behind it."""
    art = ART[key]
    spec, encoder, scaler, model = art["spec"], art["encoder"], art["scaler"], art["model"]

    raw = pd.DataFrame([row])
    text = raw[spec["categorical_columns"]].astype("object")
    text = text.where(text.notna(), "Unknown")                         # same rule as in training
    dummy_names = list(encoder.get_feature_names_out(spec["categorical_columns"]))
    dummies = pd.DataFrame(encoder.transform(text), columns=dummy_names, index=raw.index)
    numbers = raw[spec["numeric_columns"]].astype(float)
    encoded = pd.concat([numbers, dummies], axis=1)
    encoded = encoded[[c for c in spec["encoded_columns"] if c in encoded.columns]]

    scaled = scaler.transform(encoded)                                  # missing values are filled here
    chance = float(model.predict_proba(scaled)[0, 1])

    # Logistic Regression: each input pushes the score up or down by (coefficient x value).
    push = pd.Series(model.coef_[0] * scaled.iloc[0].to_numpy(float), index=scaled.columns)
    push = push[push.abs() > 0.05].sort_values()
    return chance, push


def risk_level(chance, cutoff):
    """High at or above the alert cut-off, Medium within 25% below it, otherwise Low."""
    if chance >= cutoff:
        return "High"
    if chance >= 0.75 * cutoff:
        return "Medium"
    return "Low"


def show_result(key, chance, push, what, advice):
    """Draw the result card: chance, risk level, advice and the main reasons."""
    cutoff = ART[key]["spec"]["alert_threshold"]
    level = risk_level(chance, cutoff)

    st.subheader("Result")
    st.metric(f"Chance {what}", f"{chance:.0%}")
    st.progress(min(max(chance, 0.0), 1.0))
    message = f"Risk level: {level}. {advice[level]}"
    {"High": st.error, "Medium": st.warning, "Low": st.success}[level](message)
    st.caption(f"The alert cut-off for this model is {cutoff:.0%}. Scores at or above it are marked High.")

    st.markdown("**What is pushing the score**")
    raising = push[push > 0].sort_values(ascending=False).head(3)
    lowering = push[push < 0].head(3)
    for name, value in raising.items():
        st.write(f"- Raises risk: {readable(name)}")
    for name, value in lowering.items():
        st.write(f"- Lowers risk: {readable(name)}")
    if raising.empty and lowering.empty:
        st.write("- Nothing stands out. This looks like an average case.")


def vendor_inputs(prefix, with_prequalified=True, with_reliability=True):
    """Optional vendor details. Returns None for every value when the vendor is not recorded."""
    known = st.checkbox("The vendor is recorded", value=True, key=prefix + "_known")
    out = {"vendor_known": int(known), "reliability_rating": np.nan, "vendor_safety_rating": np.nan,
           "vendor_years_active": np.nan, "is_prequalified": np.nan}
    if known:
        if with_reliability:
            out["reliability_rating"] = st.slider("Vendor reliability rating (1-5)", 1, 5, 4, key=prefix + "_rel")
        out["vendor_safety_rating"] = st.slider("Vendor safety rating (1-5)", 1, 5, 4, key=prefix + "_saf")
        out["vendor_years_active"] = st.number_input("Vendor years active", 1, 60, 20, key=prefix + "_yrs")
        if with_prequalified:
            out["is_prequalified"] = int(st.checkbox("Vendor is prequalified", value=True, key=prefix + "_pre"))
    return out

# ====================================
# 4. PAGE: HOME
# ====================================

def page_home():
    st.title("Construction Business Operations Predictive Support System")
    st.write("Five models that help managers decide where to look first: cost, schedule, safety, "
             "material delivery and supplier pricing. Pick a model in the sidebar, enter the details "
             "and read the result.")

    rows = []
    for key, name, what in [("cost_overrun", "1. Cost overruns", "Cost line will overrun its budget by more than 15%"),
                            ("schedule_delay", "2. Schedule delays", "Task will finish more than 5 days late"),
                            ("site_safety", "3. Site safety", "Incident will be serious (lost-time injury or fatality)"),
                            ("late_delivery", "4. Material delivery", "Material order will arrive late")]:
        spec = ART[key]["spec"]
        rows.append({"Model": name, "Predicts": what, "Algorithm": spec["best_model"],
                     "Test ROC AUC": spec["test_roc_auc"], "Alert cut-off": spec["alert_threshold"]})
    rows.append({"Model": "5. Price anomalies", "Predicts": "Order price is far from the material norm",
                 "Algorithm": "Isolation Forest", "Test ROC AUC": None,
                 "Alert cut-off": round(ART["price_anomalies"]["spec"]["anomaly_score_cutoff"], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("ROC AUC: 0.5 is a coin flip and 1.0 is perfect. The Isolation Forest has no ROC AUC because "
               "it learns what a normal order looks like and needs no labels.")

    st.subheader("How to use it")
    st.write("1. Choose a model in the sidebar.  2. Fill in the details.  3. Read the chance, the risk level "
             "and the main reasons.  4. Follow the suggested action.")

    st.subheader("Please keep in mind")
    st.write("- The scores rank risk to help you decide where to look first. They are not certainties.")
    st.write("- The late-delivery model is the weakest (ROC AUC about 0.60). Use it as a rough guide only.")
    st.write("- The rainy-season calendar and the 15% price tolerance are assumptions made when the models were built.")
    st.write("- The models learned from 80 projects. Treat results for very unusual projects with caution.")

# ======================================
# 5. PAGE 1: COST OVERRUNS
# ======================================

def page_cost_overrun():
    st.header("1. Cost overruns")
    st.write("Will this cost line overrun its budget by more than 15%?")
    left, right = st.columns([3, 2])
    with left:
        a, b = st.columns(2)
        with a:
            cost_category = st.selectbox("Cost category", COST_CATEGORIES, index=COST_CATEGORIES.index("Materials"))
            contract_type = st.selectbox("Contract type", CONTRACT_TYPES, index=CONTRACT_TYPES.index("Fixed Price"))
            country = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("United States"))
            project_type = st.selectbox("Project type", PROJECT_TYPES)
            line_budget = st.number_input("Budget for this cost line (USD)", 10_000, 20_000_000, 500_000, step=10_000)
            share = st.number_input("This line's share of the project's cost budget (%)", 0.1, 30.0, 3.0, step=0.1)
        with b:
            project_budget = st.number_input("Total project budget (USD)", 500_000, 500_000_000, 19_000_000, step=500_000)
            site_area = st.number_input("Site area (sqm)", 500, 200_000, 30_000, step=500)
            floors = st.number_input("Number of floors", 1, 120, 30)
            planned_days = st.number_input("Planned project duration (days)", 60, 3000, 680, step=10)
            manager_years = st.number_input("Project manager experience (years)", 0, 45, 14)
            year = st.selectbox("Cost period year", list(range(2022, 2028)), index=3)
            quarter = st.selectbox("Cost period quarter", [1, 2, 3, 4])
            is_public = st.checkbox("Public sector client", value=False)

    row = {"cost_category": cost_category, "country": country, "project_type": project_type,
           "contract_type": contract_type, "log_budgeted_usd": np.log1p(line_budget),
           "budget_share_of_project": share / 100, "period_year": year, "period_quarter": quarter,
           "project_budget_usd": project_budget, "site_area_sqm": site_area, "num_floors": floors,
           "project_planned_days": planned_days, "manager_experience_years": manager_years,
           "high_rise": int(floors >= 30), "is_public": int(is_public),
           "budget_per_sqm": project_budget / site_area}
    chance, push = score_classifier("cost_overrun", row)
    with right:
        show_result("cost_overrun", chance, push, "of a cost overrun above 15%", {
            "High": "Review this line closely: check supplier quotes, change orders and spend to date.",
            "Medium": "Keep an eye on it and check the next cost report.",
            "Low": "Normal monitoring is enough."})

# ========================================
=# 6. PAGE 2: SCHEDULE DELAYS
# ========================================
def page_schedule_delay():
    st.header("2. Schedule delays")
    st.write("Will this task finish more than 5 days late?")
    left, right = st.columns([3, 2])
    with left:
        a, b = st.columns(2)
        with a:
            task_type = st.selectbox("Task type", TASK_TYPES, index=TASK_TYPES.index("Structural Frame"))
            country = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("Nigeria"))
            project_type = st.selectbox("Project type", PROJECT_TYPES)
            contract_type = st.selectbox("Contract type", CONTRACT_TYPES, index=CONTRACT_TYPES.index("Fixed Price"))
            start = st.date_input("Planned start date", value=date(2025, 3, 1))
            end = st.date_input("Planned end date", value=date(2025, 3, 1) + timedelta(days=47))
            crew = st.number_input("Crew size", 1, 150, 32)
        with b:
            planned_days = st.number_input("Planned project duration (days)", 60, 3000, 680, step=10)
            floors = st.number_input("Number of floors", 1, 120, 30)
            site_area = st.number_input("Site area (sqm)", 500, 200_000, 30_000, step=500)
            manager_years = st.number_input("Project manager experience (years)", 0, 45, 14)
            vendor = vendor_inputs("sched")

    task_days = (end - start).days
    if task_days <= 0:
        with right:
            st.error("The planned end date must be after the planned start date.")
        return

    row = {"task_type": task_type, "country": country, "project_type": project_type, "contract_type": contract_type,
           "task_planned_days": task_days, "crew_size": crew, "planned_start_month": start.month,
           "planned_start_year": start.year, "rainy_season": is_rainy(country, start.month),
           "project_planned_days": planned_days, "num_floors": floors, "site_area_sqm": site_area,
           "manager_experience_years": manager_years, **vendor}
    chance, push = score_classifier("schedule_delay", row)
    with right:
        if row["rainy_season"]:
            st.info(f"The task starts in the rainy season for {country}.")
        show_result("schedule_delay", chance, push, "of finishing more than 5 days late", {
            "High": "Add crew or buffer days, or move the start date if you can.",
            "Medium": "Confirm crew availability and weather cover before the start.",
            "Low": "Normal planning is enough."})

# ==========================================
# 7. PAGE 3: SITE SAFETY
# ==========================================

def page_site_safety():
    st.header("3. Site safety")
    st.write("If an incident happens under these conditions, how likely is it to be serious "
             "(a lost-time injury or fatality)?")
    left, right = st.columns([3, 2])
    with left:
        a, b = st.columns(2)
        with a:
            incident_type = st.selectbox("Incident type", INCIDENT_TYPES, index=INCIDENT_TYPES.index("Fall from Height"))
            shift = st.selectbox("Shift", SHIFTS)
            weather = st.selectbox("Weather", WEATHER)
            country = st.selectbox("Country", COUNTRIES, index=COUNTRIES.index("Nigeria"))
            project_type = st.selectbox("Project type", PROJECT_TYPES)
        with b:
            experience = st.number_input("Worker experience (years)", 0, 45, 10)
            ppe = st.radio("PPE compliant?", ["Yes", "No"], index=0, horizontal=True)
            height = st.radio("Working at height?", ["Yes", "No"], index=1, horizontal=True)
            month = st.selectbox("Month of the incident", list(range(1, 13)))
            floors = st.number_input("Number of floors", 1, 120, 30)
            vendor = vendor_inputs("safety", with_prequalified=False, with_reliability=False)

    no_ppe_at_height = int(ppe == "No" and height == "Yes")
    row = {"incident_type": incident_type, "shift": shift, "weather_condition": weather,
           "project_type": project_type, "country": country, "worker_experience_years": experience,
           "ppe_compliant_yes": int(ppe == "Yes"), "work_at_height_yes": int(height == "Yes"),
           "no_ppe_at_height": no_ppe_at_height, "incident_month": month,
           "vendor_safety_rating": vendor["vendor_safety_rating"], "vendor_years_active": vendor["vendor_years_active"],
           "vendor_known": vendor["vendor_known"], "num_floors": floors, "high_rise": int(floors >= 30)}
    chance, push = score_classifier("site_safety", row)
    with right:
        if no_ppe_at_height:
            st.warning("Working at height without PPE is the most dangerous condition in the data.")
        show_result("site_safety", chance, push, "that the incident is serious", {
            "High": "Stop and fix the conditions: enforce PPE, add supervision and check the work method.",
            "Medium": "Add a supervisor check and a toolbox talk before work continues.",
            "Low": "Keep to normal safety routines."})

# ================================================
# 8. PAGE 4: MATERIAL DELIVERY
# ================================================

def page_late_delivery():
    st.header("4. Material delivery")
    st.write("Will this material order arrive after its promised date? "
             "(This is the weakest model. Use it as a rough guide.)")
    left, right = st.columns([3, 2])
    with left:
        a, b = st.columns(2)
        with a:
            material = st.selectbox("Material", MATERIALS)
            country = st.selectbox("Project country", COUNTRIES, index=COUNTRIES.index("Nigeria"))
            project_type = st.selectbox("Project type", PROJECT_TYPES)
            quantity = st.number_input("Quantity ordered", 1, 100_000, 150)
            price = st.number_input("Unit price (USD)", 0.5, 20_000.0, 25.0, step=0.5)
            order_date = st.date_input("Order date", value=date(2025, 3, 1))
            promised_date = st.date_input("Promised delivery date", value=date(2025, 3, 1) + timedelta(days=25))
        with b:
            known = st.checkbox("The supplier is recorded", value=True, key="del_known")
            supplier_country, reliability, safety, years, prequalified = None, np.nan, np.nan, np.nan, np.nan
            if known:
                supplier_country = st.selectbox("Supplier country", COUNTRIES, index=COUNTRIES.index("Nigeria"))
                reliability = st.slider("Supplier reliability rating (1-5)", 1, 5, 4, key="del_rel")
                safety = st.slider("Supplier safety rating (1-5)", 1, 5, 4, key="del_saf")
                years = st.number_input("Supplier years active", 1, 60, 20, key="del_yrs")
                prequalified = int(st.checkbox("Supplier is prequalified", value=True, key="del_pre"))

    lead_days = (promised_date - order_date).days
    if lead_days <= 0:
        with right:
            st.error("The promised date must be after the order date.")
        return

    international = np.nan if not known else int(supplier_country != country)
    row = {"material": material, "vendor_type": "Supplier" if known else "Unknown", "country": country,
           "project_type": project_type, "log_quantity": np.log1p(quantity), "unit_price_usd": price,
           "lead_time_days": lead_days, "order_month": order_date.month, "order_year": order_date.year,
           "rainy_season": is_rainy(country, promised_date.month), "international_supplier": international,
           "reliability_rating": reliability, "vendor_safety_rating": safety, "vendor_years_active": years,
           "is_prequalified": prequalified}
    chance, push = score_classifier("late_delivery", row)
    with right:
        if international == 1:
            st.info("The supplier is in a different country from the project (international order).")
        show_result("late_delivery", chance, push, "that the order arrives late", {
            "High": "Add buffer time, confirm shipping early, or consider a local supplier.",
            "Medium": "Ask the supplier to confirm the date a week before it is due.",
            "Low": "Normal order follow-up is enough."})

# =================================================
# 9. PAGE 5: PRICE ANOMALIES
# =================================================

def price_features(frame):
    """material, country, unit_price_usd, quantity -> the four inputs the Isolation Forest uses."""
    reference = ART["price_anomalies"]["reference"]
    norm = frame["material"].map(reference["material_median"])
    peer = pd.Series([reference["peer_median"].get((m, c), np.nan)
                      for m, c in zip(frame["material"], frame["country"])], index=frame.index).fillna(norm)
    out = pd.DataFrame({"price_ratio_material": frame["unit_price_usd"] / norm,
                        "price_ratio_peer": frame["unit_price_usd"] / peer,
                        "log_unit_price": np.log1p(frame["unit_price_usd"]),
                        "log_quantity": np.log1p(frame["quantity"])}, index=frame.index)
    return out[ART["price_anomalies"]["spec"]["feature_order"]], norm


def score_prices(frame):
    """Add the typical price, the price ratio, an anomaly score and a verdict to a table of orders."""
    art = ART["price_anomalies"]
    features, norm = price_features(frame)
    score = -art["model"].score_samples(art["scaler"].transform(features))
    cutoff = art["spec"]["anomaly_score_cutoff"]
    result = frame.copy()
    result["typical_price_usd"] = norm.round(2)
    result["price_vs_typical"] = (frame["unit_price_usd"] / norm).round(2)
    result["anomaly_score"] = np.round(score, 3)
    result["verdict"] = np.where(score >= cutoff, "Review", np.where(score >= 0.9 * cutoff, "Watch", "Normal"))
    result["possible_overpayment_usd"] = (frame["quantity"] * (frame["unit_price_usd"] - PRICE_TOLERANCE * norm)).clip(lower=0).round(0)
    return result


def page_price_anomalies():
    st.header("5. Price anomalies")
    st.write("Is the unit price on this order far from what is normal for the material?")
    left, right = st.columns([3, 2])
    with left:
        material = st.selectbox("Material", MATERIALS)
        country = st.selectbox("Project country", COUNTRIES, index=COUNTRIES.index("Nigeria"))
        price = st.number_input("Unit price (USD)", 0.5, 50_000.0, 25.0, step=0.5)
        quantity = st.number_input("Quantity ordered", 1, 100_000, 150)

    one = score_prices(pd.DataFrame([{"material": material, "country": country,
                                      "unit_price_usd": price, "quantity": quantity}])).iloc[0]
    with right:
        st.subheader("Result")
        st.metric("Typical price for this material", f"${one['typical_price_usd']:,.2f}")
        st.metric("This order's price compared with typical", f"{one['price_vs_typical']:.2f} times")
        message = {"Review": "Review this order before paying. The price is unusual for the material.",
                   "Watch": "Borderline. Worth a quick look.",
                   "Normal": "The price looks normal."}[one["verdict"]]
        {"Review": st.error, "Watch": st.warning, "Normal": st.success}[one["verdict"]](f"{one['verdict']}: {message}")
        if one["possible_overpayment_usd"] > 0:
            st.write(f"Possible overpayment beyond the 15% tolerance: **${one['possible_overpayment_usd']:,.0f}**")
        st.caption(f"Anomaly score {one['anomaly_score']:.3f}. Orders at or above "
                   f"{ART['price_anomalies']['spec']['anomaly_score_cutoff']:.3f} are marked Review.")

    st.subheader("Check many orders at once")
    st.write("Upload a CSV file with the columns: material, country, unit_price_usd, quantity "
             "(an order_id column is optional).")
    template = "order_id,material,country,unit_price_usd,quantity\nORD-1,Cement,Nigeria,8.5,200\nORD-2,Steel Rebar,Nigeria,4200,120\n"
    st.download_button("Download a template", template, file_name="orders_template.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload orders (CSV)", type="csv")
    if uploaded is not None:
        orders = pd.read_csv(uploaded)
        needed = ["material", "country", "unit_price_usd", "quantity"]
        missing = [c for c in needed if c not in orders.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        elif not orders["material"].isin(MATERIALS).all():
            st.error("Some materials are not recognised. Use: " + ", ".join(MATERIALS))
        else:
            scored = score_prices(orders).sort_values("anomaly_score", ascending=False)
            st.write(f"{(scored['verdict'] == 'Review').sum()} of {len(scored)} orders need review.")
            st.dataframe(scored, hide_index=True, use_container_width=True)
            st.download_button("Download results", scored.to_csv(index=False), file_name="scored_orders.csv", mime="text/csv")

# =========================================================
# 10. SIDEBAR MENU AND PAGE ROUTER
# =========================================================

st.sidebar.title("Construction Predictive Support System")
choice = st.sidebar.radio("Choose a model", PAGES)
st.sidebar.caption("A decision-support tool. Scores help you decide where to look first. "
                   "They are estimates, not certainties.")

if choice == "Home":
    page_home()
elif choice == "1. Cost overruns":
    page_cost_overrun()
elif choice == "2. Schedule delays":
    page_schedule_delay()
elif choice == "3. Site safety":
    page_site_safety()
elif choice == "4. Material delivery":
    page_late_delivery()
else:
    page_price_anomalies()

```

---

## 21. Key Findings and Insights

**Objective 1: cost overruns**
- The model ranks risk well: in the highest-risk band 92% of lines really overran, in the lowest only 11%.
- Contract type, country and cost category matter most. Cost Plus lines overran 66% of the time against 36% for Fixed Price, and Materials lines 69% against 21% for Overheads.
- Dollars follow budget size: the largest 40% of budgets hold 76% of overrun dollars.

**Objective 2: schedule delays**
- Understaffed tasks slip far more often: crews of 18 or fewer were delayed 60% of the time, the rest 34%.
- The rainy season roughly doubles the delay risk (75% against 36%).
- The model catches 88% of delays at its alert cut-off, at the price of 50% false alarms.

**Objective 3: site safety**
- Height work without PPE is the most dangerous condition: 59% of those incidents were serious, against 23% otherwise.
- Falls from height are the serious ones (52%, against 4% of slips and trips). Experience protects: 41% serious for the least experienced workers against 9% for the most. Night shifts run at 38% against 21% on days.
- No incident in the lowest-risk band was serious.

**Objective 4: late delivery**
- This is hard to predict from the data we hold. ROC AUC is only 0.60, and the riskiest 40% of orders hold just 50% of late days.
- International orders arrive late more often (48% against 29% for local orders). That is the main lever.

**Objective 5: price anomalies**
- A small set of orders carries most of the overpricing: the 5% flagged hold 88% of the money paid beyond tolerance.
- The Isolation Forest found all 62 planted anomalies, but a one-line rule (price above 2 times the material norm) found all 62 with 62 flags too. The rule is the place to start.

**Across all five**
- **The simplest model won on training evidence every time.** Logistic Regression was picked for all four classification objectives. On the test set the other models were never clearly ahead: Random Forest was higher by 0.001 on objective 2 and 0.015 on objective 4, which is within noise for 16 projects.
- **Leakage was the biggest risk.** Actual cost, actual dates, severity, lost days, delivery dates and the answer key were excluded, so results reflect what is known in advance.
- **Models rank risk; they do not predict it exactly.** At the alert cut-offs, only about half to two thirds of flagged cases were real problems. Use the scores to decide where to look first.

---

## 22. Recommendations

1. **Enforce PPE on every job at height.** Make PPE sign-off a start condition for any height task and spot-check weekly. Height work without PPE accounts for 22% of lost worker days from 6% of incidents.
2. **Set a minimum crew size for critical tasks.** Crews of 18 or fewer are the high-risk group, delayed 60% of the time against 34%.
3. **Check invoices against the material norm before payment.** Start with the one-line rule (price above 2 times the norm), then review the model's highest-scoring orders each cycle. The top 5% by score hold 88% of the money paid beyond tolerance.
4. **Add cost-control incentives to Cost Plus contracts.** Caps, shared-saving clauses or milestone cost checks, starting with the biggest contracts. Cost Plus lines overran 66% of the time.
5. **Run a weekly overrun review on the largest budget lines**, topped up by the model for risky smaller lines (anything above the 0.36 alert cut-off).
6. **Add rainy-season buffers and crew cover to schedules.** Plan outdoor and structural tasks for the dry season where possible.
7. **Control Materials prices**, mainly in Nigeria and India. Use framework agreements or price caps for key materials, and re-quote when prices move.
8. **Buffer or localise material deliveries.** Add lead-time buffer for international orders and prefer local or pre-qualified suppliers for critical materials.
9. **Add supervision on night shifts and pair new workers with experienced ones.** Serious incidents run at 38% on nights and 41% for the least experienced.
10. **Use the models as ranking tools and retrain every quarter.** Do not use the late-delivery model for decisions until better delivery data is collected.
11. **Improve the data at the source.** Standardise dates, currency and units in the entry forms, and add carrier, port and lead-time fields to orders.

---

## 23. Conclusion

I built five models for five construction problems, compared three algorithms plus a baseline for each classification objective, chose the winners on training evidence and confirmed them on projects the models had never seen. Logistic Regression solved cost overruns, schedule delays, serious safety incidents and late deliveries, and an Isolation Forest handled price anomalies. All five are saved with their scalers, encoders and feature definitions, reload and reproduce the notebook's results, and run in a Streamlit app that scores one case at a time, or a whole file of orders.

Used as a ranking tool, the models point review effort at a minority of cases that hold most of the loss. In the test data, reviewing the riskiest 40% finds 58% of overrun dollars, 63% of delay days and 83% of lost safety days, against 40% for a random review. For pricing, reviewing 5% of orders finds 88% of the money paid beyond tolerance, and a transparent one-line rule matched the model, a reminder to test a simple baseline before choosing a complex one. Late delivery is the exception: with the data available, it is only weakly predictable, and better delivery data is the way to improve it.

**Limits to keep in mind**
- Each test set holds only 16 unseen projects, so exact scores can move. The rankings are more reliable than single numbers.
- Group differences show a link, not proof of cause, so what-if figures are upper bounds.
- The rainy-season calendar and the 15% price tolerance are assumptions.
- Price anomalies are measured against planted anomalies, which are easier to find than subtle real ones.
- The data is fictional, so the findings demonstrate the method rather than the performance of a real contractor.

**Next steps:** retrain quarterly on fresh data, collect better delivery data (carrier, port, supplier lead times), pilot the review policies next to current practice and measure the difference.

---

**Author:** Adewale Oshinukan | Data Scientist | [GitHub](https://github.com/waleoshin75-sketch/construction-project-business-operations-predictive-support-system) | [Live App](https://construction-project-business-operations-predictive-support-sy.streamlit.app/)
