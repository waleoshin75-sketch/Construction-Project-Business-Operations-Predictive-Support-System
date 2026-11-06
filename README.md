# Construction Project Business Operations: Predictive Decision Support System

### An Intelligent Multi-Model Engine for Financial, Timelines and Operational Risks Mitigation

[![Live App](https://img.shields.io/badge/Live%20App-Open%20on%20Streamlit-ff4b4b?logo=streamlit&logoColor=white)](https://construction-project-business-operations-predictive-support-sy.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.x-3776ab?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Logistic%20Regression%20%7C%20Isolation%20Forest-f7931e?logo=scikit-learn&logoColor=white)

**[Try the live app](https://construction-project-business-operations-predictive-support-sy.streamlit.app/)**: enter a cost line, task, incident or material order and get a risk score in seconds.

---

## At a glance

Five models, one app. Each one tells a manager where to look first.

| # | Business objective | Model | Test ROC AUC | What it finds |
|---|--------------------|-------|:-----------:|---------------|
| 1 | Control cost overruns | Logistic Regression | **0.83** | Cost lines likely to overrun by more than 15% |
| 2 | Reduce schedule delays | Logistic Regression | **0.80** | Tasks likely to finish more than 5 days late |
| 3 | Improve site safety | Logistic Regression | **0.86** | Incidents likely to be serious (lost-time injury or fatality) |
| 4 | Make procurement reliable | Logistic Regression | 0.60 | Orders likely to arrive late (weak, used as a rough guide) |
| 5 | Detect price anomalies | Isolation Forest | n/a | Orders priced far from the material norm |

**What it is worth:** reviewing only the riskiest 40% of items finds **58%** of overrun dollars, **63%** of delay days and **83%** of lost safety days, against 40% for a random review. Reviewing 5% of orders for price finds **88%** of the money paid above the norm.

> **How to read this:** each phase shows its full, runnable Python code, followed by the **Output** it produced (printed results, tables and charts), in the order it was run.

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
11. [Separate X and Y](#11-separate-x-and-y)
12. [Train Test Split](#12-train-test-split)
13. [Feature Scaling of Numerical Columns](#13-feature-scaling-of-numerical-columns)
14. [Train Model](#14-train-model)
15. [Predict X_test](#15-predict-x_test)
16. [Compare X and Y](#16-compare-x-and-y)
17. [Evaluate Model](#17-evaluate-model)
18. [Business Analysis](#18-business-analysis)
19. [Save Model, Scaler and Features](#19-save-model-scaler-and-features)
20. [Download Model, Scaler and Features](#20-download-model-scaler-and-features)
21. [Key Findings and Insights](#21-key-findings-and-insights)
22. [Recommendations](#22-recommendations)
23. [Conclusion](#23-conclusion)
24. [Run It Yourself](#24-run-it-yourself)

---

## 1. Project Overview

Construction problems are expensive because they show up late: a cost line that quietly overruns, a task that slips, an incident that turns serious, a delivery that never arrives, an invoice with a padded price.

This project predicts all five before they become losses, and puts the models in a Streamlit app that a project manager can use without writing code.

**The five business objectives**

1. Control project cost overruns
2. Reduce schedule delays
3. Improve site safety (serious-incident triage)
4. Make material procurement more reliable (late delivery)
5. Detect procurement price and invoice anomalies

**Built with:** Python, pandas, NumPy, scikit-learn, matplotlib, joblib, Streamlit.

---

## 2. Dataset Overview

Six tables in a star layout (two dimension tables, four fact tables), plus an answer key used only to check Objective 5 at the very end.

| Table | Type | Raw rows | Clean rows | Used for |
|-------|------|---------:|-----------:|----------|
| `dim_projects` | Dimension | 84 | 80 | Project size, type, contract, country |
| `dim_vendors` | Dimension | 48 | 45 | Supplier and subcontractor ratings |
| `fact_cost_ledger` | Fact | 2,465 | 2,413 | Objective 1 |
| `fact_tasks` | Fact | 4,045 | 3,985 | Objective 2 |
| `fact_safety_incidents` | Fact | 768 | 750 | Objective 3 |
| `fact_material_orders` | Fact | 3,035 | 3,000 | Objectives 4 and 5 |

### Code: load the six tables

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = "construction_raw"

# Load and display Projects dimension data
df_projects = pd.read_csv(f"{DATA_DIR}/dim_projects.csv")
df_projects.head(5)

# Load and display Vendors dimension data
df_vendors = pd.read_csv(f"{DATA_DIR}/dim_vendors.csv")
df_vendors.head(5)

# Load and display Cost Ledger fact data
df_cost_ledger = pd.read_csv(f"{DATA_DIR}/fact_cost_ledger.csv")
df_cost_ledger.head(5)

# Load and display Material Orders fact data
df_material_orders = pd.read_csv(f"{DATA_DIR}/fact_material_orders.csv")
df_material_orders.head(5)

# Load and display Safety Incidents fact data
df_safety_incidents = pd.read_csv(f"{DATA_DIR}/fact_safety_incidents.csv")
df_safety_incidents.head(5)

# Load and display Tasks fact data
df_tasks = pd.read_csv(f"{DATA_DIR}/fact_tasks.csv")
df_tasks.head(5)
```

#### Output

| project_id   | project_name     | country        | project_type   | contract_type   | client_sector   | budget_usd   | start_date   | planned_duration_days   | site_area_sqm   | num_floors   | project_manager_experience_years   |
|:-------------|:-----------------|:---------------|:---------------|:----------------|:----------------|:-------------|:-------------|:------------------------|:----------------|:-------------|:-----------------------------------|
| PRJ-1000     | Skyline Bridge 0 | germany        | RESIDENTIAL    | Design-Build    | Private         | 23109000.0   | 2024-12-13   | 801                     | 37173           | 7            | 20                                 |
| PRJ-1001     | Summit Complex 1 | South Africa   | Infrastructure | Cost Plus       | Public          | $22,825,000  | 2022-11-06   | 903                     | 48882           |              | 28                                 |
| PRJ-1002     | SKYLINE PLAZA 2  | UAE            | Commercial     | Fixed Price     | Public          | 12456000.0   | 04-23-2022   | 896                     | 6263            | 59           | 4                                  |
| PRJ-1003     | Coral Complex 3  | NIGERIA        | Commercial     | Cost Plus       | Private         | 13.5M        | 2024-01-10   | 828                     | 30159           | 29           | 18                                 |
| PRJ-1004     | Summit Plant 4   | United Kingdom | COMMERCIAL     | Design-Build    | Private         | 31510000.0   | 2022-03-07   | 745                     | 45587           | 49           | 9                                  |

| vendor_id   | vendor_name         | vendor_type   | specialty      | country        | reliability_rating   | safety_rating   | years_active   | prequalified   |
|:------------|:--------------------|:--------------|:---------------|:---------------|:---------------------|:----------------|:---------------|:---------------|
| VND-100     | Atlas Materials Ltd | Subcontractor | Steel Erection | United Kingdom | 3                    | 3               | 22             | no             |
| VND-101     | Vertex Engineering  | Subcontractor | Civil          | Brazil         | 3                    | 4               | 24             | Yes            |
| VND-102     | Orion Industries    | Subcontractor | Civil          | uae            | 4                    | 4               | 30             | No             |
| VND-103     | Crest Works         | Subcontractor | Civil          | RSA            | 4                    | 4               | 21             | yes            |
| VND-104     | Crest Materials Ltd | Subcontractor | Steel Erection | india          |                      | 4               | 9              | TRUE           |

| ledger_id   | project_id   | period   | cost_category   | budgeted_usd   | actual_usd     | approved_change_orders_usd   |
|:------------|:-------------|:---------|:----------------|:---------------|:---------------|:-----------------------------|
| LED-60000   | PRJ-1000     | 2024-Q4  | Materials       | 967565.71      | 1015634.31     | 29026.97                     |
| LED-60001   | PRJ-1000     | 2024 Q4  | Labour          | 634323.05      | 678910.53      | 19029.69                     |
| LED-60002   | PRJ-1000     | 2024-Q4  | EQUIPMENT       | 330922.16      | 387659.2       | 0.0                          |
| LED-60003   | PRJ-1000     | 2024 Q4  | subcontractors  | 680969.68      | USD 767,379.70 | USD 0.00                     |
| LED-60004   | PRJ-1000     | 2024-Q4  | Overheads       | $214,773.49    | 241935.23      | 6443.2                       |

| order_id   | project_id   | vendor_id   | material            | quantity   | unit   | unit_price_usd   | total_cost_usd   | order_date   | promised_date   | delivered_date   | quality_passed   |
|:-----------|:-------------|:------------|:--------------------|:-----------|:-------|:-----------------|:-----------------|:-------------|:----------------|:-----------------|:-----------------|
| ORD-40000  | PRJ-1070     | VND-131     | Glass Panels        | 70         | sheets | 153.99           | $10,779.30       | 2023-07-30   | 2023-09-07      | 2023-09-06       | Yes              |
| ORD-40001  | PRJ-1048     | VND-140     | Electrical Fittings | 385        | pieces | 25.02            | 9632.7           | 2026-01-02   | 2026-02-03      | 2026-02-01       | 1                |
| ORD-40002  | PRJ-1038     | VND-135     | Glass Panels        | 147        | sheets | 108.89           | USD 16,006.83    | 2025-04-26   | 2025-05-08      | 2025-05-12       | yes              |
| ORD-40003  | PRJ-1055     | VND-140     | Electrical Fittings | 127        | pieces | 26.28            | 3337.56          | 2022-05-27   | 06-29-2022      | 2022-06-27       | 1                |
| ORD-40004  | PRJ-1056     | VND-135     | Glass Panels        | 38         | sheets | USD 109.76       | 4170.88          | 14 Oct 2025  | ??              | 2025-11-22       | TRUE             |

| incident_id   | project_id   | vendor_id   | incident_date   | incident_type      | shift   | ppe_compliant   | work_at_height   | worker_experience_years   | weather_condition   | severity         | lost_days   | reported_delay_hours   |
|:--------------|:-------------|:------------|:----------------|:-------------------|:--------|:----------------|:-----------------|:--------------------------|:--------------------|:-----------------|:------------|:-----------------------|
| INC-80000     | PRJ-1000     | VND-104     | 2024-04-06      | Slip or Trip       | Night   | no              | No               | 12                        | Clear               | Fatality         | 180         | 13                     |
| INC-80001     | PRJ-1013     | VND-116     | 2023-09-05      | Electrical Contact | Day     | No              | No               | 20                        | Clear               | Lost Time Injury | 8           | 46                     |
| INC-80002     | PRJ-1017     | VND-113     | 2024-01-08      | Trench Collapse    | Day     | 1               | No               | 8                         | Wind                | Minor            | 0           | 20                     |
| INC-80003     | PRJ-1060     |             | 2025-08-18      | Fall from Height   | Day     | No              | yes              | 0                         | HEAT                | Lost Time Injury | 13          | 22                     |
| INC-80004     | PRJ-1010     | VND-104     | 16 Oct 2023     | ELECTRICAL CONTACT | Day     | Yes             | 0                | 6                         | Clear               | MINOR            | 0           | 32                     |

| task_id   | project_id   | vendor_id   | task_type   | planned_start   | planned_end   | actual_start   | actual_end   | crew_size   | task_status   | rework_required   |
|:----------|:-------------|:------------|:------------|:----------------|:--------------|:---------------|:-------------|:------------|:--------------|:------------------|
| TSK-20200 | PRJ-1015     | VND-112     | Finishing   | 2024-03-05      | 2024-05-11    | 2024-03-04     | 2024-05-10   | 41          | Completed     | no                |
| TSK-20584 | PRJ-1045     | VND-118     | Inspection  | 2024-06-28      | 2024-07-03    | 29 Jun 2024    | 2024/07/06   | 47          | Completed     | No                |
| TSK-21204 | PRJ-1032     | VND-108     | Inspection  | 2024-04-29      | 2024-05-05    | 2024-04-30     | 2024-05-08   | 44          | Completed     | 0                 |
| TSK-22643 | PRJ-1033     | VND-110     | Finishing   | April 15, 2024  | 2024-06-14    | April 17, 2024 | 2024-06-22   | 36          | Completed     | Yes               |
| TSK-20920 | PRJ-1077     | VND-105     | roofing     | 2023-03-09      | 2023-04-04    | 2023-03-08     | 04-03-2023   | 23          | Completed     | FALSE             |

---

## 3. Data Dictionary

The raw data was deliberately messy. The right-hand column is what had to be fixed.

**dim_projects**

| Column | Meaning | Known issues |
|--------|---------|--------------|
| `project_id` | Unique project code | Duplicate rows |
| `country` | Where the project is built | Variants like UK, KSA, RSA |
| `project_type` | Residential, Commercial, Industrial, Infrastructure | Mixed case, spaces |
| `contract_type` | Fixed Price, Cost Plus, Design-Build | Case and hyphen variants |
| `client_sector` | Public or Private | None |
| `budget_usd` | Approved budget | Text like `12.5M`, `$12,500,000` |
| `start_date` | Planned start | Six date styles, junk dates |
| `planned_duration_days` | Planned length | `18 months`, `N/A`, negatives |
| `site_area_sqm` | Site area | Negative and absurd values |
| `num_floors` | Floors | Blank for infrastructure |
| `project_manager_experience_years` | Manager experience | None |

**dim_vendors**

| Column | Meaning | Known issues |
|--------|---------|--------------|
| `vendor_id`, `vendor_name` | Vendor key and name | Duplicates, spacing |
| `vendor_type` | Subcontractor or Supplier | Case, spaces |
| `reliability_rating`, `safety_rating` | 1 to 5 ratings | `4/5`, `five`, missing |
| `years_active` | Years working with the company | None |
| `prequalified` | Passed prequalification | Yes/Y/TRUE/1 variants |

**fact_cost_ledger**

| Column | Meaning | Known issues |
|--------|---------|--------------|
| `period` | Quarter of the cost | `2023-Q2`, `Q2 2023`, `2023/Q2` |
| `cost_category` | Materials, Labour, Equipment, Subcontractors, Overheads | Case, spaces |
| `budgeted_usd`, `actual_usd` | Planned and real cost | Text with symbols, negatives |
| `approved_change_orders_usd` | Approved changes | Missing values |

**fact_tasks**

| Column | Meaning | Known issues |
|--------|---------|--------------|
| `task_type` | Excavation, Foundation, Structural Frame, MEP, Finishing, Roofing, Inspection | Case, spaces |
| `planned_start`, `planned_end` | Schedule | Six date styles |
| `actual_start`, `actual_end` | What really happened | Swapped dates in some rows |
| `crew_size` | Workers on the task | 0, 999, negatives |
| `task_status` | Completed, In Progress, Cancelled | `Complete`, `In-Progress` |
| `rework_required` | Work had to be redone | Yes/No variants |

**fact_material_orders**

| Column | Meaning | Known issues |
|--------|---------|--------------|
| `material` | Cement, Steel Rebar, Timber, Glass Panels, Electrical Fittings | Case, spaces |
| `quantity`, `unit` | Amount and unit | Negatives, blanks, `t`, `MT`, `pcs` |
| `unit_price_usd`, `total_cost_usd` | Price and order value | Text with symbols, planted price spikes |
| `order_date`, `promised_date`, `delivered_date` | Delivery timeline | `not delivered`, mixed formats |
| `quality_passed` | Passed inspection | Yes/No variants |

**fact_safety_incidents**

| Column | Meaning | Known issues |
|--------|---------|--------------|
| `incident_type` | Fall from Height, Struck by Object, Electrical Contact, Equipment Collision, Trench Collapse, Slip or Trip | Case, spaces |
| `shift`, `weather_condition` | Day/Night, Clear/Rain/Heat/Wind | Mixed case |
| `ppe_compliant`, `work_at_height` | Safety conditions | Yes/No variants |
| `worker_experience_years` | Worker experience | Negatives, 99 to 150 |
| `severity` | Near Miss, Minor, Lost Time Injury, Fatality | `LTI` |
| `lost_days` | Working days lost | **Leakage** for Objective 3, excluded from inputs |

### Code: load the data dictionary

```python
data_dictionary = pd.read_csv(f"{DATA_DIR}/data_dictionary.csv")
objectives = pd.read_csv(f"{DATA_DIR}/business_objectives.csv")

display(data_dictionary)
display(objectives)
```

#### Output

| table                | column                           | data_type   | description                                                                                | known_data_issues                                   |
|:---------------------|:---------------------------------|:------------|:-------------------------------------------------------------------------------------------|:----------------------------------------------------|
| dim_projects         | project_id                       | Text (key)  | Unique project code, PRJ-####                                                              | Duplicate rows                                      |
| dim_projects         | project_name                     | Text        | Project name                                                                               | Extra spaces, mixed case                            |
| dim_projects         | country                          | Text        | Country where the project is built                                                         | Spelling variants (UK, U.K., KSA, RSA...)           |
| dim_projects         | project_type                     | Text        | Residential, Commercial, Industrial or Infrastructure                                      | Mixed case, spaces                                  |
| dim_projects         | contract_type                    | Text        | Fixed Price, Cost Plus or Design-Build                                                     | Upper case, hyphen variants                         |
| dim_projects         | client_sector                    | Text        | Public or Private client                                                                   |                                                     |
| dim_projects         | budget_usd                       | Currency    | Approved contract budget in USD                                                            | Text such as 12.5M, $12,500,000, USD 12,500,000     |
| dim_projects         | start_date                       | Date        | Planned project start date                                                                 | Six date styles, junk dates                         |
| dim_projects         | planned_duration_days            | Integer     | Planned project length in days                                                             | Text like '18 months', 'N/A', '-5'                  |
| dim_projects         | site_area_sqm                    | Integer     | Site area in square metres                                                                 | Negative and absurd values                          |
| dim_projects         | num_floors                       | Integer     | Number of floors (blank for infrastructure)                                                | Extra blanks                                        |
| dim_projects         | project_manager_experience_years | Integer     | Years of experience of the project manager                                                 |                                                     |
| dim_vendors          | vendor_id                        | Text (key)  | Unique vendor code, VND-###                                                                | Duplicate rows                                      |
| dim_vendors          | vendor_name                      | Text        | Vendor trading name                                                                        | Extra spaces, mixed case                            |
| dim_vendors          | vendor_type                      | Text        | Subcontractor or Supplier                                                                  | Mixed case, spaces                                  |
| dim_vendors          | specialty                        | Text        | Trade (subcontractors) or material (suppliers)                                             |                                                     |
| dim_vendors          | country                          | Text        | Vendor base country                                                                        | Spelling variants                                   |
| dim_vendors          | reliability_rating               | Integer 1-5 | Company rating for on-time performance                                                     | Text like '4/5', 'five', 'N/A'                      |
| dim_vendors          | safety_rating                    | Integer 1-5 | Company rating for safety performance                                                      | Missing values                                      |
| dim_vendors          | years_active                     | Integer     | Years the vendor has worked with the company                                               |                                                     |
| dim_vendors          | prequalified                     | Boolean     | Passed prequalification                                                                    | Yes/Y/TRUE/1 and No/N/FALSE/0                       |
| fact_tasks           | task_id                          | Text (key)  | Unique task code, TSK-#####                                                                | Duplicate rows                                      |
| fact_tasks           | project_id                       | Text (FK)   | Links to dim_projects                                                                      | Orphan codes, lower case                            |
| fact_tasks           | vendor_id                        | Text (FK)   | Subcontractor doing the task                                                               | Missing values                                      |
| fact_tasks           | task_type                        | Text        | Excavation, Foundation, Structural Frame, MEP Installation, Finishing, Roofing, Inspection | Mixed case, spaces                                  |
| fact_tasks           | planned_start                    | Date        | Scheduled start                                                                            | Six date styles, junk dates                         |
| fact_tasks           | planned_end                      | Date        | Scheduled finish                                                                           | Six date styles, junk dates                         |
| fact_tasks           | actual_start                     | Date        | Actual start                                                                               | Mixed formats; swapped with actual_end in some rows |
| fact_tasks           | actual_end                       | Date        | Actual finish (blank unless Completed)                                                     | Mixed formats, blanks                               |
| fact_tasks           | crew_size                        | Integer     | Workers on the task                                                                        | 0, 999, negatives, blanks                           |
| fact_tasks           | task_status                      | Text        | Completed, In Progress or Cancelled                                                        | Case, 'Complete', 'In-Progress'                     |
| fact_tasks           | rework_required                  | Boolean     | Work had to be redone                                                                      | Yes/Y/TRUE/1 variants                               |
| fact_material_orders | order_id                         | Text (key)  | Unique order code, ORD-#####                                                               | Duplicate rows                                      |
| fact_material_orders | project_id                       | Text (FK)   | Links to dim_projects                                                                      |                                                     |
| fact_material_orders | vendor_id                        | Text (FK)   | Supplier, links to dim_vendors                                                             | Missing values, orphan codes                        |
| fact_material_orders | material                         | Text        | Cement, Steel Rebar, Timber, Glass Panels, Electrical Fittings                             | Mixed case, spaces                                  |
| fact_material_orders | quantity                         | Number      | Quantity ordered, in `unit`                                                                | Negative, blank, N/A                                |
| fact_material_orders | unit                             | Text        | bags, tonnes, m3, sheets or pieces                                                         | Variants: t, MT, tons, cubic meters, CUM, pcs, nos  |
| fact_material_orders | unit_price_usd                   | Currency    | Price per unit in USD                                                                      | Text with symbols; planted price spikes             |
| fact_material_orders | total_cost_usd                   | Currency    | Order value in USD (quantity x unit price)                                                 | Text with symbols                                   |

*Showing the first 40 of 64 rows.*

| objective_no   | business_objective                             | ml_task                                           | target_definition                                                             | main_features                                                                                                                                  | tables_used                                     | signal_in_data               |
|:---------------|:-----------------------------------------------|:--------------------------------------------------|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------|:-----------------------------|
| 1              | Control project cost overruns                  | Regression and classification                     | overrun_pct = actual_usd / budgeted_usd - 1; overrun_flag = overrun_pct > 15% | cost_category, project_type, contract_type, country, period year, project size                                                                 | fact_cost_ledger, dim_projects                  | Strong                       |
| 2              | Reduce schedule delays                         | Classification and regression                     | delay_days = actual_end - planned_end; delay_flag = delay_days > 5            | task_type, vendor reliability, crew_size, project type, rainy season, planned duration                                                         | fact_tasks, dim_vendors, dim_projects           | Strong                       |
| 3              | Improve site safety                            | Classification                                    | serious_flag = severity in (Lost Time Injury, Fatality)                       | incident_type, ppe_compliant, shift, work_at_height, worker experience, weather, vendor safety_rating                                          | fact_safety_incidents, dim_vendors              | Strong                       |
| 4              | Make material procurement more reliable        | Classification                                    | late_delivery = delivered_date > promised_date                                | material, quantity, international supplier (supplier country differs from project country), rainy season, vendor reliability_rating, lead time | fact_material_orders, dim_vendors, dim_projects | Strong                       |
| 5              | Detect procurement price and invoice anomalies | Unsupervised anomaly detection (Isolation Forest) | none; score against anomaly_answer_key.csv only at the end                    | unit_price_usd relative to material and country, total_cost_usd vs quantity x unit price                                                       | fact_material_orders                            | Planted (about 2% of orders) |

---

## 4. Data Quality Audit

I measured every problem **before** touching anything, so each later fix traces back to a number.

What was checked:
- Table overview and missing values by column
- Business-rule breaks: end dates before start dates, negative costs, impossible crew sizes
- Orphan keys (cost, task and order rows pointing to projects that do not exist)
- Spelling variants in every category column
- The Objective 5 answer key, read only to confirm it matches the orders table

### Code: data quality audit

```python
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

#### Output

|                       | rows   | columns   | duplicate_rows   | missing_key   | duplicate_key_rows   | missing_cells   |
|:----------------------|:-------|:----------|:-----------------|:--------------|:---------------------|:----------------|
| dim_projects          | 84     | 12        | 4                | 0             | 8                    | 21              |
| dim_vendors           | 48     | 9         | 3                | 0             | 6                    | 3               |
| fact_cost_ledger      | 2465   | 7         | 30               | 0             | 60                   | 195             |
| fact_tasks            | 4045   | 11        | 45               | 0             | 90                   | 673             |
| fact_material_orders  | 3035   | 12        | 35               | 0             | 70                   | 86              |
| fact_safety_incidents | 768    | 13        | 18               | 0             | 36                   | 44              |

| table                 | column                     | missing   | missing_pct   |
|:----------------------|:---------------------------|:----------|:--------------|
| dim_projects          | planned_duration_days      | 1         | 1.2           |
| dim_projects          | num_floors                 | 20        | 23.8          |
| dim_vendors           | reliability_rating         | 1         | 2.1           |
| dim_vendors           | safety_rating              | 2         | 4.2           |
| fact_cost_ledger      | approved_change_orders_usd | 195       | 7.9           |
| fact_tasks            | vendor_id                  | 41        | 1             |
| fact_tasks            | actual_start               | 3         | 0.1           |
| fact_tasks            | actual_end                 | 548       | 13.5          |
| fact_tasks            | crew_size                  | 81        | 2             |
| fact_material_orders  | vendor_id                  | 25        | 0.8           |
| fact_material_orders  | quantity                   | 31        | 1             |
| fact_material_orders  | delivered_date             | 30        | 1             |
| fact_safety_incidents | vendor_id                  | 37        | 4.8           |
| fact_safety_incidents | worker_experience_years    | 7         | 0.9           |

| table                 | check                                                  | rows   |
|:----------------------|:-------------------------------------------------------|:-------|
| dim_projects          | Budget stored as text (symbols or M/K)                 | 27     |
| dim_projects          | Planned duration nonnumeric                            | 1      |
| dim_projects          | Planned duration zero or negative                      | 1      |
| dim_projects          | Site area zero or negative                             | 1      |
| dim_projects          | Site area above 1,000,000 sqm                          | 1      |
| dim_vendors           | Reliability rating nonnumeric                          | 2      |
| fact_cost_ledger      | Actual amount negative                                 | 10     |
| fact_cost_ledger      | Period not in YYYY-QN style                            | 913    |
| fact_tasks            | Crew size zero, negative or 999                        | 12     |
| fact_tasks            | Planned start unparsed                                 | 37     |
| fact_tasks            | Planned end unparsed                                   | 45     |
| fact_tasks            | Actual start unparsed                                  | 36     |
| fact_tasks            | Actual end unparsed                                    | 204    |
| fact_tasks            | Actual end before actual start (swapped)               | 97     |
| fact_tasks            | Planned end before planned start                       | 62     |
| fact_tasks            | Actual end missing (task not finished)                 | 548    |
| fact_material_orders  | Quantity zero or negative                              | 14     |
| fact_material_orders  | Order date unparsed                                    | 26     |
| fact_material_orders  | Promised date unparsed                                 | 28     |
| fact_material_orders  | Delivered date unparsed (text such as 'not delivered') | 26     |
| fact_material_orders  | Delivered before ordered                               | 49     |
| fact_safety_incidents | Worker experience negative                             | 3      |
| fact_safety_incidents | Worker experience above 60 years                       | 6      |
| fact_safety_incidents | Incident date unparsed                                 | 5      |

| relationship                                     | nonmissing_keys   | orphan_keys   |
|:-------------------------------------------------|:------------------|:--------------|
| fact_cost_ledger.project_id -> dim_projects      | 2465              | 12            |
| fact_tasks.project_id -> dim_projects            | 4045              | 15            |
| fact_material_orders.project_id -> dim_projects  | 3035              | 0             |
| fact_safety_incidents.project_id -> dim_projects | 768               | 0             |
| fact_tasks.vendor_id -> dim_vendors              | 4004              | 0             |
| fact_material_orders.vendor_id -> dim_vendors    | 3010              | 12            |
| fact_safety_incidents.vendor_id -> dim_vendors   | 731               | 0             |

| table                 | column            | raw_spellings   | after_trim_and_lowercase   |
|:----------------------|:------------------|:----------------|:---------------------------|
| dim_projects          | country           | 30              | 20                         |
| dim_projects          | project_type      | 11              | 4                          |
| dim_projects          | contract_type     | 5               | 3                          |
| dim_projects          | client_sector     | 2               | 2                          |
| dim_vendors           | vendor_type       | 6               | 2                          |
| dim_vendors           | country           | 24              | 17                         |
| dim_vendors           | prequalified      | 9               | 7                          |
| fact_cost_ledger      | cost_category     | 25              | 5                          |
| fact_tasks            | task_type         | 35              | 7                          |
| fact_tasks            | task_status       | 11              | 4                          |
| fact_tasks            | rework_required   | 10              | 8                          |
| fact_material_orders  | material          | 25              | 5                          |
| fact_material_orders  | unit              | 18              | 15                         |
| fact_material_orders  | quality_passed    | 10              | 8                          |
| fact_safety_incidents | incident_type     | 29              | 6                          |
| fact_safety_incidents | shift             | 10              | 2                          |
| fact_safety_incidents | ppe_compliant     | 10              | 8                          |
| fact_safety_incidents | work_at_height    | 10              | 8                          |
| fact_safety_incidents | weather_condition | 20              | 4                          |
| fact_safety_incidents | severity          | 13              | 5                          |

| check                            | value   |
|:---------------------------------|:--------|
| Answer key rows                  | 3,000   |
| Unique order ids in orders table | 3,000   |
| Orders missing from answer key   | 0       |
| Planted anomalies in key         | 62      |
| Anomaly rate (%)                 | 2.07    |

---

## 5. Data Cleaning and Error Checking

Every fix is written to a cleaning log, and a verification step proves the result: no duplicate keys, no impossible dates, no unmapped labels, no orphan codes.

Two decisions worth knowing:
- **Price spikes were kept on purpose.** They are what Objective 5 has to find.
- **Missing numbers were filled with a group median** (for example, crew size by task type), never a global guess.

### Code: data cleaning and error checking

```python
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

#### Output

| table                 | action                                                            | rows   |
|:----------------------|:------------------------------------------------------------------|:-------|
| dim_projects          | Exact duplicate rows removed                                      | 4      |
| dim_projects          | Impossible site area set to missing                               | 2      |
| dim_projects          | planned_duration_days missing, filled with median of project type | 2      |
| dim_projects          | site_area_sqm missing, filled with median of project type         | 2      |
| dim_projects          | num_floors missing, filled with median of project type            | 20     |
| dim_vendors           | Exact duplicate rows removed                                      | 3      |
| dim_vendors           | reliability_rating missing, filled with median of vendor type     | 1      |
| dim_vendors           | safety_rating missing, filled with median of vendor type          | 2      |
| fact_cost_ledger      | Exact duplicate rows removed                                      | 30     |
| fact_cost_ledger      | Rows with unknown project removed                                 | 12     |
| fact_cost_ledger      | Rows removed: negative or missing budget/actual cost              | 10     |
| fact_cost_ledger      | Missing change orders set to 0                                    | 194    |
| fact_tasks            | Exact duplicate rows removed                                      | 45     |
| fact_tasks            | Rows with unknown project removed                                 | 15     |
| fact_tasks            | Swapped actual start/end dates put back in order                  | 39     |
| fact_tasks            | crew_size impossible or missing, filled with median of task type  | 91     |
| fact_material_orders  | Exact duplicate rows removed                                      | 35     |
| fact_material_orders  | Unknown vendor codes set to missing (order kept)                  | 12     |
| fact_material_orders  | Negative quantity sign corrected                                  | 14     |
| fact_material_orders  | Missing quantity rebuilt from total cost / unit price             | 30     |
| fact_material_orders  | Missing quantity filled with material median                      | 0      |
| fact_material_orders  | Delivered date missing (not delivered or unreadable)              | 56     |
| fact_safety_incidents | Exact duplicate rows removed                                      | 18     |
| fact_safety_incidents | worker_experience_years impossible or missing, filled with median | 15     |

|                       | rows_before   | rows_after   | duplicate_keys   | missing_cells   |
|:----------------------|:--------------|:-------------|:-----------------|:----------------|
| dim_projects          | 84            | 80           | 0                | 0               |
| dim_vendors           | 48            | 45           | 0                | 0               |
| fact_cost_ledger      | 2465          | 2413         | 0                | 0               |
| fact_tasks            | 4045          | 3985         | 0                | 905             |
| fact_material_orders  | 3035          | 3000         | 0                | 147             |
| fact_safety_incidents | 768           | 750          | 0                | 41              |

| table                 | column         | missing   |
|:----------------------|:---------------|:----------|
| fact_tasks            | vendor_id      | 40        |
| fact_tasks            | planned_start  | 37        |
| fact_tasks            | planned_end    | 45        |
| fact_tasks            | actual_start   | 39        |
| fact_tasks            | actual_end     | 744       |
| fact_material_orders  | vendor_id      | 37        |
| fact_material_orders  | order_date     | 26        |
| fact_material_orders  | promised_date  | 28        |
| fact_material_orders  | delivered_date | 56        |
| fact_safety_incidents | vendor_id      | 37        |
| fact_safety_incidents | incident_date  | 4         |

| check                                   | rows   |
|:----------------------------------------|:-------|
| Tasks: actual end before actual start   | 0      |
| Tasks: planned end before planned start | 0      |
| Orders: delivered before ordered        | 0      |
| Orders: promised before ordered         | 0      |
| Ledger: orphan project codes            | 0      |
| Tasks: orphan project codes             | 0      |
| Orders: orphan project codes            | 0      |
| Incidents: orphan project codes         | 0      |
| Unmapped category labels (all tables)   | 0      |

|                 | country (projects)   | task_type   | cost_category   | material   | incident_type   | severity   |
|:----------------|:---------------------|:------------|:----------------|:-----------|:----------------|:-----------|
| distinct labels | 10                   | 7           | 5               | 5          | 6               | 4          |

---

## 6. Feature Engineering

Raw columns become signals a model can use, and the four targets are created here. Every feature is logged with its meaning.


> The rainy-season calendar (Nigeria and India June to September, Brazil December to March, UK October to January) is a domain assumption, not something measured from the data.

### Code: feature engineering

```python
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

#### Output

| table                | feature                 | meaning                                                                                  |
|:---------------------|:------------------------|:-----------------------------------------------------------------------------------------|
| dim_projects         | start_year              | Year the project started                                                                 |
| dim_projects         | start_month             | Month the project started                                                                |
| dim_projects         | planned_end_date        | Start date + planned duration                                                            |
| dim_projects         | budget_per_sqm          | Budget per square metre of site                                                          |
| dim_projects         | budget_per_day          | Budget per planned day                                                                   |
| dim_projects         | budget_per_floor        | Budget per floor                                                                         |
| dim_projects         | size_band               | Small / Medium / Large site (three equal groups)                                         |
| dim_projects         | high_rise               | 1 if 30 or more floors                                                                   |
| dim_projects         | is_public               | 1 if public sector client                                                                |
| dim_projects         | manager_experience_band | Junior (<10), Mid (10-19), Senior (20+ years)                                            |
| dim_vendors          | vendor_score            | Average of reliability and safety rating                                                 |
| dim_vendors          | vendor_experience_band  | New (<10), Established (10-24), Veteran (25+ years active)                               |
| dim_vendors          | is_prequalified         | 1 if the vendor is prequalified                                                          |
| fact_cost_ledger     | period_year             | Year of the cost period                                                                  |
| fact_cost_ledger     | period_quarter          | Quarter (1-4) of the cost period                                                         |
| fact_cost_ledger     | overrun_usd             | Actual minus budget in USD                                                               |
| fact_cost_ledger     | overrun_pct             | Actual / budget - 1 (objective 1 regression target)                                      |
| fact_cost_ledger     | overrun_flag            | 1 if overrun is above 15% (objective 1 classification target)                            |
| fact_cost_ledger     | change_order_pct        | Approved change orders as a share of budget                                              |
| fact_cost_ledger     | budget_share_of_project | This line's share of the project's ledger budget                                         |
| fact_cost_ledger     | log_budgeted_usd        | Log of the budgeted amount                                                               |
| fact_tasks           | planned_duration_days   | Planned end minus planned start                                                          |
| fact_tasks           | planned_start_year      | Year the task was planned to start                                                       |
| fact_tasks           | planned_start_month     | Month the task was planned to start                                                      |
| fact_tasks           | rainy_season            | 1 if the planned start month is in the project country's rainy season                    |
| fact_tasks           | crew_band               | Small (<20), Medium (20-39), Large (40+) crew                                            |
| fact_tasks           | vendor_known            | 1 if the vendor on the task is recorded                                                  |
| fact_tasks           | delay_days              | Actual end minus planned end, completed tasks only (objective 2 regression target)       |
| fact_tasks           | delay_flag              | 1 if delay is more than 5 days, completed tasks only (objective 2 classification target) |
| fact_tasks           | actual_duration_days    | Actual end minus actual start, completed tasks only                                      |
| fact_material_orders | order_year              | Year the order was placed                                                                |
| fact_material_orders | order_month             | Month the order was placed                                                               |
| fact_material_orders | lead_time_days          | Promised date minus order date                                                           |
| fact_material_orders | actual_lead_days        | Delivered date minus order date                                                          |
| fact_material_orders | delivery_delay_days     | Delivered date minus promised date                                                       |
| fact_material_orders | late_delivery           | 1 if delivered after the promised date (objective 4 target)                              |
| fact_material_orders | international_supplier  | 1 if supplier country differs from project country                                       |
| fact_material_orders | rainy_season            | 1 if the promised month is in the project country's rainy season                         |
| fact_material_orders | log_quantity            | Log of the quantity ordered                                                              |
| fact_material_orders | price_ratio_material    | Unit price / median price of the same material                                           |

*Showing the first 40 of 48 rows.*

|                       | rows   | columns   | new_features   |
|:----------------------|:-------|:----------|:---------------|
| dim_projects          | 80     | 22        | 10             |
| dim_vendors           | 45     | 12        | 3              |
| fact_cost_ledger      | 2413   | 15        | 8              |
| fact_tasks            | 3985   | 20        | 9              |
| fact_material_orders  | 3000   | 24        | 12             |
| fact_safety_incidents | 750    | 19        | 6              |

| objective          | target        | usable_rows   | positive_rate   |
|:-------------------|:--------------|:--------------|:----------------|
| 1 Cost overrun     | overrun_flag  | 2413          | 46.7            |
| 2 Schedule delay   | delay_flag    | 3237          | 40.9            |
| 3 Serious incident | serious_flag  | 750           | 25.1            |
| 4 Late delivery    | late_delivery | 2918          | 45.9            |

|    | late_rate   |
|:---|:------------|
| 0  | 0.2857      |
| 1  | 0.4799      |

|    | delay_rate   |
|:---|:-------------|
| 0  | 0.3578       |
| 1  | 0.7465       |

|    | serious_rate   |
|:---|:---------------|
| 0  | 0.2287         |
| 1  | 0.587          |

---

## 7. Create Relationship Table

Every fact table links to the dimension tables (projects, vendors, dates) through a star schema. A join test confirms that a many-to-one join never adds or loses a row.

### Code: relationship table (star schema)

```python
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

#### Output

![Star schema](images/s04_01.png)

| child_table           | child_key      | parent_table   | parent_key   | kind        | child_rows   | missing_key   | orphan_keys   | match_pct   | parent_key_unique   |
|:----------------------|:---------------|:---------------|:-------------|:------------|:-------------|:--------------|:--------------|:------------|:--------------------|
| fact_cost_ledger      | project_id     | dim_projects   | project_id   | many-to-one | 2413         | 0             | 0             | 100         | True                |
| fact_tasks            | project_id     | dim_projects   | project_id   | many-to-one | 3985         | 0             | 0             | 100         | True                |
| fact_material_orders  | project_id     | dim_projects   | project_id   | many-to-one | 3000         | 0             | 0             | 100         | True                |
| fact_safety_incidents | project_id     | dim_projects   | project_id   | many-to-one | 750          | 0             | 0             | 100         | True                |
| fact_tasks            | vendor_id      | dim_vendors    | vendor_id    | many-to-one | 3985         | 40            | 0             | 100         | True                |
| fact_material_orders  | vendor_id      | dim_vendors    | vendor_id    | many-to-one | 3000         | 37            | 0             | 100         | True                |
| fact_safety_incidents | vendor_id      | dim_vendors    | vendor_id    | many-to-one | 750          | 37            | 0             | 100         | True                |
| fact_cost_ledger      | date_key       | dim_date       | date_key     | many-to-one | 2413         | 0             | 0             | 100         | True                |
| fact_tasks            | date_key       | dim_date       | date_key     | many-to-one | 3985         | 37            | 0             | 100         | True                |
| fact_material_orders  | date_key       | dim_date       | date_key     | many-to-one | 3000         | 26            | 0             | 100         | True                |
| fact_safety_incidents | date_key       | dim_date       | date_key     | many-to-one | 750          | 4             | 0             | 100         | True                |
| dim_projects          | start_date_key | dim_date       | date_key     | many-to-one | 80           | 0             | 0             | 100         | True                |

|                       | rows_before   | rows_after   | rows_preserved   | project_attributes_found   |
|:----------------------|:--------------|:-------------|:-----------------|:---------------------------|
| fact_cost_ledger      | 2413          | 2413         | True             | 2413                       |
| fact_tasks            | 3985          | 3985         | True             | 3985                       |
| fact_material_orders  | 3000          | 3000         | True             | 3000                       |
| fact_safety_incidents | 750           | 750          | True             | 750                        |

| date       | date_key   | year   | quarter   | month   | month_name   | year_month   | week_of_year   | day_of_week   | is_weekend   |
|:-----------|:-----------|:-------|:----------|:--------|:-------------|:-------------|:---------------|:--------------|:-------------|
| 2022-01-01 | 20220101   | 2022   | 1         | 1       | January      | 2022-01      | 52             | Saturday      | 1            |
| 2022-01-02 | 20220102   | 2022   | 1         | 1       | January      | 2022-01      | 52             | Sunday        | 1            |
| 2022-01-03 | 20220103   | 2022   | 1         | 1       | January      | 2022-01      | 1              | Monday        | 0            |

```text
dim_date: 1,938 days, 2022-01-01 to 2027-04-22
```

---

## 8. Exploratory Data Analysis

Nine charts answered the business questions before any modelling started. The headline: problems are the norm, not the exception.

| Outcome | Share of records |
|---------|:----------------:|
| Cost lines that overran by more than 15% | **47%** |
| Tasks that finished more than 5 days late | **41%** |
| Incidents that were serious | **25%** |
| Material orders that arrived late | **46%** |

### Code: exploratory data analysis

```python
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

#### Output

![Portfolio Overview](images/s05_01.png)

![Cost Overruns By Category](images/s05_02.png)

![Cost Overruns By Project](images/s05_03.png)

![Cost Overruns Over Time](images/s05_04.png)

![Task Delays Drivers](images/s05_05.png)

![Task Delay Size And Timing](images/s05_06.png)

![Safety Serious Incidents](images/s05_07.png)

![Late Deliveries](images/s05_08.png)

![Price Anomalies](images/s05_09.png)

| objective          | factor                 | highest_group    | highest_rate_pct   | lowest_group        | lowest_rate_pct   | gap_points   |
|:-------------------|:-----------------------|:-----------------|:-------------------|:--------------------|:------------------|:-------------|
| 1 Cost overrun     | cost_category          | Materials        | 68.7               | Overheads           | 20.9              | 47.8         |
| 1 Cost overrun     | country                | Nigeria          | 68.7               | United States       | 27.4              | 41.2         |
| 1 Cost overrun     | contract_type          | Cost Plus        | 65.7               | Fixed Price         | 36.2              | 29.5         |
| 1 Cost overrun     | high_rise              | 1                | 53.9               | 0                   | 35.4              | 18.6         |
| 1 Cost overrun     | project_type           | Infrastructure   | 58.3               | Residential         | 40.9              | 17.5         |
| 1 Cost overrun     | period_year            | 2023             | 53                 | 2024                | 40.6              | 12.4         |
| 1 Cost overrun     | client_sector          | Private          | 49.2               | Public              | 41.8              | 7.5          |
| 1 Cost overrun     | size_band              | Large            | 49.1               | Medium              | 45.2              | 3.9          |
| 2 Schedule delay   | rainy_season           | 1.0              | 74.7               | 0.0                 | 35.8              | 38.9         |
| 2 Schedule delay   | task_type              | Structural Frame | 55.7               | Inspection          | 20.2              | 35.6         |
| 2 Schedule delay   | crew_band              | Small            | 59.2               | Large               | 24.7              | 34.5         |
| 2 Schedule delay   | country                | United Kingdom   | 57                 | Saudi Arabia        | 31                | 26           |
| 2 Schedule delay   | reliability_rating     | 3.0              | 53.2               | 5.0                 | 31.3              | 21.9         |
| 2 Schedule delay   | project_type           | Infrastructure   | 45.4               | Residential         | 36.2              | 9.2          |
| 2 Schedule delay   | rework_required        | Yes              | 43                 | No                  | 40.6              | 2.4          |
| 2 Schedule delay   | vendor_known           | 0                | 41.2               | 1                   | 40.9              | 0.3          |
| 3 Serious incident | incident_type          | Fall from Height | 51.7               | Slip or Trip        | 3.7               | 48           |
| 3 Serious incident | no_ppe_at_height       | 1                | 58.7               | 0                   | 22.9              | 35.8         |
| 3 Serious incident | experience_band        | Novice           | 41.8               | Veteran             | 10.6              | 31.2         |
| 3 Serious incident | work_at_height         | Yes              | 43.2               | No                  | 17.5              | 25.6         |
| 3 Serious incident | ppe_compliant          | No               | 41.5               | Yes                 | 21.1              | 20.4         |
| 3 Serious incident | shift                  | Night            | 37.9               | Day                 | 20.7              | 17.2         |
| 3 Serious incident | weather_condition      | Rain             | 28.4               | Clear               | 24                | 4.4          |
| 4 Late delivery    | international_supplier | 1.0              | 48                 | 0.0                 | 28.6              | 19.4         |
| 4 Late delivery    | reliability_rating     | 3.0              | 51.9               | 5.0                 | 37.5              | 14.4         |
| 4 Late delivery    | material               | Glass Panels     | 53.9               | Electrical Fittings | 40.6              | 13.2         |
| 4 Late delivery    | rainy_season           | 1.0              | 56.5               | 0.0                 | 44.3              | 12.2         |
| 4 Late delivery    | country                | Australia        | 49.6               | UAE                 | 42.9              | 6.7          |
| 4 Late delivery    | lead_band              | 2-4 weeks        | 46.4               | Over 6 weeks        | 41.5              | 4.9          |

```text
Saved 9 charts and factor_ranking.csv to construction_eda/
```

---

## 9. Analytical Base Table

One modelling table per objective. Each holds **only what is known at the moment of the decision**. Anything that is a result of the outcome (actual cost, actual dates, severity, lost days, delivery date, the answer key) is treated as leakage and left out. This is why the scores are honest.

### Code: analytical base table

```python
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

#### Output

|                                          | rows   | numeric_features   | text_features   | target              | positive_rate_pct   | missing_cells   |
|:-----------------------------------------|:-------|:-------------------|:----------------|:--------------------|:--------------------|:----------------|
| Objective 1: cost overrun                | 2413   | 12                 | 4               | overrun_flag        | 46.7                | 0               |
| Objective 2: schedule delay              | 3237   | 14                 | 4               | delay_flag          | 40.9                | 290             |
| Objective 3: serious safety incident     | 750    | 10                 | 5               | serious_flag        | 25.1                | 78              |
| Objective 4: late material delivery      | 2918   | 11                 | 4               | late_delivery       | 45.9                | 294             |
| Objective 5: price and invoice anomalies | 3000   | 4                  | 0               | none (unsupervised) |                     | 0               |

| objective                            | column                 | missing   | missing_pct   |
|:-------------------------------------|:-----------------------|:----------|:--------------|
| Objective 2: schedule delay          | task_planned_days      | 67        | 2.1           |
| Objective 2: schedule delay          | planned_start_month    | 29        | 0.9           |
| Objective 2: schedule delay          | planned_start_year     | 29        | 0.9           |
| Objective 2: schedule delay          | rainy_season           | 29        | 0.9           |
| Objective 2: schedule delay          | reliability_rating     | 34        | 1.1           |
| Objective 2: schedule delay          | vendor_safety_rating   | 34        | 1.1           |
| Objective 2: schedule delay          | vendor_years_active    | 34        | 1.1           |
| Objective 2: schedule delay          | is_prequalified        | 34        | 1.1           |
| Objective 3: serious safety incident | incident_month         | 4         | 0.5           |
| Objective 3: serious safety incident | vendor_safety_rating   | 37        | 4.9           |
| Objective 3: serious safety incident | vendor_years_active    | 37        | 4.9           |
| Objective 4: late material delivery  | lead_time_days         | 24        | 0.8           |
| Objective 4: late material delivery  | order_month            | 24        | 0.8           |
| Objective 4: late material delivery  | order_year             | 24        | 0.8           |
| Objective 4: late material delivery  | international_supplier | 37        | 1.3           |
| Objective 4: late material delivery  | reliability_rating     | 37        | 1.3           |
| Objective 4: late material delivery  | vendor_safety_rating   | 37        | 1.3           |
| Objective 4: late material delivery  | vendor_years_active    | 37        | 1.3           |
| Objective 4: late material delivery  | is_prequalified        | 37        | 1.3           |
| Objective 4: late material delivery  | vendor_type            | 37        | 1.3           |

|                                          | left_out_on_purpose                                                                                  |
|:-----------------------------------------|:-----------------------------------------------------------------------------------------------------|
| Objective 1: cost overrun                | actual_usd, overrun_usd, overrun_pct, overrun_flag, approved_change_orders_usd, change_order_pct     |
| Objective 2: schedule delay              | actual_start, actual_end, actual_duration_days, delay_days, delay_flag, task_status, rework_required |
| Objective 3: serious safety incident     | severity, serious_flag, lost_days, reported_delay_hours                                              |
| Objective 4: late material delivery      | delivered_date, actual_lead_days, delivery_delay_days, late_delivery, quality_passed                 |
| Objective 5: price and invoice anomalies | is_anomaly, total_vs_calculated_pct                                                                  |

```text
Objective 1: cost overrun  (2413 rows x 17 columns)
```

| log_budgeted_usd   | budget_share_of_project   | period_year   | period_quarter   | project_budget_usd   | site_area_sqm   | num_floors   | project_planned_days   | manager_experience_years   | high_rise   | is_public   | budget_per_sqm   | cost_category   | country   | project_type   | contract_type   | overrun_flag   |
|:-------------------|:--------------------------|:--------------|:-----------------|:---------------------|:----------------|:-------------|:-----------------------|:---------------------------|:------------|:------------|:-----------------|:----------------|:----------|:---------------|:----------------|:---------------|
| 13.7825            | 0.0424                    | 2024          | 4                | 23,109,000           | 37173           | 7            | 801                    | 20                         | 0           | 0           | 621.66           | Materials       | Germany   | Residential    | Design-Build    | 0              |
| 13.3603            | 0.0278                    | 2024          | 4                | 23,109,000           | 37173           | 7            | 801                    | 20                         | 0           | 0           | 621.66           | Labour          | Germany   | Residential    | Design-Build    | 0              |
| 12.7096            | 0.0145                    | 2024          | 4                | 23,109,000           | 37173           | 7            | 801                    | 20                         | 0           | 0           | 621.66           | Equipment       | Germany   | Residential    | Design-Build    | 1              |
| 13.4313            | 0.0298                    | 2024          | 4                | 23,109,000           | 37173           | 7            | 801                    | 20                         | 0           | 0           | 621.66           | Subcontractors  | Germany   | Residential    | Design-Build    | 0              |
| 12.2773            | 0.0094                    | 2024          | 4                | 23,109,000           | 37173           | 7            | 801                    | 20                         | 0           | 0           | 621.66           | Overheads       | Germany   | Residential    | Design-Build    | 0              |

```text
Objective 2: schedule delay  (3237 rows x 19 columns)
```

| task_planned_days   | crew_size   | planned_start_month   | planned_start_year   | rainy_season   | reliability_rating   | vendor_safety_rating   | vendor_years_active   | is_prequalified   | vendor_known   | project_planned_days   | num_floors   | site_area_sqm   | manager_experience_years   | task_type        | country        | project_type   | contract_type   | delay_flag   |
|:--------------------|:------------|:----------------------|:---------------------|:---------------|:---------------------|:-----------------------|:----------------------|:------------------|:---------------|:-----------------------|:-------------|:----------------|:---------------------------|:-----------------|:---------------|:---------------|:----------------|:-------------|
| 72                  | 39          | 8                     | 2,023                | 0              | 3                    | 3                      | 22                    | 0                 | 1              | 896                    | 59           | 6263            | 4                          | Structural Frame | UAE            | Commercial     | Fixed Price     | 0            |
| 72                  | 44          | 1                     | 2,025                | 0              | 5                    | 4                      | 35                    | 1                 | 1              | 801                    | 7            | 37173           | 20                         | Structural Frame | Germany        | Residential    | Design-Build    | 0            |
| 70                  | 5           | 10                    | 2,023                | 1              | 4                    | 4                      | 27                    | 0                 | 1              | 583                    | 36           | 57824           | 22                         | Structural Frame | United Kingdom | Commercial     | Design-Build    | 1            |
| 63                  | 59          | 6                     | 2,024                | 1              | 5                    | 4                      | 35                    | 1                 | 1              | 880                    | 3            | 52144           | 28                         | Finishing        | Nigeria        | Industrial     | Fixed Price     | 1            |
| 32                  | 28          | 11                    | 2,022                | 0              | 3                    | 4                      | 24                    | 1                 | 1              | 335                    | 8            | 47463           | 3                          | Roofing          | South Africa   | Commercial     | Design-Build    | 0            |

```text
Objective 3: serious safety incident  (750 rows x 16 columns)
```

| worker_experience_years   | ppe_compliant_yes   | work_at_height_yes   | no_ppe_at_height   | incident_month   | vendor_safety_rating   | vendor_years_active   | vendor_known   | num_floors   | high_rise   | incident_type      | shift   | weather_condition   | project_type   | country        | serious_flag   |
|:--------------------------|:--------------------|:---------------------|:-------------------|:-----------------|:-----------------------|:----------------------|:---------------|:-------------|:------------|:-------------------|:--------|:--------------------|:---------------|:---------------|:---------------|
| 12                        | 0                   | 0                    | 0                  | 4                | 4                      | 9                     | 1              | 7            | 0           | Slip or Trip       | Night   | Clear               | Residential    | Germany        | 1              |
| 20                        | 0                   | 0                    | 0                  | 9                | 3                      | 9                     | 1              | 30           | 1           | Electrical Contact | Day     | Clear               | Infrastructure | Brazil         | 1              |
| 8                         | 1                   | 0                    | 0                  | 1                | 4                      | 8                     | 1              | 36           | 1           | Trench Collapse    | Day     | Wind                | Commercial     | United Kingdom | 0              |
| 0                         | 0                   | 1                    | 1                  | 8                |                        |                       | 0              | 8            | 0           | Fall from Height   | Day     | Heat                | Commercial     | South Africa   | 1              |
| 6                         | 1                   | 0                    | 0                  | 10               | 4                      | 9                     | 1              | 15           | 0           | Electrical Contact | Day     | Clear               | Industrial     | United States  | 0              |

```text
Objective 4: late material delivery  (2918 rows x 16 columns)
```

| log_quantity   | unit_price_usd   | lead_time_days   | order_month   | order_year   | rainy_season   | international_supplier   | reliability_rating   | vendor_safety_rating   | vendor_years_active   | is_prequalified   | material            | vendor_type   | country       | project_type   | late_delivery   |
|:---------------|:-----------------|:-----------------|:--------------|:-------------|:---------------|:-------------------------|:---------------------|:-----------------------|:----------------------|:------------------|:--------------------|:--------------|:--------------|:---------------|:----------------|
| 4.2627         | 153.99           | 39               | 7             | 2,023        | 0              | 1                        | 4                    | 4                      | 3                     | 1                 | Glass Panels        | Supplier      | United States | Commercial     | 0               |
| 5.9558         | 25.02            | 32               | 1             | 2,026        | 0              | 1                        | 5                    | 4                      | 39                    | 1                 | Electrical Fittings | Supplier      | Australia     | Infrastructure | 0               |
| 4.9972         | 108.89           | 12               | 4             | 2,025        | 0              | 1                        | 4                    | 4                      | 35                    | 1                 | Glass Panels        | Supplier      | Brazil        | Industrial     | 1               |
| 4.852          | 26.28            | 33               | 5             | 2,022        | 0              | 1                        | 5                    | 4                      | 39                    | 1                 | Electrical Fittings | Supplier      | United States | Infrastructure | 0               |
| 5.5175         | 17.3             | 12               | 9             | 2,025        | 0              | 1                        | 5                    | 5                      | 12                    | 1                 | Electrical Fittings | Supplier      | Brazil        | Residential    | 0               |

```text
Objective 5: price and invoice anomalies  (3000 rows x 4 columns)
```

| price_ratio_material   | price_ratio_peer   | log_unit_price   | log_quantity   |
|:-----------------------|:-------------------|:-----------------|:---------------|
| 1.239                  | 1.0707             | 5.0434           | 4.2627         |
| 1.1451                 | 0.9546             | 3.2589           | 5.9558         |
| 0.8761                 | 0.9968             | 4.6995           | 4.9972         |
| 1.2027                 | 1.0648             | 3.3062           | 4.852          |
| 0.8831                 | 0.9315             | 4.7074           | 3.6636         |

```text
Saved 5 analytical base tables to construction_abt/
```

---

## 10. Encode Categorical Columns

One-Hot Encoding turns text into 0/1 columns.
- Missing text gets its own `Unknown` label, so no row is lost.
- A label the model has never seen encodes as all zeros instead of crashing the app.
- The encoders are saved, so new rows in the app are encoded exactly the same way.

### Code: encode categorical columns

```python
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

#### Output

|                                          | rows   | columns_before   | columns_after   | text_columns_encoded   | new_columns   | numeric_missing_left_for_pipeline   |
|:-----------------------------------------|:-------|:-----------------|:----------------|:-----------------------|:--------------|:------------------------------------|
| Objective 1: cost overrun                | 2413   | 17               | 35              | 4                      | 22            | 0                                   |
| Objective 2: schedule delay              | 3237   | 19               | 39              | 4                      | 24            | 290                                 |
| Objective 3: serious safety incident     | 750    | 16               | 37              | 5                      | 26            | 78                                  |
| Objective 4: late material delivery      | 2918   | 16               | 33              | 4                      | 21            | 257                                 |
| Objective 5: price and invoice anomalies | 3000   | 4                | 4               | 0                      | 0             | 0                                   |

| objective                            | column            | levels   | categories                                                                                                 |
|:-------------------------------------|:------------------|:---------|:-----------------------------------------------------------------------------------------------------------|
| Objective 1: cost overrun            | cost_category     | 5        | Equipment, Labour, Materials, Overheads, Subcontractors                                                    |
| Objective 1: cost overrun            | country           | 10       | Australia, Brazil, Germany, India, Nigeria, Saudi Arabia, South Africa, UAE, United Kingdom, United States |
| Objective 1: cost overrun            | project_type      | 4        | Commercial, Industrial, Infrastructure, Residential                                                        |
| Objective 1: cost overrun            | contract_type     | 3        | Cost Plus, Design-Build, Fixed Price                                                                       |
| Objective 2: schedule delay          | task_type         | 7        | Excavation, Finishing, Foundation, Inspection, MEP Installation, Roofing, Structural Frame                 |
| Objective 2: schedule delay          | country           | 10       | Australia, Brazil, Germany, India, Nigeria, Saudi Arabia, South Africa, UAE, United Kingdom, United States |
| Objective 2: schedule delay          | project_type      | 4        | Commercial, Industrial, Infrastructure, Residential                                                        |
| Objective 2: schedule delay          | contract_type     | 3        | Cost Plus, Design-Build, Fixed Price                                                                       |
| Objective 3: serious safety incident | incident_type     | 6        | Electrical Contact, Equipment Collision, Fall from Height, Slip or Trip, Struck by Object, Trench Collapse |
| Objective 3: serious safety incident | shift             | 2        | Day, Night                                                                                                 |
| Objective 3: serious safety incident | weather_condition | 4        | Clear, Heat, Rain, Wind                                                                                    |
| Objective 3: serious safety incident | project_type      | 4        | Commercial, Industrial, Infrastructure, Residential                                                        |
| Objective 3: serious safety incident | country           | 10       | Australia, Brazil, Germany, India, Nigeria, Saudi Arabia, South Africa, UAE, United Kingdom, United States |
| Objective 4: late material delivery  | material          | 5        | Cement, Electrical Fittings, Glass Panels, Steel Rebar, Timber                                             |
| Objective 4: late material delivery  | vendor_type       | 2        | Supplier, Unknown                                                                                          |
| Objective 4: late material delivery  | country           | 10       | Australia, Brazil, Germany, India, Nigeria, Saudi Arabia, South Africa, UAE, United Kingdom, United States |
| Objective 4: late material delivery  | project_type      | 4        | Commercial, Industrial, Infrastructure, Residential                                                        |

```text
An unseen category encodes as all zeros: True
```

| log_budgeted_usd   | budget_share_of_project   | period_year   | period_quarter   | project_budget_usd   | site_area_sqm   | num_floors   | project_planned_days   | manager_experience_years   | high_rise   | is_public   | budget_per_sqm   | cost_category_Equipment   | cost_category_Labour   | cost_category_Materials   | cost_category_Overheads   | cost_category_Subcontractors   | country_Australia   | country_Brazil   | country_Germany   | country_India   | country_Nigeria   | country_Saudi Arabia   | country_South Africa   | country_UAE   | country_United Kingdom   | country_United States   | project_type_Commercial   | project_type_Industrial   | project_type_Infrastructure   | project_type_Residential   | contract_type_Cost Plus   | contract_type_Design-Build   | contract_type_Fixed Price   | overrun_flag   |
|:-------------------|:--------------------------|:--------------|:-----------------|:---------------------|:----------------|:-------------|:-----------------------|:---------------------------|:------------|:------------|:-----------------|:--------------------------|:-----------------------|:--------------------------|:--------------------------|:-------------------------------|:--------------------|:-----------------|:------------------|:----------------|:------------------|:-----------------------|:-----------------------|:--------------|:-------------------------|:------------------------|:--------------------------|:--------------------------|:------------------------------|:---------------------------|:--------------------------|:-----------------------------|:----------------------------|:---------------|
| 13.7825            | 0.0424                    | 2024          | 4                | 23,109,000           | 37173           | 7            | 801                    | 20                         | 0           | 0           | 621.66           | 0                         | 0                      | 1                         | 0                         | 0                              | 0                   | 0                | 1                 | 0               | 0                 | 0                      | 0                      | 0             | 0                        | 0                       | 0                         | 0                         | 0                             | 1                          | 0                         | 1                            | 0                           | 0              |
| 13.3603            | 0.0278                    | 2024          | 4                | 23,109,000           | 37173           | 7            | 801                    | 20                         | 0           | 0           | 621.66           | 0                         | 1                      | 0                         | 0                         | 0                              | 0                   | 0                | 1                 | 0               | 0                 | 0                      | 0                      | 0             | 0                        | 0                       | 0                         | 0                         | 0                             | 1                          | 0                         | 1                            | 0                           | 0              |
| 12.7096            | 0.0145                    | 2024          | 4                | 23,109,000           | 37173           | 7            | 801                    | 20                         | 0           | 0           | 621.66           | 1                         | 0                      | 0                         | 0                         | 0                              | 0                   | 0                | 1                 | 0               | 0                 | 0                      | 0                      | 0             | 0                        | 0                       | 0                         | 0                         | 0                             | 1                          | 0                         | 1                            | 0                           | 1              |
| 13.4313            | 0.0298                    | 2024          | 4                | 23,109,000           | 37173           | 7            | 801                    | 20                         | 0           | 0           | 621.66           | 0                         | 0                      | 0                         | 0                         | 1                              | 0                   | 0                | 1                 | 0               | 0                 | 0                      | 0                      | 0             | 0                        | 0                       | 0                         | 0                         | 0                             | 1                          | 0                         | 1                            | 0                           | 0              |
| 12.2773            | 0.0094                    | 2024          | 4                | 23,109,000           | 37173           | 7            | 801                    | 20                         | 0           | 0           | 621.66           | 0                         | 0                      | 0                         | 1                         | 0                              | 0                   | 0                | 1                 | 0               | 0                 | 0                      | 0                      | 0             | 0                        | 0                       | 0                         | 0                         | 0                             | 1                          | 0                         | 1                            | 0                           | 0              |

```text
Saved 5 encoded tables, encoders.joblib and column_spec.json to construction_encoded/
```

---

## 11. Separate X and Y

For each objective the target becomes `y` and the allowed inputs become `X`. Checks confirm that no target or leakage column is hiding inside `X`.

### Code: separate X and y

```python
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

#### Output

|                                          | rows   | features_in_X   | target_y            | positive_rate_pct   | second_target       | missing_cells_in_X   | projects   |
|:-----------------------------------------|:-------|:----------------|:--------------------|:--------------------|:--------------------|:---------------------|:-----------|
| Objective 1: cost overrun                | 2413   | 34              | overrun_flag        | 46.7                | overrun_pct         | 0                    | 80         |
| Objective 2: schedule delay              | 3237   | 38              | delay_flag          | 40.9                | delay_days          | 290                  | 80         |
| Objective 3: serious safety incident     | 750    | 36              | serious_flag        | 25.1                | -                   | 78                   | 80         |
| Objective 4: late material delivery      | 2918   | 32              | late_delivery       | 45.9                | delivery_delay_days | 257                  | 80         |
| Objective 5: price and invoice anomalies | 3000   | 4               | none (unsupervised) |                     | -                   | 0                    | 80         |

```text
Objective 1: cost overrun: X is 2413 rows x 34 columns
First features: log_budgeted_usd, budget_share_of_project, period_year, period_quarter, project_budget_usd, site_area_sqm ...
y 'overrun_flag': 1287 zeros, 1126 ones

Objective 2: schedule delay: X is 3237 rows x 38 columns
First features: task_planned_days, crew_size, planned_start_month, planned_start_year, rainy_season, reliability_rating ...
y 'delay_flag': 1913 zeros, 1324 ones

Objective 3: serious safety incident: X is 750 rows x 36 columns
First features: worker_experience_years, ppe_compliant_yes, work_at_height_yes, no_ppe_at_height, incident_month, vendor_safety_rating ...
y 'serious_flag': 562 zeros, 188 ones

Objective 4: late material delivery: X is 2918 rows x 32 columns
First features: log_quantity, unit_price_usd, lead_time_days, order_month, order_year, rainy_season ...
y 'late_delivery': 1579 zeros, 1339 ones

Objective 5: price and invoice anomalies: X is 3000 rows x 4 columns
First features: price_ratio_material, price_ratio_peer, log_unit_price, log_quantity ...

Saved X, y, second targets and project groups to construction_xy/
```

---

## 12. Train Test Split

The split keeps **whole projects together** (`GroupShuffleSplit`): a project is either entirely in training or entirely in testing. The test set is 16 projects the model has never seen. A random row split would let the model memorise a project and make the results look better than they are.

### Code: train test split

```python
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

#### Output

|                                          | train_rows   | test_rows   | train_projects   | test_projects   | train_positive_pct   | test_positive_pct   | seed   | test_share_pct   |
|:-----------------------------------------|:-------------|:------------|:-----------------|:----------------|:---------------------|:--------------------|:-------|:-----------------|
| Objective 1: cost overrun                | 1931         | 482         | 64               | 16              | 46.7                 | 46.5                | 222    | 20               |
| Objective 2: schedule delay              | 2591         | 646         | 64               | 16              | 40.9                 | 40.9                | 54     | 20               |
| Objective 3: serious safety incident     | 598          | 152         | 64               | 16              | 25.1                 | 25                  | 51     | 20.3             |
| Objective 4: late material delivery      | 2336         | 582         | 64               | 16              | 45.9                 | 45.9                | 140    | 19.9             |
| Objective 5: price and invoice anomalies | 3000         | 0           | 80               | 0               |                      |                     |        | 0                |

```text
No project appears in both train and test, for every objective with a target.
Objective 5 keeps all orders together (no target to split on).

Saved the split tables to construction_split/
```

---

## 13. Feature Scaling of Numerical Columns

A median imputer plus `StandardScaler` for numeric columns, and a most-frequent imputer for 0/1 columns. Everything is fitted on the training rows only, so nothing from the test set leaks in.

### Code: feature scaling

```python
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

#### Output

|                                          | features   | scaled_numeric   | left_as_0_or_1   | gaps_filled_train   | gaps_filled_test   | train_mean   | train_std   |
|:-----------------------------------------|:-----------|:-----------------|:-----------------|:--------------------|:-------------------|:-------------|:------------|
| Objective 1: cost overrun                | 34         | 10               | 24               | 0                   | 0                  | 0            | 1           |
| Objective 2: schedule delay              | 38         | 11               | 27               | 235                 | 55                 | 0            | 1           |
| Objective 3: serious safety incident     | 36         | 5                | 31               | 70                  | 8                  | 0            | 1           |
| Objective 4: late material delivery      | 32         | 8                | 24               | 205                 | 52                 | 0            | 1           |
| Objective 5: price and invoice anomalies | 4          | 4                | 0                | 0                   | 0                  | 0            | 1           |

```text
Before and after scaling, first five scaled columns of Objective 1: cost overrun:
```

|                         | before_mean   | before_std   | after_mean   | after_std   | test_mean_after   |
|:------------------------|:--------------|:-------------|:-------------|:------------|:------------------|
| log_budgeted_usd        | 13.07         | 1.02         | 0            | 1           | 0.076             |
| budget_share_of_project | 0.03          | 0.02         | 0            | 1           | 0.002             |
| period_year             | 2,023.68      | 1.09         | 0            | 1           | 0.22              |
| period_quarter          | 2.54          | 1.13         | 0            | 1           | -0.014            |
| project_budget_usd      | 23,422,423.61 | 18,041,087.7 | 0            | 1           | 0.11              |

```text
Test columns are scaled with the TRAIN averages, so their means are close to 0 but not exactly 0.

Saved scaled tables, scalers.joblib and feature_lists.json to construction_scaled/
```

---

## 14. Train Model

For each classification objective I compared four models, from the simplest to the strongest, using GroupKFold cross-validation (whole projects held out in each fold) and average precision.

**Logistic Regression won all four times.** The models are simple, fast and explainable, which is what a site manager needs to trust them.

Alert cut-offs were chosen on out-of-fold training predictions, never on the test set:

| Objective | Alert cut-off |
|-----------|:-------------:|
| 1. Cost overrun | 0.36 |
| 2. Schedule delay | 0.28 |
| 3. Site safety | 0.37 |
| 4. Late delivery | 0.46 |

**Objective 5** uses an **Isolation Forest** (500 trees, 5% contamination) on four price features. It needs no labels: it learns what a normal order looks like and scores how far each order sits from it.

### Code: train model, Objective 1: Control project cost overruns

```python
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

#### Output

```text
Objective 1: cost overrun: 1931 training rows, 34 features, 46.7% overrun
```

|                                              | roc_auc   | roc_auc_std   | average_precision   | precision   | recall   | f1    | seconds   |
|:---------------------------------------------|:----------|:--------------|:--------------------|:------------|:---------|:------|:----------|
| Baseline (always predicts the common answer) | 0.5       | 0             | 0.468               | 0           | 0        | 0     | 0.039     |
| Logistic Regression                          | 0.746     | 0.031         | 0.714               | 0.641       | 0.692    | 0.663 | 0.07      |
| Random Forest                                | 0.697     | 0.01          | 0.652               | 0.615       | 0.628    | 0.615 | 2.439     |
| Gradient Boosting                            | 0.67      | 0.019         | 0.621               | 0.601       | 0.603    | 0.591 | 0.954     |

```text
Best model on cross-validation (average precision): Logistic Regression
The test set has not been touched yet. It is used in the next two steps.
```

### Code: train model, Objective 2: Reduce schedule delays

```python
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

#### Output

```text
Objective 2: schedule delay: 2591 training rows, 38 features, 40.9% delayed tasks
```

|                                              | roc_auc   | roc_auc_std   | average_precision   | precision   | recall   | f1    | seconds   |
|:---------------------------------------------|:----------|:--------------|:--------------------|:------------|:---------|:------|:----------|
| Baseline (always predicts the common answer) | 0.5       | 0             | 0.409               | 0           | 0        | 0     | 0.037     |
| Logistic Regression                          | 0.776     | 0.023         | 0.712               | 0.667       | 0.572    | 0.615 | 0.075     |
| Random Forest                                | 0.771     | 0.015         | 0.704               | 0.683       | 0.52     | 0.589 | 2.779     |
| Gradient Boosting                            | 0.76      | 0.014         | 0.688               | 0.652       | 0.563    | 0.602 | 1.205     |

```text
Best model on cross-validation (average precision): Logistic Regression
Training tasks come from 64 projects; every fold holds out whole projects.
The test set has not been touched yet. It is used in the next two steps.
```

### Code: train model, Objective 3: Improve site safety

```python
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

#### Output

```text
Objective 3: serious safety incident: 598 training rows, 36 features, 25.1% serious incidents
```

|                                              | roc_auc   | roc_auc_std   | average_precision   | precision   | recall   | f1    | seconds   |
|:---------------------------------------------|:----------|:--------------|:--------------------|:------------|:---------|:------|:----------|
| Baseline (always predicts the common answer) | 0.5       | 0             | 0.251               | 0           | 0        | 0     | 0.04      |
| Logistic Regression                          | 0.863     | 0.026         | 0.667               | 0.661       | 0.553    | 0.585 | 0.061     |
| Random Forest                                | 0.862     | 0.03          | 0.643               | 0.742       | 0.326    | 0.432 | 2.037     |
| Gradient Boosting                            | 0.86      | 0.026         | 0.652               | 0.639       | 0.585    | 0.601 | 0.799     |

```text
Best model on cross-validation (average precision): Logistic Regression
Training incidents come from 64 projects; every fold holds out whole projects.
The test set has not been touched yet. It is used in the next two steps.
```

### Code: train model, Objective 4: Make material procurement more reliable

```python
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

#### Output

```text
Objective 4: late material delivery: 2336 training rows, 32 features, 45.9% late deliveries
```

|                                              | roc_auc   | roc_auc_std   | average_precision   | precision   | recall   | f1    | seconds   |
|:---------------------------------------------|:----------|:--------------|:--------------------|:------------|:---------|:------|:----------|
| Baseline (always predicts the common answer) | 0.5       | 0             | 0.459               | 0           | 0        | 0     | 0.042     |
| Logistic Regression                          | 0.631     | 0.025         | 0.588               | 0.576       | 0.484    | 0.521 | 0.081     |
| Random Forest                                | 0.631     | 0.023         | 0.581               | 0.568       | 0.464    | 0.509 | 2.422     |
| Gradient Boosting                            | 0.615     | 0.023         | 0.559               | 0.544       | 0.502    | 0.52  | 1.001     |

```text
Best model on cross-validation (average precision): Logistic Regression
Training orders come from 64 projects; every fold holds out whole projects.
The test set has not been touched yet. It is used in the next two steps.
```

### Code: train model, Objective 5: Detect procurement price and invoice anomalies

```python
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

#### Output

```text
Objective 5: price and invoice anomalies: 3000 orders, 4 features ['price_ratio_material', 'price_ratio_peer', 'log_unit_price', 'log_quantity']
No labels are used. The model learns what a normal order looks like and scores how unusual each one is.
```

|                                           | flagged   | score_stability_across_seeds   | overlap_with_local_outlier_method   | flagged_median_abs_log_ratio   | normal_median_abs_log_ratio   | seconds   |
|:------------------------------------------|:----------|:-------------------------------|:------------------------------------|:-------------------------------|:------------------------------|:----------|
| Isolation Forest, 200 trees, 256 samples  | 150       | 0.981                          | 0.316                               | 0.335                          | 0.093                         | 1.492     |
| Isolation Forest, 500 trees, 256 samples  | 150       | 0.992                          | 0.304                               | 0.332                          | 0.093                         | 3.357     |
| Isolation Forest, 500 trees, 1024 samples | 150       | 0.992                          | 0.333                               | 0.341                          | 0.093                         | 3.616     |

```text
Chosen model (most stable scores across random seeds): Isolation Forest, 500 trees, 256 samples
150 of 3000 orders flagged (5.0%) | anomaly score cut-off 0.538

Flagged orders next to normal ones, in real units (medians):
```

|         | price_ratio_material   | price_ratio_peer   | log_unit_price   | log_quantity   |
|:--------|:-----------------------|:-------------------|:-----------------|:---------------|
| flagged | 1.28                   | 1.14               | 4.75             | 5.34           |
| normal  | 1                      | 1                  | 3.25             | 5              |

```text
Top 10 most unusual orders:
```

|      | order_id   | project_id   | anomaly_score   | flagged   | score_rank_pct   |
|:-----|:-----------|:-------------|:----------------|:----------|:-----------------|
| 1939 | ORD-41939  | PRJ-1037     | 0.7995          | True      | 1                |
| 209  | ORD-40209  | PRJ-1056     | 0.7983          | True      | 1                |
| 1841 | ORD-41841  | PRJ-1007     | 0.787           | True      | 0.999            |
| 124  | ORD-40124  | PRJ-1033     | 0.7826          | True      | 0.999            |
| 99   | ORD-40099  | PRJ-1033     | 0.7817          | True      | 0.999            |
| 2612 | ORD-42612  | PRJ-1042     | 0.7742          | True      | 0.998            |
| 706  | ORD-40706  | PRJ-1061     | 0.7735          | True      | 0.998            |
| 2132 | ORD-42132  | PRJ-1023     | 0.7734          | True      | 0.998            |
| 2170 | ORD-42170  | PRJ-1058     | 0.7727          | True      | 0.997            |
| 1025 | ORD-41025  | PRJ-1050     | 0.7718          | True      | 0.997            |

```text
The answer key (is_anomaly) has not been opened. It is used only at the final scoring step.
```

---

## 15. Predict X_test

Each model scores the unseen test projects, and the probabilities are turned into alerts with the cut-offs above.

### Code: predict X_test, Objective 1: Control project cost overruns

```python
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

#### Output

```text
482 test cost lines scored by 4 models | real overruns: 224 (46.5%) | 16 test projects the models have never seen

A useful model gives real overruns a clearly higher average chance (a bigger gap is better):
```

|                                              | avg chance, real overruns   | avg chance, other lines   | gap   |
|:---------------------------------------------|:----------------------------|:--------------------------|:------|
| Baseline (always predicts the common answer) | 0.467                       | 0.467                     | 0     |
| Logistic Regression                          | 0.68                        | 0.329                     | 0.351 |
| Random Forest                                | 0.566                       | 0.394                     | 0.173 |
| Gradient Boosting                            | 0.611                       | 0.344                     | 0.267 |

```text
First 10 test cost lines (each model column = chance of a more than 15% overrun):
```

|    | ledger_id   | project_id   | actual_overrun_flag   | actual_overrun_pct   | Baseline (always predicts the common answer)   | Logistic Regression   | Random Forest   | Gradient Boosting   |
|:---|:------------|:-------------|:----------------------|:---------------------|:-----------------------------------------------|:----------------------|:----------------|:--------------------|
| 0  | LED-60000   | PRJ-1000     | 0                     | 5                    | 0.467                                          | 0.532                 | 0.573           | 0.718               |
| 1  | LED-60001   | PRJ-1000     | 0                     | 7                    | 0.467                                          | 0.319                 | 0.395           | 0.324               |
| 2  | LED-60002   | PRJ-1000     | 1                     | 17.2                 | 0.467                                          | 0.216                 | 0.261           | 0.281               |
| 3  | LED-60003   | PRJ-1000     | 0                     | 12.7                 | 0.467                                          | 0.38                  | 0.444           | 0.442               |
| 4  | LED-60004   | PRJ-1000     | 0                     | 12.6                 | 0.467                                          | 0.087                 | 0.182           | 0.227               |
| 5  | LED-60005   | PRJ-1000     | 0                     | 8.9                  | 0.467                                          | 0.41                  | 0.531           | 0.552               |
| 6  | LED-60006   | PRJ-1000     | 0                     | 5.4                  | 0.467                                          | 0.218                 | 0.342           | 0.245               |
| 7  | LED-60007   | PRJ-1000     | 0                     | -3.6                 | 0.467                                          | 0.148                 | 0.269           | 0.216               |
| 8  | LED-60008   | PRJ-1000     | 0                     | -3.5                 | 0.467                                          | 0.274                 | 0.42            | 0.312               |
| 9  | LED-60009   | PRJ-1000     | 0                     | -1.8                 | 0.467                                          | 0.057                 | 0.153           | 0.06                |

### Code: predict X_test, Objective 2: Reduce schedule delays

```python
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

#### Output

```text
646 test tasks scored by 4 models | real delays: 264 (40.9%) | 16 test projects the models have never seen

A useful model gives real delays a clearly higher average chance (a bigger gap is better):
```

|                                              | avg chance, real delays   | avg chance, on-time tasks   | gap   |
|:---------------------------------------------|:--------------------------|:----------------------------|:------|
| Baseline (always predicts the common answer) | 0.409                     | 0.409                       | 0     |
| Logistic Regression                          | 0.601                     | 0.321                       | 0.28  |
| Random Forest                                | 0.535                     | 0.336                       | 0.198 |
| Gradient Boosting                            | 0.59                      | 0.305                       | 0.285 |

```text
First 10 test tasks (each model column = chance of a delay):
```

|    | task_id   | project_id   | actual_delay_flag   | actual_delay_days   | Baseline (always predicts the common answer)   | Logistic Regression   | Random Forest   | Gradient Boosting   |
|:---|:----------|:-------------|:--------------------|:--------------------|:-----------------------------------------------|:----------------------|:----------------|:--------------------|
| 1  | TSK-20002 | PRJ-1000     | 0                   | 4                   | 0.409                                          | 0.229                 | 0.301           | 0.057               |
| 11 | TSK-20014 | PRJ-1045     | 0                   | 4                   | 0.409                                          | 0.355                 | 0.354           | 0.355               |
| 24 | TSK-20030 | PRJ-1029     | 1                   | 10                  | 0.409                                          | 0.381                 | 0.386           | 0.518               |
| 29 | TSK-20036 | PRJ-1072     | 1                   | 17                  | 0.409                                          | 0.409                 | 0.375           | 0.317               |
| 41 | TSK-20049 | PRJ-1072     | 0                   | 1                   | 0.409                                          | 0.161                 | 0.278           | 0.174               |
| 46 | TSK-20055 | PRJ-1063     | 0                   | 4                   | 0.409                                          | 0.193                 | 0.24            | 0.083               |
| 49 | TSK-20058 | PRJ-1040     | 1                   | 15                  | 0.409                                          | 0.905                 | 0.697           | 0.777               |
| 53 | TSK-20063 | PRJ-1028     | 0                   | 0                   | 0.409                                          | 0.519                 | 0.544           | 0.735               |
| 54 | TSK-20064 | PRJ-1046     | 1                   | 6                   | 0.409                                          | 0.385                 | 0.398           | 0.446               |
| 60 | TSK-20073 | PRJ-1072     | 1                   | 13                  | 0.409                                          | 0.506                 | 0.477           | 0.59                |

### Code: predict X_test, Objective 3: Improve site safety

```python
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

#### Output

```text
152 test incidents scored by 4 models | serious incidents: 38 (25.0%) | 16 test projects the models have never seen

A useful model gives serious incidents a clearly higher average chance (a bigger gap is better):
```

|                                              | avg chance, serious incidents   | avg chance, other incidents   | gap   |
|:---------------------------------------------|:--------------------------------|:------------------------------|:------|
| Baseline (always predicts the common answer) | 0.251                           | 0.251                         | 0     |
| Logistic Regression                          | 0.499                           | 0.164                         | 0.335 |
| Random Forest                                | 0.412                           | 0.204                         | 0.207 |
| Gradient Boosting                            | 0.455                           | 0.166                         | 0.289 |

```text
First 10 test incidents (each model column = chance of being serious):
```

|    | incident_id   | project_id   | actual_serious_flag   | Baseline (always predicts the common answer)   | Logistic Regression   | Random Forest   | Gradient Boosting   |
|:---|:--------------|:-------------|:----------------------|:-----------------------------------------------|:----------------------|:----------------|:--------------------|
| 0  | INC-80000     | PRJ-1000     | 1                     | 0.251                                          | 0.091                 | 0.154           | 0.006               |
| 26 | INC-80026     | PRJ-1029     | 1                     | 0.251                                          | 0.474                 | 0.326           | 0.301               |
| 28 | INC-80028     | PRJ-1012     | 1                     | 0.251                                          | 0.791                 | 0.576           | 0.637               |
| 32 | INC-80032     | PRJ-1064     | 0                     | 0.251                                          | 0.384                 | 0.376           | 0.609               |
| 37 | INC-80037     | PRJ-1044     | 0                     | 0.251                                          | 0.016                 | 0.04            | 0.002               |
| 42 | INC-80042     | PRJ-1064     | 0                     | 0.251                                          | 0.007                 | 0.092           | 0.011               |
| 43 | INC-80043     | PRJ-1015     | 0                     | 0.251                                          | 0.109                 | 0.132           | 0.053               |
| 57 | INC-80057     | PRJ-1026     | 1                     | 0.251                                          | 0.205                 | 0.255           | 0.034               |
| 74 | INC-80074     | PRJ-1026     | 0                     | 0.251                                          | 0.072                 | 0.076           | 0.041               |
| 78 | INC-80078     | PRJ-1044     | 0                     | 0.251                                          | 0.689                 | 0.627           | 0.915               |

### Code: predict X_test, Objective 4: Make material procurement more reliable

```python
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

#### Output

```text
582 test orders scored by 4 models | late orders: 267 (45.9%) | 16 test projects the models have never seen

A useful model gives late orders a clearly higher average chance (a bigger gap is better):
```

|                                              | avg chance, late orders   | avg chance, on-time orders   | gap   |
|:---------------------------------------------|:--------------------------|:-----------------------------|:------|
| Baseline (always predicts the common answer) | 0.459                     | 0.459                        | 0     |
| Logistic Regression                          | 0.486                     | 0.432                        | 0.054 |
| Random Forest                                | 0.472                     | 0.421                        | 0.051 |
| Gradient Boosting                            | 0.489                     | 0.417                        | 0.072 |

```text
First 10 test orders (each model column = chance of being late):
```

|    | order_id   | project_id   | actual_late_flag   | actual_delay_days   | Baseline (always predicts the common answer)   | Logistic Regression   | Random Forest   | Gradient Boosting   |
|:---|:-----------|:-------------|:-------------------|:--------------------|:-----------------------------------------------|:----------------------|:----------------|:--------------------|
| 8  | ORD-40010  | PRJ-1061     | 1                  | 3                   | 0.459                                          | 0.661                 | 0.607           | 0.863               |
| 10 | ORD-40012  | PRJ-1077     | 0                  | 0                   | 0.459                                          | 0.403                 | 0.37            | 0.409               |
| 11 | ORD-40013  | PRJ-1041     | 1                  | 10                  | 0.459                                          | 0.806                 | 0.764           | 0.832               |
| 16 | ORD-40019  | PRJ-1077     | 0                  | -2                  | 0.459                                          | 0.391                 | 0.457           | 0.33                |
| 22 | ORD-40025  | PRJ-1041     | 0                  | 0                   | 0.459                                          | 0.384                 | 0.362           | 0.523               |
| 28 | ORD-40032  | PRJ-1014     | 0                  | -1                  | 0.459                                          | 0.447                 | 0.442           | 0.466               |
| 31 | ORD-40035  | PRJ-1012     | 0                  | -2                  | 0.459                                          | 0.42                  | 0.358           | 0.248               |
| 42 | ORD-40047  | PRJ-1019     | 1                  | 3                   | 0.459                                          | 0.643                 | 0.544           | 0.506               |
| 45 | ORD-40050  | PRJ-1045     | 1                  | 3                   | 0.459                                          | 0.636                 | 0.568           | 0.537               |
| 47 | ORD-40052  | PRJ-1007     | 0                  | -1                  | 0.459                                          | 0.583                 | 0.408           | 0.432               |

### Code: predict X_test, Objective 5: Detect procurement price and invoice anomalies

```python
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

#### Output

```text
3,000 orders scored | 150 flagged for review (5.0%) | 80 projects
Flag cut-off (anomaly score): 0.538

Orders by risk band:
```

|                       | orders   |
|:----------------------|:---------|
| Top 1%                | 30       |
| Next 4% (flagged)     | 120      |
| Next 15% (watch list) | 450      |
| Normal                | 2400     |

```text
Why the flagged orders were flagged:
```

|                                   | flagged_orders   |
|:----------------------------------|:-----------------|
| priced above the material norm    | 88               |
| priced below the material norm    | 39               |
| unusual order quantity            | 19               |
| priced below similar orders       | 3                |
| unusual mix of price and quantity | 1                |

```text
Projects with the most flagged orders:
```

|          | flagged_orders   | share_of_flags   |
|:---------|:-----------------|:-----------------|
| PRJ-1051 | 7                | 0.047            |
| PRJ-1042 | 7                | 0.047            |
| PRJ-1057 | 7                | 0.047            |
| PRJ-1013 | 5                | 0.033            |
| PRJ-1009 | 5                | 0.033            |
| PRJ-1026 | 5                | 0.033            |
| PRJ-1007 | 5                | 0.033            |
| PRJ-1054 | 4                | 0.027            |

```text
Top 10 orders to review first:
```

|      | order_id   | project_id   | anomaly_score   | price_ratio_material   | price_ratio_peer   | unit_price_usd   | main_reason                    |
|:-----|:-----------|:-------------|:----------------|:-----------------------|:-------------------|:-----------------|:-------------------------------|
| 1939 | ORD-41939  | PRJ-1037     | 0.7995          | 6.57                   | 5.64               | 4,675.01         | priced above the material norm |
| 209  | ORD-40209  | PRJ-1056     | 0.7983          | 5.97                   | 6.16               | 4,248.26         | priced above the material norm |
| 1841 | ORD-41841  | PRJ-1007     | 0.787           | 5.84                   | 5                  | 4,153.72         | priced above the material norm |
| 124  | ORD-40124  | PRJ-1033     | 0.7826          | 6.85                   | 6.05               | 59.21            | priced above the material norm |
| 99   | ORD-40099  | PRJ-1033     | 0.7817          | 6.74                   | 5.95               | 58.21            | priced above the material norm |
| 2612 | ORD-42612  | PRJ-1042     | 0.7742          | 6.07                   | 5.59               | 52.48            | priced above the material norm |
| 706  | ORD-40706  | PRJ-1061     | 0.7735          | 6.17                   | 5.78               | 134.92           | priced above the material norm |
| 2132 | ORD-42132  | PRJ-1023     | 0.7734          | 5.14                   | 5.64               | 44.37            | priced above the material norm |
| 2170 | ORD-42170  | PRJ-1058     | 0.7727          | 6.02                   | 5.88               | 131.62           | priced above the material norm |
| 1025 | ORD-41025  | PRJ-1050     | 0.7718          | 6.01                   | 5.31               | 51.94            | priced above the material norm |

```text
Saved all_orders_scored.csv and review_queue.csv to construction_charts/objective_5_price_anomalies/. The answer key has still not been opened.
```

---

## 16. Compare X and Y

Predicted and actual outcomes side by side, to see where each model agrees with reality and where it misses.

### Code: compare X and y, Objective 1: Control project cost overruns

```python
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

#### Output

![Objective 1: predicted vs actual](images/s13_01.png)

```text
Overall overrun rate in training: 46.7%

Inputs in the top 8 of at least 2 of 3 models:
```

|                          | Logistic Regression   | Random Forest   | Gradient Boosting   | average   |
|:-------------------------|:----------------------|:----------------|:--------------------|:----------|
| contract_type            | 0.0829                | 0.0772          | 0.088               | 0.0827    |
| country                  | 0.0817                | 0.027           | 0.0721              | 0.0603    |
| cost_category            | 0.0528                | 0.0346          | 0.0294              | 0.0389    |
| num_floors               | 0.003                 | 0.0076          | 0.0262              | 0.0123    |
| project_planned_days     | 0.0004                | 0.0113          | 0.0239              | 0.0119    |
| high_rise                | 0.0087                | 0.0035          | 0                   | 0.0041    |
| manager_experience_years | 0.0045                | 0.0023          | 0.0053              | 0.004     |

```text
What the data says about the ten most important inputs:
```

| input                    | compared                                    | rate_at_low_pct   | rate_at_high_pct   | gap_points   |
|:-------------------------|:--------------------------------------------|:------------------|:-------------------|:-------------|
| contract_type            | Fixed Price vs Cost Plus                    | 38                | 64.3               | 26.3         |
| country                  | United States vs Nigeria                    | 24.8              | 67.4               | 42.6         |
| cost_category            | Overheads vs Materials                      | 20.6              | 68.6               | 48           |
| budget_share_of_project  | lowest quarter vs highest quarter of values | 28.4              | 59.5               | 31.1         |
| num_floors               | lowest quarter vs highest quarter of values | 38.7              | 47.6               | 8.9          |
| project_planned_days     | lowest quarter vs highest quarter of values | 39.1              | 47.3               | 8.2          |
| period_quarter           | lowest quarter vs highest quarter of values | 43.8              | 53                 | 9.2          |
| high_rise                | value 0 vs value 1                          | 35.4              | 53.6               | 18.2         |
| manager_experience_years | lowest quarter vs highest quarter of values | 50.3              | 44.4               | -5.9         |
| project_type             | Commercial vs Infrastructure                | 40.3              | 58.1               | 17.8         |

```text
Importance of every input, by model (average drop in ROC AUC):
```

|                          | Logistic Regression   | Random Forest   | Gradient Boosting   | average   |
|:-------------------------|:----------------------|:----------------|:--------------------|:----------|
| contract_type            | 0.0829                | 0.0772          | 0.088               | 0.0827    |
| country                  | 0.0817                | 0.027           | 0.0721              | 0.0603    |
| cost_category            | 0.0528                | 0.0346          | 0.0294              | 0.0389    |
| budget_share_of_project  | 0.001                 | 0.0017          | 0.0415              | 0.0147    |
| num_floors               | 0.003                 | 0.0076          | 0.0262              | 0.0123    |
| project_planned_days     | 0.0004                | 0.0113          | 0.0239              | 0.0119    |
| period_quarter           | 0.0022                | 0.0022          | 0.0144              | 0.0063    |
| high_rise                | 0.0087                | 0.0035          | 0                   | 0.0041    |
| manager_experience_years | 0.0045                | 0.0023          | 0.0053              | 0.004     |
| project_type             | 0.0053                | -0.0001         | 0.0047              | 0.0033    |
| period_year              | 0                     | 0.0036          | 0.0038              | 0.0025    |
| log_budgeted_usd         | 0.0084                | -0.008          | 0.0053              | 0.0019    |
| project_budget_usd       | -0.0011               | 0.0038          | 0.0019              | 0.0015    |
| budget_per_sqm           | 0.0065                | -0.0027         | 0.0008              | 0.0015    |
| is_public                | 0.0008                | 0.0019          | -0.0008             | 0.0006    |
| site_area_sqm            | -0.0002               | -0.002          | 0.0028              | 0.0002    |

### Code: compare X and y, Objective 2: Reduce schedule delays

```python
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

#### Output

![Objective 2: predicted vs actual](images/s17_01.png)

```text
Overall delay rate in training: 40.9%

Inputs in the top 8 of at least 2 of 3 models:
```

|                      | Logistic Regression   | Random Forest   | Gradient Boosting   | average   |
|:---------------------|:----------------------|:----------------|:--------------------|:----------|
| crew_size            | 0.1267                | 0.1183          | 0.1243              | 0.1231    |
| rainy_season         | 0.087                 | 0.0835          | 0.0801              | 0.0835    |
| task_type            | 0.0286                | 0.0235          | 0.0237              | 0.0253    |
| task_planned_days    | 0.0103                | 0.0102          | 0.0198              | 0.0134    |
| reliability_rating   | 0.014                 | 0.0103          | 0.0161              | 0.0134    |
| vendor_safety_rating | 0.008                 | 0.0095          | 0.0177              | 0.0117    |
| vendor_years_active  | -0.0012               | 0.0041          | 0.0149              | 0.0059    |
| project_planned_days | 0.0024                | 0.0011          | 0.0059              | 0.0031    |

```text
What the data says about the ten most important inputs:
```

| input                | compared                                    | rate_at_low_pct   | rate_at_high_pct   | gap_points   |
|:---------------------|:--------------------------------------------|:------------------|:-------------------|:-------------|
| crew_size            | lowest quarter vs highest quarter of values | 59.3              | 24.2               | -35.1        |
| rainy_season         | value 0 vs value 1                          | 35.9              | 75.9               | 40           |
| task_type            | Inspection vs Structural Frame              | 20.4              | 54.7               | 34.3         |
| task_planned_days    | lowest quarter vs highest quarter of values | 30.7              | 51.6               | 20.9         |
| reliability_rating   | lowest quarter vs highest quarter of values | 44                | 37.7               | -6.3         |
| vendor_safety_rating | lowest quarter vs highest quarter of values | 49.6              | 37.1               | -12.5        |
| vendor_years_active  | lowest quarter vs highest quarter of values | 38.5              | 34.6               | -3.9         |
| project_planned_days | lowest quarter vs highest quarter of values | 37.6              | 43.8               | 6.2          |
| site_area_sqm        | lowest quarter vs highest quarter of values | 42.2              | 43.1               | 0.9          |
| country              | Saudi Arabia vs Brazil                      | 31                | 55                 | 24           |

```text
Importance of every input, by model (average drop in ROC AUC):
```

|                          | Logistic Regression   | Random Forest   | Gradient Boosting   | average   |
|:-------------------------|:----------------------|:----------------|:--------------------|:----------|
| crew_size                | 0.1267                | 0.1183          | 0.1243              | 0.1231    |
| rainy_season             | 0.087                 | 0.0835          | 0.0801              | 0.0835    |
| task_type                | 0.0286                | 0.0235          | 0.0237              | 0.0253    |
| task_planned_days        | 0.0103                | 0.0102          | 0.0198              | 0.0134    |
| reliability_rating       | 0.014                 | 0.0103          | 0.0161              | 0.0134    |
| vendor_safety_rating     | 0.008                 | 0.0095          | 0.0177              | 0.0117    |
| vendor_years_active      | -0.0012               | 0.0041          | 0.0149              | 0.0059    |
| project_planned_days     | 0.0024                | 0.0011          | 0.0059              | 0.0031    |
| site_area_sqm            | 0.0012                | 0.0028          | 0.0031              | 0.0024    |
| country                  | 0.005                 | -0.001          | 0.0007              | 0.0016    |
| project_type             | 0.0023                | 0.0009          | 0.0008              | 0.0013    |
| num_floors               | -0.0001               | 0.0027          | 0.0004              | 0.001     |
| planned_start_year       | -0.0001               | 0.0012          | 0.0008              | 0.0006    |
| vendor_known             | -0.0001               | 0               | 0                   | 0         |
| manager_experience_years | 0.0001                | -0.0009         | 0                   | -0.0003   |
| is_prequalified          | 0.0001                | -0.0013         | 0                   | -0.0004   |
| planned_start_month      | -0.0001               | -0.0016         | 0.0003              | -0.0005   |
| contract_type            | -0.0012               | 0.0004          | -0.0019             | -0.0009   |

### Code: compare X and y, Objective 3: Improve site safety

```python
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

#### Output

![Objective 3: predicted vs actual](images/s21_01.png)

```text
Share of serious incidents in training: 25.1%

Inputs in the top 8 of at least 2 of 3 models:
```

|                         | Logistic Regression   | Random Forest   | Gradient Boosting   | average   |
|:------------------------|:----------------------|:----------------|:--------------------|:----------|
| incident_type           | 0.1452                | 0.122           | 0.1495              | 0.1389    |
| worker_experience_years | 0.1101                | 0.1026          | 0.1094              | 0.1074    |
| ppe_compliant_yes       | 0.0482                | 0.019           | 0.0389              | 0.0354    |
| shift                   | 0.0282                | 0.0346          | 0.0361              | 0.033     |
| work_at_height_yes      | 0.0178                | 0.0274          | 0.0172              | 0.0208    |
| vendor_safety_rating    | 0.0165                | 0.0072          | 0.0189              | 0.0142    |

```text
What the data says about the ten most important inputs:
```

| input                   | compared                                    | rate_at_low_pct   | rate_at_high_pct   | gap_points   |
|:------------------------|:--------------------------------------------|:------------------|:-------------------|:-------------|
| incident_type           | Slip or Trip vs Fall from Height            | 2.9               | 53.4               | 50.5         |
| worker_experience_years | lowest quarter vs highest quarter of values | 39.7              | 9.4                | -30.3        |
| ppe_compliant_yes       | value 0 vs value 1                          | 41.2              | 21.1               | -20.1        |
| shift                   | Day vs Night                                | 20.5              | 38.7               | 18.2         |
| work_at_height_yes      | value 0 vs value 1                          | 18.3              | 42.1               | 23.8         |
| vendor_safety_rating    | lowest quarter vs highest quarter of values | 32                | 22.3               | -9.7         |
| country                 | Australia vs Brazil                         | 13.5              | 32.9               | 19.4         |
| no_ppe_at_height        | value 0 vs value 1                          | 22.8              | 59.5               | 36.7         |
| incident_month          | lowest quarter vs highest quarter of values | 24.3              | 29.1               | 4.8          |
| vendor_known            | value 0 vs value 1                          | 21.2              | 25.3               | 4.1          |

```text
Importance of every input, by model (average drop in ROC AUC):
```

|                         | Logistic Regression   | Random Forest   | Gradient Boosting   | average   |
|:------------------------|:----------------------|:----------------|:--------------------|:----------|
| incident_type           | 0.1452                | 0.122           | 0.1495              | 0.1389    |
| worker_experience_years | 0.1101                | 0.1026          | 0.1094              | 0.1074    |
| ppe_compliant_yes       | 0.0482                | 0.019           | 0.0389              | 0.0354    |
| shift                   | 0.0282                | 0.0346          | 0.0361              | 0.033     |
| work_at_height_yes      | 0.0178                | 0.0274          | 0.0172              | 0.0208    |
| vendor_safety_rating    | 0.0165                | 0.0072          | 0.0189              | 0.0142    |
| country                 | -0.001                | 0               | 0.0094              | 0.0028    |
| no_ppe_at_height        | 0.0009                | 0.0022          | 0                   | 0.001     |
| incident_month          | 0.0016                | -0.0053         | 0.0052              | 0.0005    |
| vendor_known            | 0.0003                | 0.0002          | 0                   | 0.0002    |
| project_type            | -0.0003               | -0.0075         | 0.0067              | -0.0004   |
| high_rise               | 0.002                 | -0.0044         | 0                   | -0.0008   |
| weather_condition       | 0.0018                | -0.0017         | -0.0039             | -0.0013   |
| num_floors              | 0.0036                | -0.0095         | -0.0037             | -0.0032   |
| vendor_years_active     | -0.0027               | -0.0129         | -0.0124             | -0.0093   |

### Code: compare X and y, Objective 4: Make material procurement more reliable

```python
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

#### Output

![Objective 4: predicted vs actual](images/s25_01.png)

```text
Share of late deliveries in training: 45.9%

Inputs in the top 8 of at least 2 of 3 models:
```

|                        | Logistic Regression   | Random Forest   | Gradient Boosting   | average   |
|:-----------------------|:----------------------|:----------------|:--------------------|:----------|
| log_quantity           | 0.0301                | 0.0336          | 0.0414              | 0.0351    |
| vendor_safety_rating   | 0.0289                | 0.0203          | 0.0346              | 0.0279    |
| international_supplier | 0.0311                | 0.0165          | 0.0229              | 0.0235    |
| vendor_years_active    | 0.0012                | 0.0088          | 0.0184              | 0.0095    |
| rainy_season           | 0.012                 | 0.0032          | 0.0047              | 0.0066    |
| country                | 0.0016                | 0.01            | 0.0014              | 0.0044    |
| unit_price_usd         | 0.0025                | -0.0019         | 0.0113              | 0.0039    |
| material               | 0.0063                | 0.0041          | -0.0017             | 0.0029    |
| project_type           | -0.008                | 0.0084          | 0                   | 0.0001    |

```text
What the data says about the ten most important inputs:
```

| input                  | compared                                    | rate_at_low_pct   | rate_at_high_pct   | gap_points   |
|:-----------------------|:--------------------------------------------|:------------------|:-------------------|:-------------|
| log_quantity           | lowest quarter vs highest quarter of values | 34.9              | 56.5               | 21.6         |
| vendor_safety_rating   | lowest quarter vs highest quarter of values | 53.7              | 40.7               | -13          |
| international_supplier | value 0 vs value 1                          | 29.1              | 47.9               | 18.8         |
| vendor_years_active    | lowest quarter vs highest quarter of values | 41.6              | 42                 | 0.4          |
| rainy_season           | value 0 vs value 1                          | 44.2              | 56                 | 11.8         |
| country                | South Africa vs Australia                   | 41.6              | 51.3               | 9.7          |
| unit_price_usd         | lowest quarter vs highest quarter of values | 42.8              | 49.1               | 6.3          |
| material               | Electrical Fittings vs Glass Panels         | 40.2              | 54.1               | 13.9         |
| reliability_rating     | lowest quarter vs highest quarter of values | 49.1              | 44.4               | -4.7         |
| project_type           | Infrastructure vs Residential               | 44.6              | 48.2               | 3.6          |

```text
Importance of every input, by model (average drop in ROC AUC):
```

|                        | Logistic Regression   | Random Forest   | Gradient Boosting   | average   |
|:-----------------------|:----------------------|:----------------|:--------------------|:----------|
| log_quantity           | 0.0301                | 0.0336          | 0.0414              | 0.0351    |
| vendor_safety_rating   | 0.0289                | 0.0203          | 0.0346              | 0.0279    |
| international_supplier | 0.0311                | 0.0165          | 0.0229              | 0.0235    |
| vendor_years_active    | 0.0012                | 0.0088          | 0.0184              | 0.0095    |
| rainy_season           | 0.012                 | 0.0032          | 0.0047              | 0.0066    |
| country                | 0.0016                | 0.01            | 0.0014              | 0.0044    |
| unit_price_usd         | 0.0025                | -0.0019         | 0.0113              | 0.0039    |
| material               | 0.0063                | 0.0041          | -0.0017             | 0.0029    |
| reliability_rating     | 0.001                 | -0.0003         | -0.0002             | 0.0002    |
| project_type           | -0.008                | 0.0084          | 0                   | 0.0001    |
| is_prequalified        | 0.0004                | 0.0005          | -0.0007             | 0.0001    |
| vendor_type            | -0.0004               | 0.0001          | -0.0009             | -0.0004   |
| lead_time_days         | -0.0007               | -0.0028         | -0.0013             | -0.0016   |
| order_month            | -0.0009               | 0.0001          | -0.0061             | -0.0023   |
| order_year             | -0.0038               | -0.0071         | -0.0112             | -0.0074   |

### Code: compare X and y, Objective 5: Detect procurement price and invoice anomalies

```python
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

#### Output

![Objective 5: predicted vs actual](images/s29_01.png)

```text
150 flagged orders compared with 2850 normal orders

Flagged vs normal (real units, and the gap in standard-deviation units):
```

|                      | normal_median_real   | flagged_median_real   | gap_in_std_units   |
|:---------------------|:---------------------|:----------------------|:-------------------|
| price_ratio_peer     | 0.999                | 1.142                 | 2.706              |
| price_ratio_material | 0.998                | 1.283                 | 2.607              |
| log_unit_price       | 3.255                | 4.75                  | 0.504              |
| log_quantity         | 4.997                | 5.342                 | 0.193              |

```text
Share of the original flags that disappear when one feature is shuffled:
```

|                      | flags_lost_pct   |
|:---------------------|:-----------------|
| price_ratio_peer     | 42.1             |
| log_quantity         | 41.7             |
| price_ratio_material | 39.9             |
| log_unit_price       | 19.9             |

```text
Flag rate by price vs the material's usual price:
```

|                       | orders   | flagged   | flag_rate   |
|:----------------------|:---------|:----------|:------------|
| below 0.87 (cheap)    | 433      | 39        | 9           |
| 0.87 to 1.15 (normal) | 2037     | 23        | 1.1         |
| 1.15 to 2 (dear)      | 468      | 26        | 5.6         |
| above 2 (very dear)   | 62       | 62        | 100         |

```text
Flag rate by price vs similar orders:
```

|                       | orders   | flagged   | flag_rate   |
|:----------------------|:---------|:----------|:------------|
| below 0.87 (cheap)    | 91       | 20        | 22          |
| 0.87 to 1.15 (normal) | 2763     | 56        | 2           |
| 1.15 to 2 (dear)      | 84       | 12        | 14.3        |
| above 2 (very dear)   | 62       | 62        | 100         |

---

## 17. Evaluate Model

| Objective | Test ROC AUC | Reading |
|-----------|:------------:|---------|
| 1. Cost overrun | **0.83** | Strong. Range across resampled projects: 0.75 to 0.87 |
| 2. Schedule delay | **0.80** | Strong |
| 3. Site safety | **0.86** | Strong |
| 4. Late delivery | 0.60 | Weak. Use as a rough guide only |
| 5. Price anomalies | n/a | Finds all 62 planted anomalies in a review list of 150 orders |


> **Honest note:** the test set is only 16 projects, so exact scores can move. The rankings are more reliable than any single number. For Objective 5, a simple rule (price above 2x the material norm) also finds all 62, so the model's real value is on subtler cases that real invoices may hold.

### Code: evaluate model, Objective 1: Control project cost overruns

```python
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

#### Output

![Objective 1: model evaluation](images/s14_01.png)

```text
Best model: Logistic Regression | alert cut-off 0.36 (chosen on training folds, not on the test set)
Test set: 482 cost lines from 16 unseen projects, 46.5% overran

Scorecard on the test set:
```

|                                              | threshold   | roc_auc   | average_precision   | accuracy   | precision   | recall   | f1    |
|:---------------------------------------------|:------------|:----------|:--------------------|:-----------|:------------|:---------|:------|
| Logistic Regression                          | 0.36        | 0.827     | 0.824               | 0.72       | 0.658       | 0.826    | 0.733 |
| Gradient Boosting                            | 0.5         | 0.8       | 0.782               | 0.716      | 0.687       | 0.714    | 0.7   |
| Random Forest                                | 0.5         | 0.784     | 0.759               | 0.703      | 0.665       | 0.728    | 0.695 |
| Baseline (always predicts the common answer) | 0.5         | 0.5       | 0.465               | 0.535      | 0           | 0        | 0     |

```text
ROC AUC of the best model: 0.827 (95% range when whole projects are resampled: 0.749 to 0.874)
Overruns caught: 185 of 224 (83%) | false alarms: 96 of 258 (37%) | when it flags a line, it is right 66% of the time
The two highest risk bands hold 67% of all real overruns in 40% of the lines

Real overrun rate by risk band:
```

|         | cost_lines   | real_overrun_rate   | avg_predicted_chance   |
|:--------|:-------------|:--------------------|:-----------------------|
| Lowest  | 97           | 0.113               | 0.081                  |
| Low     | 96           | 0.271               | 0.256                  |
| Middle  | 96           | 0.385               | 0.482                  |
| High    | 96           | 0.635               | 0.73                   |
| Highest | 97           | 0.918               | 0.913                  |

```text
Confusion matrix:
```

|                        | Predicted within budget   | Predicted overrun   |
|:-----------------------|:--------------------------|:--------------------|
| Actually within budget | 162                       | 96                  |
| Actually overran       | 39                        | 185                 |

### Code: evaluate model, Objective 2: Reduce schedule delays

```python
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

#### Output

![Objective 2: model evaluation](images/s18_01.png)

```text
Best model: Logistic Regression | alert cut-off 0.28 (chosen on training folds, not on the test set)
Test set: 646 tasks from 16 unseen projects, 40.9% delayed

Scorecard on the test set:
```

|                                              | threshold   | roc_auc   | average_precision   | accuracy   | precision   | recall   | f1    |
|:---------------------------------------------|:------------|:----------|:--------------------|:-----------|:------------|:---------|:------|
| Random Forest                                | 0.5         | 0.806     | 0.748               | 0.745      | 0.714       | 0.625    | 0.667 |
| Logistic Regression                          | 0.28        | 0.805     | 0.755               | 0.653      | 0.547       | 0.875    | 0.673 |
| Gradient Boosting                            | 0.5         | 0.799     | 0.748               | 0.737      | 0.688       | 0.652    | 0.669 |
| Baseline (always predicts the common answer) | 0.5         | 0.5       | 0.409               | 0.591      | 0           | 0        | 0     |

```text
ROC AUC of the best model: 0.805 (95% range when whole projects are resampled: 0.773 to 0.830)
Delays caught: 231 of 264 (88%) | false alarms: 191 of 382 (50%) | when it flags a task, it is right 55% of the time
The two highest risk bands hold 67% of all real delays in 40% of the tasks

Real delay rate by risk band:
```

|         | tasks   | real_delay_rate   | avg_predicted_chance   |
|:--------|:--------|:------------------|:-----------------------|
| Lowest  | 130     | 0.092             | 0.106                  |
| Low     | 129     | 0.256             | 0.244                  |
| Middle  | 129     | 0.318             | 0.404                  |
| High    | 129     | 0.589             | 0.605                  |
| Highest | 129     | 0.791             | 0.822                  |

```text
What a stricter alert cut-off would change (test set):
```

|      | tasks_flagged_pct   | delays_caught_pct   | right_when_flagged_pct   | false_alarm_pct   |
|:-----|:--------------------|:--------------------|:-------------------------|:------------------|
| 0.28 | 65.3                | 87.5                | 54.7                     | 50                |
| 0.35 | 55.4                | 80.7                | 59.5                     | 38                |
| 0.4  | 50.2                | 77.7                | 63.3                     | 31.2              |
| 0.45 | 45.4                | 73.1                | 65.9                     | 26.2              |
| 0.5  | 40.2                | 67.8                | 68.8                     | 21.2              |

```text
Confusion matrix:
```

|                  | Predicted on time   | Predicted delay   |
|:-----------------|:--------------------|:------------------|
| Actually on time | 191                 | 191               |
| Actually delayed | 33                  | 231               |

### Code: evaluate model, Objective 3: Improve site safety

```python
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

#### Output

![Objective 3: model evaluation](images/s22_01.png)

```text
Best model: Logistic Regression | alert cut-off 0.37 (chosen on training folds, not on the test set)
Test set: 152 incidents from 16 unseen projects, 25.0% serious

Scorecard on the test set:
```

|                                              | threshold   | roc_auc   | average_precision   | accuracy   | precision   | recall   | f1    |
|:---------------------------------------------|:------------|:----------|:--------------------|:-----------|:------------|:---------|:------|
| Logistic Regression                          | 0.37        | 0.857     | 0.696               | 0.803      | 0.591       | 0.684    | 0.634 |
| Random Forest                                | 0.5         | 0.837     | 0.601               | 0.776      | 0.625       | 0.263    | 0.37  |
| Gradient Boosting                            | 0.5         | 0.81      | 0.525               | 0.796      | 0.606       | 0.526    | 0.563 |
| Baseline (always predicts the common answer) | 0.5         | 0.5       | 0.25                | 0.75       | 0           | 0        | 0     |

```text
ROC AUC of the best model: 0.857 (95% range when whole projects are resampled: 0.798 to 0.918)
Serious incidents caught: 26 of 38 (68%) | false alarms: 18 of 114 (16%) | when it flags an incident, it is right 59% of the time
The two highest risk bands hold 82% of all serious incidents in 40% of the incidents

Serious incident rate by risk band:
```

|         | incidents   | real_serious_rate   | avg_predicted_chance   |
|:--------|:------------|:--------------------|:-----------------------|
| Lowest  | 31          | 0                   | 0.01                   |
| Low     | 30          | 0.067               | 0.048                  |
| Middle  | 30          | 0.167               | 0.141                  |
| High    | 30          | 0.4                 | 0.363                  |
| Highest | 31          | 0.613               | 0.67                   |

```text
What a different alert cut-off would change (test set):
```

|      | incidents_flagged_pct   | serious_caught_pct   | right_when_flagged_pct   | false_alarm_pct   |
|:-----|:------------------------|:---------------------|:-------------------------|:------------------|
| 0.3  | 33.6                    | 73.7                 | 54.9                     | 20.2              |
| 0.37 | 28.9                    | 68.4                 | 59.1                     | 15.8              |
| 0.4  | 27.6                    | 65.8                 | 59.5                     | 14.9              |
| 0.5  | 20.4                    | 50                   | 61.3                     | 10.5              |

```text
Confusion matrix:
```

|                      | Predicted not serious   | Predicted serious   |
|:---------------------|:------------------------|:--------------------|
| Actually not serious | 96                      | 18                  |
| Actually serious     | 12                      | 26                  |

### Code: evaluate model, Objective 4: Make material procurement more reliable

```python
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

#### Output

![Objective 4: model evaluation](images/s26_01.png)

```text
Best model: Logistic Regression | alert cut-off 0.46 (chosen on training folds, not on the test set)
Test set: 582 orders from 16 unseen projects, 45.9% late

Scorecard on the test set:
```

|                                              | threshold   | roc_auc   | average_precision   | accuracy   | precision   | recall   | f1    |
|:---------------------------------------------|:------------|:----------|:--------------------|:-----------|:------------|:---------|:------|
| Random Forest                                | 0.5         | 0.617     | 0.582               | 0.598      | 0.584       | 0.431    | 0.496 |
| Gradient Boosting                            | 0.5         | 0.608     | 0.567               | 0.576      | 0.543       | 0.476    | 0.507 |
| Logistic Regression                          | 0.46        | 0.602     | 0.568               | 0.576      | 0.535       | 0.573    | 0.553 |
| Baseline (always predicts the common answer) | 0.5         | 0.5       | 0.459               | 0.541      | 0           | 0        | 0     |

```text
ROC AUC of the best model: 0.602 (95% range when whole projects are resampled: 0.571 to 0.639)
Late orders caught: 153 of 267 (57%) | false alarms: 133 of 315 (42%) | when it flags an order, it is right 53% of the time
The two highest risk bands hold 48% of all late orders in 40% of the orders

Late order rate by risk band:
```

|         | orders   | real_late_rate   | avg_predicted_chance   |
|:--------|:---------|:-----------------|:-----------------------|
| Lowest  | 117      | 0.359            | 0.256                  |
| Low     | 116      | 0.422            | 0.381                  |
| Middle  | 116      | 0.405            | 0.455                  |
| High    | 116      | 0.526            | 0.53                   |
| Highest | 117      | 0.581            | 0.661                  |

```text
What a different alert cut-off would change (test set):
```

|      | orders_flagged_pct   | late_caught_pct   | right_when_flagged_pct   | false_alarm_pct   |
|:-----|:---------------------|:------------------|:-------------------------|:------------------|
| 0.4  | 64.1                 | 70                | 50.1                     | 59                |
| 0.46 | 49.1                 | 57.3              | 53.5                     | 42.2              |
| 0.5  | 36.6                 | 45.3              | 56.8                     | 29.2              |
| 0.55 | 25.4                 | 32.6              | 58.8                     | 19.4              |

```text
Confusion matrix:
```

|                  | Predicted on time   | Predicted late   |
|:-----------------|:--------------------|:-----------------|
| Actually on time | 182                 | 133              |
| Actually late    | 114                 | 153              |

### Code: evaluate model, Objective 5: Detect procurement price and invoice anomalies

```python
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

#### Output

![Objective 5: model evaluation](images/s30_01.png)

```text
Answer key opened from construction_raw/anomaly_answer_key.csv: 62 real anomalies in 3,000 orders (2.1%)
```

|                                    | value   |
|:-----------------------------------|:--------|
| orders                             | 3,000   |
| real anomalies                     | 62      |
| orders flagged                     | 150     |
| anomalies caught                   | 62      |
| anomalies missed                   | 0       |
| false alarms                       | 88      |
| recall (share of anomalies caught) | 1       |
| precision (flags that are real)    | 0.413   |
| F1                                 | 0.585   |
| ROC AUC of the score               | 1       |
| average precision of the score     | 1       |

```text
If we review exactly 62 orders (the true anomaly count), 100% of them are real anomalies.

Catch rate at different review budgets:
```

|     | orders_reviewed   | anomalies_caught   | recall_pct   | precision_pct   | random_review_would_catch   |
|:----|:------------------|:-------------------|:-------------|:----------------|:----------------------------|
| 1%  | 30                | 30                 | 48.4         | 100             | 0.6                         |
| 2%  | 60                | 60                 | 96.8         | 100             | 1.2                         |
| 3%  | 90                | 62                 | 100          | 68.9            | 1.9                         |
| 5%  | 150               | 62                 | 100          | 41.3            | 3.1                         |
| 10% | 300               | 62                 | 100          | 20.7            | 6.2                         |

```text
Isolation Forest against a one-line rule:
```

|                                               | orders flagged   | anomalies caught   | recall   | precision   |
|:----------------------------------------------|:-----------------|:-------------------|:---------|:------------|
| Isolation Forest (top 5%)                     | 150              | 62                 | 1        | 0.413       |
| Simple rule: price above 2x the material norm | 62               | 62                 | 1        | 1           |

```text
Price level of every order, real anomalies and flags:
```

|                | orders   | real_anomalies   | flagged   |
|:---------------|:---------|:-----------------|:----------|
| cheap (<0.87)  | 460      | 0                | 40        |
| normal         | 2028     | 0                | 24        |
| dear (1.15-2)  | 450      | 0                | 24        |
| very dear (>2) | 62       | 62               | 62        |

```text
Why the false alarms were flagged:
```

|                                   | false_alarms   |
|:----------------------------------|:---------------|
| priced below the material norm    | 39             |
| priced above the material norm    | 26             |
| unusual order quantity            | 19             |
| priced below similar orders       | 3              |
| unusual mix of price and quantity | 1              |

```text
Real anomalies the model missed: 0
```

---

## 18. Business Analysis

What does a model actually save? Compare reviewing the riskiest slice of items against reviewing at random.

| Objective | Review budget | Share of the loss found | Random review |
|-----------|:-------------:|:-----------------------:|:-------------:|
| 1. Overrun dollars | Top 40% | **58%** | 40% |
| 2. Delay days | Top 40% | **63%** | 40% |
| 3. Lost safety days | Top 40% | **83%** | 40% |
| 5. Money paid above the norm | Top 5% of orders | **88%** | 5% |

### Code: business analysis

```python
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

#### Output

![Business analysis: share of loss found by reviewing the riskiest slice](images/s31_01.png)

```text
Loss captured by reviewing the riskiest slice (test set for Objectives 1-4, all orders for Objective 5):
```

|                                          | loss measured                                       | total loss in test set   | items reviewed   | loss inside reviewed items   | share of loss captured   | capture vs random (times)   |
|:-----------------------------------------|:----------------------------------------------------|:-------------------------|:-----------------|:-----------------------------|:-------------------------|:----------------------------|
| Objective 1: cost overrun                | money spent above budget (USD)                      | 60,809,956.46            | 193              | 35,247,424.94                | 0.5796                   | 1.4491                      |
| Objective 2: schedule delay              | task days lost to delay (days)                      | 3,120                    | 258              | 1,977                        | 0.6337                   | 1.5841                      |
| Objective 3: serious safety incident     | worker days lost to incidents (days)                | 1,540                    | 61               | 1,271                        | 0.8253                   | 2.0633                      |
| Objective 4: late material delivery      | days materials arrived late (days)                  | 1,649                    | 233              | 822                          | 0.4985                   | 1.2462                      |
| Objective 5: price and invoice anomalies | money paid beyond 15% above the material norm (USD) | 6,095,147.3093           | 150              | 5,339,208.1098               | 0.876                    | 17.5195                     |

```text
Objective 1 check: dollars are concentrated in big budget lines, so what ranking finds the most overrun dollars?
```

|                      | share of overrun dollars found   |
|:---------------------|:---------------------------------|
| Random review        | 40                               |
| Model risk only      | 58                               |
| Largest budgets only | 75.6                             |
| Model risk x budget  | 74.5                             |

```text
The three worst segments for each objective (whole cleaned data):
```

|                                      | segment                          | share of items   | share of loss   | times the average   |
|:-------------------------------------|:---------------------------------|:-----------------|:----------------|:--------------------|
| Objective 1: cost overrun            | cost_category = Materials        | 0.1998           | 0.4048          | 2.0265              |
| Objective 1: cost overrun            | country = India                  | 0.0651           | 0.0991          | 1.5231              |
| Objective 1: cost overrun            | country = United States          | 0.0862           | 0.1243          | 1.4414              |
| Objective 2: schedule delay          | rainy_season = 1.0               | 0.1317           | 0.2418          | 1.8354              |
| Objective 2: schedule delay          | crew_band = Small                | 0.2816           | 0.3908          | 1.388               |
| Objective 2: schedule delay          | task_type = Structural Frame     | 0.1739           | 0.2383          | 1.3704              |
| Objective 3: serious safety incident | no_ppe_at_height = 1             | 0.0613           | 0.2152          | 3.5087              |
| Objective 3: serious safety incident | incident_type = Fall from Height | 0.1987           | 0.4517          | 2.2738              |
| Objective 3: serious safety incident | incident_type = Trench Collapse  | 0.0853           | 0.1813          | 2.1247              |
| Objective 4: late material delivery  | material = Glass Panels          | 0.2473           | 0.3032          | 1.2257              |
| Objective 4: late material delivery  | rainy_season = 1.0               | 0.1287           | 0.1571          | 1.221               |
| Objective 4: late material delivery  | country = Nigeria                | 0.1213           | 0.1369          | 1.1284              |

```text
What-if scenarios (upper bounds: they assume the gap is caused by the factor, which the data cannot prove):
```

|                                      | scenario                                                           | rate in group   | rate elsewhere   | events avoided   | loss avoided    | loss unit   | share of all events   |
|:-------------------------------------|:-------------------------------------------------------------------|:----------------|:-----------------|:-----------------|:----------------|:------------|:----------------------|
| Objective 1: cost overrun            | Cost Plus lines overrun as rarely as the rest                      | 0.657           | 0.4052           | 148.3646         | 22,692,902.0712 | USD         | 0.1318                |
| Objective 2: schedule delay          | Tasks with a crew of 18 or fewer are delayed as rarely as the rest | 0.601           | 0.3415           | 218.4192         | 2,278.7253      | days        | 0.165                 |
| Objective 3: serious safety incident | Work at height without PPE is as safe as the rest                  | 0.587           | 0.2287           | 16.4801          | 1,022.9878      | days        | 0.0877                |
| Objective 4: late material delivery  | International orders arrive late as rarely as local ones           | 0.4799          | 0.3092           | 436.7772         | 2,624.5755      | days        | 0.3262                |

```text
Objective 5: 150 orders flagged out of 3,000 hold $5,339,208 of the $6,095,147 paid beyond the 15% tolerance (88%).
Assumptions: the material norm is the fair price; a 40% review budget for Objectives 1-4 and 5% for Objective 5; losses are measured in the dollars or days recorded in the cleaned tables.
```

---

## 19. Save Model, Scaler and Features

Everything the app needs is saved with `joblib`: for each model, the model, the scaler, the encoder and a features file (column order, categories, alert cut-off). The price model also saves its price reference tables.


A round-trip test reloads every file and checks it reproduces the same predictions.

### Code: save model, scaler and features

```python
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

#### Output

|                                          | model               | rows re-scored   | features identical   | max prediction difference   | unseen labels handled   |
|:-----------------------------------------|:--------------------|:-----------------|:---------------------|:----------------------------|:------------------------|
| Objective 1: cost overrun                | Logistic Regression | 482              | True                 | 0                           | True                    |
| Objective 2: schedule delay              | Logistic Regression | 646              | True                 | 0                           | True                    |
| Objective 3: serious safety incident     | Logistic Regression | 152              | True                 | 0                           | True                    |
| Objective 4: late material delivery      | Logistic Regression | 582              | True                 | 0                           | True                    |
| Objective 5: price and invoice anomalies | Isolation Forest    | 3000             | True                 | 0.0011                      | True                    |

```text
Saved 21 files to construction_models/:
```

|                                        | size_kb   |
|:---------------------------------------|:----------|
| cost_overrun_encoder.joblib            | 1.8       |
| cost_overrun_features.joblib           | 1.6       |
| cost_overrun_model.joblib              | 2.1       |
| cost_overrun_scaler.joblib             | 6.5       |
| late_delivery_encoder.joblib           | 1.8       |
| late_delivery_features.joblib          | 1.6       |
| late_delivery_model.joblib             | 2         |
| late_delivery_scaler.joblib            | 6         |
| model_card.joblib                      | 0.9       |
| price_anomalies_features.joblib        | 0.7       |
| price_anomalies_model.joblib           | 5,185.3   |
| price_anomalies_price_reference.joblib | 0.9       |
| price_anomalies_scaler.joblib          | 3         |
| schedule_delay_encoder.joblib          | 1.9       |
| schedule_delay_features.joblib         | 1.8       |
| schedule_delay_model.joblib            | 2.2       |
| schedule_delay_scaler.joblib           | 6.9       |
| site_safety_encoder.joblib             | 2.1       |
| site_safety_features.joblib            | 1.8       |
| site_safety_model.joblib               | 2.2       |
| site_safety_scaler.joblib              | 6.6       |

```text
Objectives 1-4: the saved pipeline (raw row -> encode -> scale -> model) reproduces the notebook's predictions exactly.
Objective 5: scores match to about 0.001, because training used price ratios rounded to 4 decimals. The same orders are flagged.
Download them in the next step.
```

---

## 20. Download Model, Scaler and Features

All 21 saved files are zipped into one `construction_models.zip` and copied to the Downloads folder. The step re-opens the zip, confirms it is not corrupt, and loads every file again to prove nothing was damaged.

### Code: download model, scaler and features

```python
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

#### Output

```text
construction_models.zip
```

|                 | model              | scaler            | features   | encoder             |
|:----------------|:-------------------|:------------------|:-----------|:--------------------|
| cost_overrun    | LogisticRegression | ColumnTransformer | 34         | OneHotEncoder       |
| schedule_delay  | LogisticRegression | ColumnTransformer | 38         | OneHotEncoder       |
| site_safety     | LogisticRegression | ColumnTransformer | 36         | OneHotEncoder       |
| late_delivery   | LogisticRegression | ColumnTransformer | 32         | OneHotEncoder       |
| price_anomalies | IsolationForest    | ColumnTransformer | 4          | none (numbers only) |

```text
construction_models.zip: 21 files, 1.3 MB, all files load again after unzipping
Zip saved here:        construction_models.zip
Copy in Downloads:     <your Downloads folder>/construction_models.zip

To use the models later, unzip and call joblib.load('construction_models/<objective>_model.joblib'), the same for _scaler and _features.
```

---

## 21. Key Findings and Insights

**Cost**
- **Contract type is the strongest driver.** Cost Plus lines overran 66% of the time, Fixed Price 36%.
- **Materials are the hotspot.** 69% of Materials lines overran, against 21% for Overheads. Nigeria ran at 69% against 27% in the United States.
- **Dollars follow budget size.** The largest 40% of budgets hold 76% of overrun dollars.

**Schedule**
- **Small crews slip.** Crews of 18 or fewer were delayed 60% of the time, the rest 34%.
- **The rainy season roughly doubles delay risk.** 75% against 36% in the dry season.

**Safety**
- **Height work without PPE is the most dangerous condition.** 59% of those incidents were serious, against 23% otherwise.
- **Falls are the serious ones.** 52% of falls from height were serious, against 4% of slips and trips.
- **Experience and shift matter.** Serious rate was 41% for the least experienced workers against 9% for the most, and 38% on night shifts against 21% on days.

**Procurement**
- **International orders are late more often.** 48% against 29% for local orders.
- **Late delivery is hard to predict** from the data we hold (ROC AUC 0.60).
- **A small set of orders carries most of the overpricing.** The top 5% by anomaly score hold 88% of the money paid above the norm.

### Code: key findings and insights

```python
from pathlib import Path
from IPython.display import Markdown, display as show
from sklearn.metrics import roc_auc_score

OUT_DIR = Path("construction_business")
OUT_DIR.mkdir(exist_ok=True)

def rate(frame, target, mask):
    """Share of rows with the event, inside the mask (in %)."""
    return 100 * frame.loc[mask, target].mean()

def source_of(key, target=None):
    cfg = OBJECTIVES[key]
    frame = cfg["source"].drop_duplicates(subset=cfg["id_col"])
    return frame.dropna(subset=[target]) if target else frame

def test_auc(key, pack):
    X_test, y_test = scaled[key]["X_test"], splits[key]["y_test"]
    return roc_auc_score(y_test, pack["trained"][pack["best_name"]].predict_proba(X_test)[:, 1])

auc = {"cost_overrun": test_auc("cost_overrun", models_obj1), "schedule_delay": test_auc("schedule_delay", models_obj2),
       "site_safety": test_auc("site_safety", models_obj3), "late_delivery": test_auc("late_delivery", models_obj4)}

# ---------- the numbers behind each finding, all computed from the cleaned data and the saved results ----------
ledger = source_of("cost_overrun", "overrun_flag")
tasks = source_of("schedule_delay", "delay_flag")
incidents = source_of("site_safety", "serious_flag")
orders = source_of("late_delivery", "late_delivery")
crew_cut = tasks["crew_size"].quantile(0.25)
night = incidents["shift"].astype("string").eq("Night")
inexperienced = incidents["worker_experience_years"] <= incidents["worker_experience_years"].quantile(0.25)
experienced = incidents["worker_experience_years"] >= incidents["worker_experience_years"].quantile(0.75)
cost_cat = ledger["cost_category"].astype("string")
country_l = ledger["country"].astype("string")
contract = ledger["contract_type"].astype("string")
fall = incidents["incident_type"].astype("string").eq("Fall from Height")
slip = incidents["incident_type"].astype("string").eq("Slip or Trip")
anomalies_total = int(scored["is_anomaly"].sum())
rule_row = rule_check.loc["Simple rule: price above 2x the material norm"]
model_row = rule_check.loc["Isolation Forest (top 5%)"]
sum_ = summary["share of loss captured"]
which = {name: i for i, name in enumerate(summary.index)}
captured = {k: sum_.iloc[i] for k, i in zip(["o1", "o2", "o3", "o4", "o5"], range(5))}
scen = scenarios["loss avoided"]

findings = pd.DataFrame([
    # ---- portfolio ----
    ("Portfolio", "Problems are common, not rare",
     f"{rate(ledger, 'overrun_flag', slice(None)):.0f}% of cost lines overran by more than 15%, {rate(tasks, 'delay_flag', slice(None)):.0f}% of tasks finished late, "
     f"{rate(incidents, 'serious_flag', slice(None)):.0f}% of incidents were serious and {rate(orders, 'late_delivery', slice(None)):.0f}% of material orders arrived late.",
     "Control has to be built into routine work, because these are not exceptions."),
    # ---- objective 1 ----
    ("1. Cost overruns", "Contract type is the strongest overrun driver",
     f"Cost Plus lines overran {rate(ledger, 'overrun_flag', contract.eq('Cost Plus')):.0f}% of the time, Fixed Price {rate(ledger, 'overrun_flag', contract.eq('Fixed Price')):.0f}%.",
     "Cost Plus gives the contractor little reason to hold costs down."),
    ("1. Cost overruns", "Materials and one country stand out",
     f"Materials lines overran {rate(ledger, 'overrun_flag', cost_cat.eq('Materials')):.0f}% of the time against {rate(ledger, 'overrun_flag', cost_cat.eq('Overheads')):.0f}% for Overheads. "
     f"Nigeria ran at {rate(ledger, 'overrun_flag', country_l.eq('Nigeria')):.0f}% against {rate(ledger, 'overrun_flag', country_l.eq('United States')):.0f}% in the United States.",
     "Material price control and country-specific cost planning matter most."),
    ("1. Cost overruns", "Dollars follow budget size, not just risk",
     f"Reviewing the largest 40% of budgets finds {100 * objective_1_ranking.loc['Largest budgets only'].iloc[0]:.0f}% of overrun dollars. The risk score alone finds {100 * objective_1_ranking.loc['Model risk only'].iloc[0]:.0f}%.",
     "Use the model to flag risky small lines, and watch the big budget lines regardless."),
    # ---- objective 2 ----
    ("2. Schedule delay", "Understaffed tasks slip far more often",
     f"Tasks with a crew of {crew_cut:.0f} or fewer were delayed {rate(tasks, 'delay_flag', tasks['crew_size'] <= crew_cut):.0f}% of the time, the rest {rate(tasks, 'delay_flag', tasks['crew_size'] > crew_cut):.0f}%.",
     "Crew size is the one delay driver that managers control directly."),
    ("2. Schedule delay", "The rainy season roughly doubles the delay risk",
     f"Rainy-season tasks were delayed {rate(tasks, 'delay_flag', tasks['rainy_season'] == 1):.0f}% of the time, dry-season tasks {rate(tasks, 'delay_flag', tasks['rainy_season'] == 0):.0f}%. "
     f"(The rainy-season calendar is a domain assumption.)",
     "Plan buffers and crew levels around the rainy months."),
    # ---- objective 3 ----
    ("3. Site safety", "Height work without PPE is the most dangerous condition",
     f"{rate(incidents, 'serious_flag', incidents['no_ppe_at_height'] == 1):.0f}% of incidents involving height work without PPE were serious, against {rate(incidents, 'serious_flag', incidents['no_ppe_at_height'] == 0):.0f}% otherwise.",
     "Enforce PPE at height first. It is cheap and the effect is large."),
    ("3. Site safety", "Falls are the serious ones; experience protects",
     f"{rate(incidents, 'serious_flag', fall):.0f}% of falls from height were serious, against {rate(incidents, 'serious_flag', slip):.0f}% of slips and trips. "
     f"Least-experienced workers: {rate(incidents, 'serious_flag', inexperienced):.0f}% serious, most-experienced: {rate(incidents, 'serious_flag', experienced):.0f}%. "
     f"Night shifts: {rate(incidents, 'serious_flag', night):.0f}% against {rate(incidents, 'serious_flag', ~night):.0f}% on days.",
     "Pair new workers with experienced ones and add supervision on night shifts and at height."),
    # ---- objective 4 ----
    ("4. Late delivery", "Late delivery is hard to predict from the data we hold",
     f"The model reaches a ROC AUC of only {auc['late_delivery']:.2f} (0.50 is a coin flip). The riskiest 40% of orders hold {100 * captured['o4']:.0f}% of late days.",
     "Treat the model as a rough guide, not a control."),
    ("4. Late delivery", "International orders arrive late more often",
     f"International orders were late {rate(orders, 'late_delivery', orders['international_supplier'] == 1):.0f}% of the time, local orders {rate(orders, 'late_delivery', orders['international_supplier'] == 0):.0f}%.",
     "Add buffer time, or use local suppliers for critical materials."),
    # ---- objective 5 ----
    ("5. Price anomalies", "A small set of orders carries most of the overpricing",
     f"The 5% of orders the Isolation Forest flags hold {100 * captured['o5']:.0f}% of the money paid beyond 15% above the material norm.",
     "A 150-order review list covers nearly all of the pricing risk."),
    ("5. Price anomalies", "The model finds every planted anomaly, but so does a simple rule",
     f"The model caught {int(model_row['anomalies caught'])} of {anomalies_total} real anomalies ({int(model_row['orders flagged'])} orders flagged). "
     f"A one-line rule (price above 2x the material norm) also caught {int(rule_row['anomalies caught'])} with {int(rule_row['orders flagged'])} flags.",
     "Start with the rule. Keep the model for subtler cases that real invoices may contain."),
    # ---- models ----
    ("Models", "Simple models were enough",
     f"Logistic Regression was the best model for all four classification objectives. Test ROC AUC: overrun {auc['cost_overrun']:.2f}, delay {auc['schedule_delay']:.2f}, "
     f"safety {auc['site_safety']:.2f}, late delivery {auc['late_delivery']:.2f}.",
     "The models are explainable, which helps with trust and adoption on site."),
], columns=["area", "finding", "evidence", "why it matters"])

findings.to_csv(OUT_DIR / "key_findings.csv", index=False)
markdown = "# Key findings and insights\n\n"
for area, group in findings.groupby("area", sort=False):
    markdown += f"## {area}\n"
    for _, row in group.iterrows():
        markdown += f"- **{row['finding']}.** {row['evidence']} *{row['why it matters']}*\n"
    markdown += "\n"
(OUT_DIR / "key_findings.md").write_text(markdown, encoding="utf-8")

# ---------- Verification ----------
show(Markdown(markdown))
print(f"{len(findings)} findings saved to {OUT_DIR}/key_findings.csv and key_findings.md")
```

#### Output

#### Key findings and insights

#### Portfolio
- **Problems are common, not rare.** 47% of cost lines overran by more than 15%, 41% of tasks finished late, 25% of incidents were serious and 46% of material orders arrived late. *Control has to be built into routine work, because these are not exceptions.*

#### 1. Cost overruns
- **Contract type is the strongest overrun driver.** Cost Plus lines overran 66% of the time, Fixed Price 36%. *Cost Plus gives the contractor little reason to hold costs down.*
- **Materials and one country stand out.** Materials lines overran 69% of the time against 21% for Overheads. Nigeria ran at 69% against 27% in the United States. *Material price control and country-specific cost planning matter most.*
- **Dollars follow budget size, not just risk.** Reviewing the largest 40% of budgets finds 76% of overrun dollars. The risk score alone finds 58%. *Use the model to flag risky small lines, and watch the big budget lines regardless.*

#### 2. Schedule delay
- **Understaffed tasks slip far more often.** Tasks with a crew of 18 or fewer were delayed 60% of the time, the rest 34%. *Crew size is the one delay driver that managers control directly.*
- **The rainy season roughly doubles the delay risk.** Rainy-season tasks were delayed 75% of the time, dry-season tasks 36%. (The rainy-season calendar is a domain assumption.) *Plan buffers and crew levels around the rainy months.*

#### 3. Site safety
- **Height work without PPE is the most dangerous condition.** 59% of incidents involving height work without PPE were serious, against 23% otherwise. *Enforce PPE at height first. It is cheap and the effect is large.*
- **Falls are the serious ones; experience protects.** 52% of falls from height were serious, against 4% of slips and trips. Least-experienced workers: 41% serious, most-experienced: 9%. Night shifts: 38% against 21% on days. *Pair new workers with experienced ones and add supervision on night shifts and at height.*

#### 4. Late delivery
- **Late delivery is hard to predict from the data we hold.** The model reaches a ROC AUC of only 0.60 (0.50 is a coin flip). The riskiest 40% of orders hold 50% of late days. *Treat the model as a rough guide, not a control.*
- **International orders arrive late more often.** International orders were late 48% of the time, local orders 29%. *Add buffer time, or use local suppliers for critical materials.*

#### 5. Price anomalies
- **A small set of orders carries most of the overpricing.** The 5% of orders the Isolation Forest flags hold 88% of the money paid beyond 15% above the material norm. *A 150-order review list covers nearly all of the pricing risk.*
- **The model finds every planted anomaly, but so does a simple rule.** The model caught 62 of 62 real anomalies (150 orders flagged). A one-line rule (price above 2x the material norm) also caught 62 with 62 flags. *Start with the rule. Keep the model for subtler cases that real invoices may contain.*

#### Models
- **Simple models were enough.** Logistic Regression was the best model for all four classification objectives. Test ROC AUC: overrun 0.83, delay 0.80, safety 0.86, late delivery 0.60. *The models are explainable, which helps with trust and adoption on site.*

```text
13 findings saved to construction_business/key_findings.csv and key_findings.md
```

---

## 22. Recommendations

Ordered by priority: high impact, low effort first. Impact figures are upper bounds, not promises.

| # | Recommendation | Effort | Why |
|---|----------------|:------:|-----|
| 1 | **Enforce PPE on every job at height** | Low | 3.5x the average lost days per incident |
| 2 | **Set a minimum crew size for critical tasks** | Medium | Crews of 18 or fewer are the high-risk group |
| 3 | **Check invoices against the material norm before payment** | Low | A 2x-norm rule caught all 62 known anomalies |
| 4 | **Add cost-control incentives to Cost Plus contracts** | High | 66% overrun rate against 36% |
| 5 | **Weekly overrun review on the largest budget lines**, topped up by the model | Low | Largest 40% of budgets hold 76% of overrun dollars |
| 6 | **Add rainy-season buffers and crew cover** | Low | Delay rate 75% against 36% |
| 7 | **Control materials prices**, mainly in Nigeria and India | Medium | Materials hold 40% of overrun dollars in 20% of lines |
| 8 | **Buffer or localise material deliveries** | Medium | International orders late 48% against 29% |
| 9 | **Add night-shift supervision and pair new workers with experienced ones** | Medium | 38% serious on nights, 41% for the least experienced |
| 10 | **Use the models as ranking tools and retrain every quarter** | Low | Keeps attention on the right items without false confidence |
| 11 | **Capture better delivery and invoice data at the source** | Medium | Would lift the weakest model most |

### Code: recommendations

```python
from pathlib import Path
from IPython.display import Markdown, display as show

OUT_DIR = Path("construction_business")
OUT_DIR.mkdir(exist_ok=True)

# ---------- numbers behind each recommendation (live from the earlier steps) ----------
what_if = scenarios["loss avoided"]                          # upper bounds, one row per objective 1-4
saved_cost_plus = what_if.iloc[0]                            # USD
saved_crew_days = what_if.iloc[1]                            # task days
saved_ppe_days = what_if.iloc[2]                             # worker days
saved_intl_days = what_if.iloc[3]                            # delay days
share = summary["share of loss captured"]
o5_share = share.iloc[4]
o5_money = summary["loss inside reviewed items"].iloc[4]
o5_total = summary["total loss in test set"].iloc[4]
largest_budget_share = objective_1_ranking.loc["Largest budgets only"].iloc[0]
risk_budget_share = objective_1_ranking.loc["Model risk x budget"].iloc[0]
anomalies_total = int(scored["is_anomaly"].sum())
rule_orders = int(rule_check.loc["Simple rule: price above 2x the material norm", "orders flagged"])
rule_caught = int(rule_check.loc["Simple rule: price above 2x the material norm", "anomalies caught"])
flagged_orders = int(rule_check.loc["Isolation Forest (top 5%)", "orders flagged"])

def millions(value):
    return f"${value / 1e6:,.1f}M"

recommendations = pd.DataFrame([
    (1, "Enforce PPE on every job at height",
     "Height work without PPE is the most dangerous condition (3.5 times the average lost days per incident).",
     f"Up to {saved_ppe_days:,.0f} worker days saved if these incidents were as safe as the rest.",
     "Low", "Site safety manager",
     "Share of height tasks with PPE confirmed; serious incidents per 100 height tasks",
     "Make PPE sign-off a start condition for any height task, and spot-check weekly."),
    (2, "Set a minimum crew size for critical tasks",
     "Tasks with the smallest crews were delayed about 60% of the time, against 34% for the rest.",
     f"Up to {saved_crew_days:,.0f} task days saved if small-crew tasks slipped as rarely as the rest.",
     "Medium", "Project managers",
     "Share of tasks below minimum crew; delayed-task rate",
     "Set the minimum from the data (crews of 18 or fewer are the high-risk group) and review the exceptions."),
    (3, "Check invoices against the material norm before payment",
     f"A one-line rule (price above 2x the material norm) caught {rule_caught} of the {anomalies_total} known anomalies with only {rule_orders} flags.",
     f"The weekly top-5% review holds {100 * o5_share:.0f}% ({millions(o5_money)} of {millions(o5_total)}) of money paid beyond 15% above the norm.",
     "Low", "Procurement and accounts payable",
     "Money recovered or blocked; share of flagged orders confirmed as errors",
     f"Start with the rule, then review the Isolation Forest's {flagged_orders} highest-scoring orders each cycle."),
    (4, "Add cost-control incentives to Cost Plus contracts",
     "Cost Plus lines overran 66% of the time against 36% for Fixed Price.",
     f"Up to {millions(saved_cost_plus)} of overrun if these lines overran as rarely as the rest (an upper bound).",
     "High", "Commercial and contracts team",
     "Overrun rate and overrun dollars by contract type",
     "Add caps, shared-saving clauses or milestone cost checks to Cost Plus contracts, starting with the biggest ones."),
    (5, "Run a weekly overrun review on the largest budget lines, topped up by the model",
     f"Overrun dollars sit in big lines: the largest 40% of budgets hold {100 * largest_budget_share:.0f}% of them. "
     f"Ranking by risk times budget finds {100 * risk_budget_share:.0f}%.",
     "The model adds value on smaller lines that the budget-only view would skip.",
     "Low", "Cost controllers",
     "Overrun dollars found early; lines reviewed per week",
     "Review all large lines plus any line the cost-overrun model scores above its 0.36 alert cut-off."),
    (6, "Add rainy-season buffers and crew cover to schedules",
     "Rainy-season tasks were delayed about 75% of the time against 36% in the dry season, and late material orders also rise.",
     "A large share of delay risk is predictable from the calendar.",
     "Low", "Planners",
     "Delay rate in rainy against dry months",
     "Plan outdoor and structural tasks for the dry season where possible, and add buffer days when they cannot move."),
    (7, "Control Materials prices, mainly in Nigeria and India",
     "Materials lines overran 69% of the time against 21% for Overheads, and Nigeria and India run well above the average.",
     "Materials hold 40% of all overrun dollars in 20% of cost lines.",
     "Medium", "Procurement",
     "Materials overrun rate by country; price ratio to the material norm",
     "Use framework agreements or price caps for key materials, and re-quote when prices move."),
    (8, "Buffer or localise material deliveries",
     "International orders were late 48% of the time against 29% for local orders.",
     f"Up to {saved_intl_days:,.0f} delay days if international orders arrived as reliably as local ones. Delivery risk is only weakly predictable.",
     "Medium", "Supply chain",
     "Late-delivery rate by supplier origin; days late per order",
     "Add lead-time buffer for international orders, and prefer local or pre-qualified suppliers for critical materials."),
    (9, "Add supervision on night shifts and pair new workers with experienced ones",
     "Serious incidents were 38% on night shifts against 21% on days, and 41% for the least-experienced workers against 9% for the most.",
     "Targets the people and shifts where serious injuries concentrate.",
     "Medium", "Site safety manager and HR",
     "Serious incident rate by shift and by experience band",
     "Assign a supervisor to every night shift and a buddy to every worker in the least-experienced quarter of the workforce."),
    (10, "Use the models as ranking tools and review them every quarter",
     "The models rank risk well for overruns, delays and safety, and poorly for late delivery (ROC AUC 0.60).",
     "Keeps staff attention on the right items without false confidence.",
     "Low", "Data team",
     "ROC AUC on new data; share of flagged items confirmed",
     "Retrain quarterly on new data. Do not use the late-delivery model for decisions until better delivery data is collected."),
    (11, "Capture better delivery and invoice data at the source",
     "Cleaning had to repair mixed date formats, currency symbols, unit spellings and orphan project codes.",
     "Cleaner data improves every model, and it would lift the weak late-delivery model most.",
     "Medium", "Data and IT teams",
     "Share of records passing validation at entry",
     "Standardise dates, currency and units in the entry forms, and add carrier, port and lead-time fields to orders."),
], columns=["priority", "recommendation", "evidence", "expected impact", "effort", "owner", "how to measure", "first step"])

recommendations.to_csv(OUT_DIR / "recommendations.csv", index=False)
markdown = "# Recommendations\n\nOrdered by priority: high impact and low effort first. Impact figures are upper bounds, not promises.\n\n"
for _, row in recommendations.iterrows():
    markdown += (f"## {row['priority']}. {row['recommendation']}\n"
                 f"- **Why:** {row['evidence']}\n- **Expected impact:** {row['expected impact']}\n"
                 f"- **Effort:** {row['effort']} | **Owner:** {row['owner']}\n"
                 f"- **How to measure:** {row['how to measure']}\n- **First step:** {row['first step']}\n\n")
markdown += ("## Limits to keep in mind\n- Impact figures come from comparing groups in historical data, which shows a link and not proof of cause.\n"
             "- The rainy-season calendar and the 15% price tolerance are assumptions.\n"
             "- The Objective 5 results are measured against the known anomalies in the dataset, which are easier to find than subtle real ones.\n")
(OUT_DIR / "recommendations.md").write_text(markdown, encoding="utf-8")

# ---------- Verification ----------
show(Markdown(markdown))
print(f"{len(recommendations)} recommendations saved to {OUT_DIR}/recommendations.csv and recommendations.md")
```

#### Output

#### Recommendations

Ordered by priority: high impact and low effort first. Impact figures are upper bounds, not promises.

#### 1. Enforce PPE on every job at height
- **Why:** Height work without PPE is the most dangerous condition (3.5 times the average lost days per incident).
- **Expected impact:** Up to 1,023 worker days saved if these incidents were as safe as the rest.
- **Effort:** Low | **Owner:** Site safety manager
- **How to measure:** Share of height tasks with PPE confirmed; serious incidents per 100 height tasks
- **First step:** Make PPE sign-off a start condition for any height task, and spot-check weekly.

#### 2. Set a minimum crew size for critical tasks
- **Why:** Tasks with the smallest crews were delayed about 60% of the time, against 34% for the rest.
- **Expected impact:** Up to 2,279 task days saved if small-crew tasks slipped as rarely as the rest.
- **Effort:** Medium | **Owner:** Project managers
- **How to measure:** Share of tasks below minimum crew; delayed-task rate
- **First step:** Set the minimum from the data (crews of 18 or fewer are the high-risk group) and review the exceptions.

#### 3. Check invoices against the material norm before payment
- **Why:** A one-line rule (price above 2x the material norm) caught 62 of the 62 known anomalies with only 62 flags.
- **Expected impact:** The weekly top-5% review holds 88% ($5.3M of $6.1M) of money paid beyond 15% above the norm.
- **Effort:** Low | **Owner:** Procurement and accounts payable
- **How to measure:** Money recovered or blocked; share of flagged orders confirmed as errors
- **First step:** Start with the rule, then review the Isolation Forest's 150 highest-scoring orders each cycle.

#### 4. Add cost-control incentives to Cost Plus contracts
- **Why:** Cost Plus lines overran 66% of the time against 36% for Fixed Price.
- **Expected impact:** Up to $22.7M of overrun if these lines overran as rarely as the rest (an upper bound).
- **Effort:** High | **Owner:** Commercial and contracts team
- **How to measure:** Overrun rate and overrun dollars by contract type
- **First step:** Add caps, shared-saving clauses or milestone cost checks to Cost Plus contracts, starting with the biggest ones.

#### 5. Run a weekly overrun review on the largest budget lines, topped up by the model
- **Why:** Overrun dollars sit in big lines: the largest 40% of budgets hold 76% of them. Ranking by risk times budget finds 75%.
- **Expected impact:** The model adds value on smaller lines that the budget-only view would skip.
- **Effort:** Low | **Owner:** Cost controllers
- **How to measure:** Overrun dollars found early; lines reviewed per week
- **First step:** Review all large lines plus any line the cost-overrun model scores above its 0.36 alert cut-off.

#### 6. Add rainy-season buffers and crew cover to schedules
- **Why:** Rainy-season tasks were delayed about 75% of the time against 36% in the dry season, and late material orders also rise.
- **Expected impact:** A large share of delay risk is predictable from the calendar.
- **Effort:** Low | **Owner:** Planners
- **How to measure:** Delay rate in rainy against dry months
- **First step:** Plan outdoor and structural tasks for the dry season where possible, and add buffer days when they cannot move.

#### 7. Control Materials prices, mainly in Nigeria and India
- **Why:** Materials lines overran 69% of the time against 21% for Overheads, and Nigeria and India run well above the average.
- **Expected impact:** Materials hold 40% of all overrun dollars in 20% of cost lines.
- **Effort:** Medium | **Owner:** Procurement
- **How to measure:** Materials overrun rate by country; price ratio to the material norm
- **First step:** Use framework agreements or price caps for key materials, and re-quote when prices move.

#### 8. Buffer or localise material deliveries
- **Why:** International orders were late 48% of the time against 29% for local orders.
- **Expected impact:** Up to 2,625 delay days if international orders arrived as reliably as local ones. Delivery risk is only weakly predictable.
- **Effort:** Medium | **Owner:** Supply chain
- **How to measure:** Late-delivery rate by supplier origin; days late per order
- **First step:** Add lead-time buffer for international orders, and prefer local or pre-qualified suppliers for critical materials.

#### 9. Add supervision on night shifts and pair new workers with experienced ones
- **Why:** Serious incidents were 38% on night shifts against 21% on days, and 41% for the least-experienced workers against 9% for the most.
- **Expected impact:** Targets the people and shifts where serious injuries concentrate.
- **Effort:** Medium | **Owner:** Site safety manager and HR
- **How to measure:** Serious incident rate by shift and by experience band
- **First step:** Assign a supervisor to every night shift and a buddy to every worker in the least-experienced quarter of the workforce.

#### 10. Use the models as ranking tools and review them every quarter
- **Why:** The models rank risk well for overruns, delays and safety, and poorly for late delivery (ROC AUC 0.60).
- **Expected impact:** Keeps staff attention on the right items without false confidence.
- **Effort:** Low | **Owner:** Data team
- **How to measure:** ROC AUC on new data; share of flagged items confirmed
- **First step:** Retrain quarterly on new data. Do not use the late-delivery model for decisions until better delivery data is collected.

#### 11. Capture better delivery and invoice data at the source
- **Why:** Cleaning had to repair mixed date formats, currency symbols, unit spellings and orphan project codes.
- **Expected impact:** Cleaner data improves every model, and it would lift the weak late-delivery model most.
- **Effort:** Medium | **Owner:** Data and IT teams
- **How to measure:** Share of records passing validation at entry
- **First step:** Standardise dates, currency and units in the entry forms, and add carrier, port and lead-time fields to orders.

#### Limits to keep in mind
- Impact figures come from comparing groups in historical data, which shows a link and not proof of cause.
- The rainy-season calendar and the 15% price tolerance are assumptions.
- The Objective 5 results are measured against the known anomalies in the dataset, which are easier to find than subtle real ones.

```text
11 recommendations saved to construction_business/recommendations.csv and recommendations.md
```

---

## 23. Conclusion

Cost, schedule and safety risk can be ranked well enough to point managers at the slice of work that matters most, and a short review list catches most of the pricing risk. Late delivery is the exception: better delivery data (carrier, port, supplier lead times) is the way to improve it.

**Limits to keep in mind**
- Test sets are small (16 unseen projects), so exact scores can move.
- Group differences show a link, not proof of cause. Impact figures are upper bounds.
- The rainy-season calendar and the 15% price tolerance are assumptions.
- Price anomalies are measured against planted anomalies, which are easier to find than subtle real ones.

**Next steps:** retrain quarterly, collect better delivery data, and track the recommendation measures to see whether the actions work.

### Code: conclusion

```python
from pathlib import Path
from IPython.display import Markdown, display as show
from sklearn.metrics import roc_auc_score

OUT_DIR = Path("construction_business")
OUT_DIR.mkdir(exist_ok=True)

def test_auc(key, pack):
    X_test, y_test = scaled[key]["X_test"], splits[key]["y_test"]
    return roc_auc_score(y_test, pack["trained"][pack["best_name"]].predict_proba(X_test)[:, 1])

packs = {"cost_overrun": models_obj1, "schedule_delay": models_obj2, "site_safety": models_obj3, "late_delivery": models_obj4}
auc = {key: test_auc(key, pack) for key, pack in packs.items()}
captured = summary["share of loss captured"]
rows_used = {key: len(abts[key]) for key in OBJECTIVES}
model_row = rule_check.loc["Isolation Forest (top 5%)"]
n_anomalies = int(scored["is_anomaly"].sum())

scorecard = pd.DataFrame([
    ("1. Cost overruns", "Logistic Regression", f"ROC AUC {auc['cost_overrun']:.2f}", f"{100 * captured.iloc[0]:.0f}% of overrun dollars in the riskiest 40% of lines", "Good ranking, dollars follow budget size"),
    ("2. Schedule delay", "Logistic Regression", f"ROC AUC {auc['schedule_delay']:.2f}", f"{100 * captured.iloc[1]:.0f}% of delay days in the riskiest 40% of tasks", "Good ranking"),
    ("3. Site safety", "Logistic Regression", f"ROC AUC {auc['site_safety']:.2f}", f"{100 * captured.iloc[2]:.0f}% of lost days in the riskiest 40% of incidents", "Strongest model"),
    ("4. Late delivery", "Logistic Regression", f"ROC AUC {auc['late_delivery']:.2f}", f"{100 * captured.iloc[3]:.0f}% of late days in the riskiest 40% of orders", "Weak, use only as a rough guide"),
    ("5. Price anomalies", "Isolation Forest", f"{int(model_row['anomalies caught'])} of {n_anomalies} known anomalies found", f"{100 * captured.iloc[4]:.0f}% of overpaid dollars in 5% of orders", "Excellent here, but a simple rule matches it"),
], columns=["objective", "model", "test result", "business value", "verdict"]).set_index("objective")

markdown = f"""# Conclusion

## What we set out to do
Build machine learning models on six construction tables (projects, vendors, cost ledger, tasks, material orders and safety incidents) that help managers control cost, schedule, safety and procurement risk.

## What we did
We audited and cleaned the data, engineered features, built a star schema, and prepared one analytical base table per business objective. We kept whole projects together in the train and test split so that results reflect new projects, not memorised ones. We compared four models for each classification objective and used an Isolation Forest for price anomalies. The saved models, scalers and feature lists are in `construction_models/`.

## What we found
- Cost overruns are driven by contract type, cost category and country, not by project size. Cost Plus and Materials lines are the main hotspots.
- Delays are a staffing and weather problem. Small crews and the rainy season account for most of the risk.
- Serious safety incidents concentrate in height work without PPE, falls, inexperienced workers and night shifts.
- Late deliveries are hard to predict from the data we have. International orders are late more often, and that is the main lever.
- A small set of orders carries most of the overpricing, and the Isolation Forest finds all {n_anomalies} known anomalies in a review list of {int(model_row['orders flagged'])} orders.

## What the models are worth
Reviewing only the riskiest slice finds {100 * captured.iloc[0]:.0f}% of overrun dollars, {100 * captured.iloc[1]:.0f}% of delay days and {100 * captured.iloc[2]:.0f}% of lost safety days, against 40% for a random review. The price-anomaly list finds {100 * captured.iloc[4]:.0f}% of overpaid dollars in 5% of orders.

## Limits
- Test sets are small, with 16 unseen projects each, so exact scores can move. The rankings are more reliable than single numbers.
- Group differences show a link, not proof of cause. The what-if figures are upper bounds.
- The rainy-season calendar and the 15% price tolerance are assumptions.
- Price anomalies are measured against known anomalies, which are easier to find than subtle real ones, and a simple price rule performs just as well here.
- The late-delivery model is weak, and it should be used only as a rough guide.

## Next steps
Put the models behind a simple app so managers can score new cost lines, tasks, incidents and orders. Retrain every quarter. Collect better delivery data (carrier, port, supplier lead times) to improve the weakest model. Track the measures in the recommendations to see whether the actions work.
"""

(OUT_DIR / "conclusion.md").write_text(markdown, encoding="utf-8")
scorecard.to_csv(OUT_DIR / "final_scorecard.csv")

# ---------- Verification ----------
display(scorecard)
show(Markdown(markdown))
print(f"Rows used: {rows_used}")
print(f"Conclusion saved to {OUT_DIR}/conclusion.md and final_scorecard.csv")
```

#### Output

|                    | model               | test result                    | business value                                      | verdict                                      |
|:-------------------|:--------------------|:-------------------------------|:----------------------------------------------------|:---------------------------------------------|
| 1. Cost overruns   | Logistic Regression | ROC AUC 0.83                   | 58% of overrun dollars in the riskiest 40% of lines | Good ranking, dollars follow budget size     |
| 2. Schedule delay  | Logistic Regression | ROC AUC 0.80                   | 63% of delay days in the riskiest 40% of tasks      | Good ranking                                 |
| 3. Site safety     | Logistic Regression | ROC AUC 0.86                   | 83% of lost days in the riskiest 40% of incidents   | Strongest model                              |
| 4. Late delivery   | Logistic Regression | ROC AUC 0.60                   | 50% of late days in the riskiest 40% of orders      | Weak, use only as a rough guide              |
| 5. Price anomalies | Isolation Forest    | 62 of 62 known anomalies found | 88% of overpaid dollars in 5% of orders             | Excellent here, but a simple rule matches it |

#### Conclusion

#### What we set out to do
Build machine learning models on six construction tables (projects, vendors, cost ledger, tasks, material orders and safety incidents) that help managers control cost, schedule, safety and procurement risk.

#### What we did
We audited and cleaned the data, engineered features, built a star schema, and prepared one analytical base table per business objective. We kept whole projects together in the train and test split so that results reflect new projects, not memorised ones. We compared four models for each classification objective and used an Isolation Forest for price anomalies. The saved models, scalers and feature lists are in `construction_models/`.

#### What we found
- Cost overruns are driven by contract type, cost category and country, not by project size. Cost Plus and Materials lines are the main hotspots.
- Delays are a staffing and weather problem. Small crews and the rainy season account for most of the risk.
- Serious safety incidents concentrate in height work without PPE, falls, inexperienced workers and night shifts.
- Late deliveries are hard to predict from the data we have. International orders are late more often, and that is the main lever.
- A small set of orders carries most of the overpricing, and the Isolation Forest finds all 62 known anomalies in a review list of 150 orders.

#### What the models are worth
Reviewing only the riskiest slice finds 58% of overrun dollars, 63% of delay days and 83% of lost safety days, against 40% for a random review. The price-anomaly list finds 88% of overpaid dollars in 5% of orders.

#### Limits
- Test sets are small, with 16 unseen projects each, so exact scores can move. The rankings are more reliable than single numbers.
- Group differences show a link, not proof of cause. The what-if figures are upper bounds.
- The rainy-season calendar and the 15% price tolerance are assumptions.
- Price anomalies are measured against known anomalies, which are easier to find than subtle real ones, and a simple price rule performs just as well here.
- The late-delivery model is weak, and it should be used only as a rough guide.

#### Next steps
Put the models behind a simple app so managers can score new cost lines, tasks, incidents and orders. Retrain every quarter. Collect better delivery data (carrier, port, supplier lead times) to improve the weakest model. Track the measures in the recommendations to see whether the actions work.

```text
Rows used: {'cost_overrun': 2413, 'schedule_delay': 3237, 'site_safety': 750, 'late_delivery': 2918, 'price_anomalies': 3000}
Conclusion saved to construction_business/conclusion.md and final_scorecard.csv
```

---

## 24. Run It Yourself

**Live app:** [construction-project-business-operations-predictive-support-sy.streamlit.app](https://construction-project-business-operations-predictive-support-sy.streamlit.app/)

Each page of the app takes plain inputs, works out the engineered features itself, and returns a probability, a Low / Medium / High risk level, the main reasons behind the score and a suggested action. The price page can also score a whole CSV of orders at once.

```
construction-project-business-operations-predictive-support-system/
├── app.py                    # Streamlit app
├── requirements.txt
├── construction_models/      # saved models, scalers, encoders, features (.joblib)
├── images/                   # charts used in this README
├── notebook/                 # the step-by-step Python scripts
└── README.md
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

> Use the same scikit-learn version the models were trained with, so the saved files load correctly.

---

**Author:** Adewale Oshinukan, Data Scientist, Lagos, Nigeria
[GitHub](https://github.com/waleoshin75-sketch) | [Portfolio](https://waleoshin75-sketch.github.io/web-portfolio/)
