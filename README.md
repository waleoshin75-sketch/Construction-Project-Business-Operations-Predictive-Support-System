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

---

## 4. Data Quality Audit

I measured every problem **before** touching anything, so each later fix traces back to a number.

What was checked:
- Table overview and missing values by column
- Business-rule breaks: end dates before start dates, negative costs, impossible crew sizes
- Orphan keys (cost, task and order rows pointing to projects that do not exist)
- Spelling variants in every category column
- The Objective 5 answer key, read only to confirm it matches the orders table

---

## 5. Data Cleaning and Error Checking

Every fix is written to a cleaning log, and a verification step proves the result: no duplicate keys, no impossible dates, no unmapped labels, no orphan codes.

Two decisions worth knowing:
- **Price spikes were kept on purpose.** They are what Objective 5 has to find.
- **Missing numbers were filled with a group median** (for example, crew size by task type), never a global guess.

### Cleaning code: helper functions

```python
MISSING_TOKENS = {"", "n/a", "na", "null", "none", "nan", "-", "tbd", "pending", "??",
                  "00/00/0000", "not delivered"}

def tidy_text(series):
    """Trim, collapse repeated spaces and turn placeholder text into a real missing value."""
    text = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return text.mask(text.str.lower().isin(MISSING_TOKENS))

def canonical(series, valid_values, aliases=None):
    """Map any spelling to one official label. Unknown spellings become missing."""
    lookup = {re.sub(r"[^a-z0-9]+", "", v.lower()): v for v in valid_values}
    for alias, official in (aliases or {}).items():
        lookup[re.sub(r"[^a-z0-9]+", "", alias.lower())] = official
    text = tidy_text(series).str.replace("³", "3", regex=False)
    return squash(text).map(lookup).astype("string")

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

def fill_with_group_median(frame, column, group):
    group_median = frame.groupby(group)[column].transform("median")
    frame[column] = frame[column].fillna(group_median).fillna(frame[column].median())
```

### Cleaning code: dates and table-level fixes

```python
# Each messy date style has its own separator, so every format is read exactly.
DATE_PATTERNS = [
    (re.compile(r"^\d{4}-\d{2}-\d{2}$"), "%Y-%m-%d"),
    (re.compile(r"^\d{2}/\d{2}/\d{4}$"), "%d/%m/%Y"),
    (re.compile(r"^\d{2}-\d{2}-\d{4}$"), "%m-%d-%Y"),
    (re.compile(r"^\d{4}/\d{2}/\d{2}$"), "%Y/%m/%d"),
    (re.compile(r"^\d{1,2} [A-Za-z]{3} \d{4}$"), "%d %b %Y"),
    (re.compile(r"^[A-Za-z]+ \d{1,2}, \d{4}$"), "%B %d, %Y"),
]

# Tasks: put swapped actual start/end dates back in order, fix impossible crew sizes
swapped = tasks["actual_end"] < tasks["actual_start"]
tasks.loc[swapped, ["actual_start", "actual_end"]] = tasks.loc[swapped, ["actual_end", "actual_start"]].values
bad_crew = tasks["crew_size"].le(0) | tasks["crew_size"].gt(100)
tasks.loc[bad_crew, "crew_size"] = np.nan
fill_with_group_median(tasks, "crew_size", "task_type")

# Orders: rebuild a missing quantity from total cost / unit price, else use the material median
rebuilt = (orders["total_cost_usd"] / orders["unit_price_usd"]).round()
can_rebuild = orders["quantity"].isna() & rebuilt.gt(0)
orders.loc[can_rebuild, "quantity"] = rebuilt[can_rebuild]
fill_with_group_median(orders, "quantity", "material")
```

---

## 6. Feature Engineering

Raw columns become signals a model can use, and the four targets are created here. Every feature is logged with its meaning.

### Feature engineering code

```python
# Projects
p["budget_per_sqm"] = (p["budget_usd"] / p["site_area_sqm"]).round(2)
p["high_rise"] = (p["num_floors"] >= 30).astype(int)
p["is_public"] = (p["client_sector"] == "Public").astype(int)

# Cost ledger (Objective 1)
l["overrun_pct"] = (l["actual_usd"] / l["budgeted_usd"] - 1).round(4)
l["overrun_flag"] = (l["overrun_pct"] > 0.15).astype(int)
l["budget_share_of_project"] = l["budgeted_usd"] / l.groupby("project_id")["budgeted_usd"].transform("sum")
l["log_budgeted_usd"] = np.log1p(l["budgeted_usd"]).round(4)

# Tasks (Objective 2)
t["planned_duration_days"] = (t["planned_end"] - t["planned_start"]).dt.days
t["rainy_season"] = in_rainy_season(t["task_country"], t["planned_start"].dt.month)
t["delay_days"] = (t["actual_end"] - t["planned_end"]).dt.days.where(finished)
t["delay_flag"] = (t["delay_days"] > 5).astype(float).where(finished)

# Material orders (Objectives 4 and 5)
o["lead_time_days"] = (o["promised_date"] - o["order_date"]).dt.days
o["late_delivery"] = (o["delivered_date"] > o["promised_date"]).astype(float).where(
    o["delivered_date"].notna() & o["promised_date"].notna())
o["international_supplier"] = (o["supplier_country"] != o["project_country"]).astype(float)

# Price features: compare each price with its peers
material_median = o.groupby("material")["unit_price_usd"].transform("median")
group_median = o.groupby(["material", "project_country"])["unit_price_usd"].transform("median")
group_count = o.groupby(["material", "project_country"])["unit_price_usd"].transform("count")
peer_median = group_median.where(group_count >= 20, material_median)
o["price_ratio_material"] = (o["unit_price_usd"] / material_median).round(4)
o["price_ratio_peer"] = (o["unit_price_usd"] / peer_median).round(4)

# Safety incidents (Objective 3)
s["serious_flag"] = s["severity"].isin(["Lost Time Injury", "Fatality"]).astype(int)
s["no_ppe_at_height"] = ((s["ppe_compliant"] == "No") & (s["work_at_height"] == "Yes")).astype(int)
```

> The rainy-season calendar (Nigeria and India June to September, Brazil December to March, UK October to January) is a domain assumption, not something measured from the data.

---

## 7. Create Relationship Table

Every fact table links to the dimension tables (projects, vendors, dates) through a star schema. A join test confirms that a many-to-one join never adds or loses a row.

![Star schema](images/star_schema.png)

---

## 8. Exploratory Data Analysis

Nine charts answered the business questions before any modelling started. The headline: problems are the norm, not the exception.

| Outcome | Share of records |
|---------|:----------------:|
| Cost lines that overran by more than 15% | **47%** |
| Tasks that finished more than 5 days late | **41%** |
| Incidents that were serious | **25%** |
| Material orders that arrived late | **46%** |

---

## 9. Analytical Base Table

One modelling table per objective. Each holds **only what is known at the moment of the decision**. Anything that is a result of the outcome (actual cost, actual dates, severity, lost days, delivery date, the answer key) is treated as leakage and left out. This is why the scores are honest.

---

## 10. Encode Categorical Columns

One-Hot Encoding turns text into 0/1 columns.
- Missing text gets its own `Unknown` label, so no row is lost.
- A label the model has never seen encodes as all zeros instead of crashing the app.
- The encoders are saved, so new rows in the app are encoded exactly the same way.

---

## 11. Separate X and Y

For each objective the target becomes `y` and the allowed inputs become `X`. Checks confirm that no target or leakage column is hiding inside `X`.

---

## 12. Train Test Split

The split keeps **whole projects together** (`GroupShuffleSplit`): a project is either entirely in training or entirely in testing. The test set is 16 projects the model has never seen. A random row split would let the model memorise a project and make the results look better than they are.

---

## 13. Feature Scaling of Numerical Columns

A median imputer plus `StandardScaler` for numeric columns, and a most-frequent imputer for 0/1 columns. Everything is fitted on the training rows only, so nothing from the test set leaks in.

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

---

## 15. Predict X_test

Each model scores the unseen test projects, and the probabilities are turned into alerts with the cut-offs above.

---

## 16. Compare X and Y

Predicted and actual outcomes side by side, to see where each model agrees with reality and where it misses.

| Objective 1: Cost overruns | Objective 2: Schedule delays |
|:-:|:-:|
| ![Objective 1](images/objective_1_compare_x_and_y.png) | ![Objective 2](images/objective_2_compare_x_and_y.png) |
| **Objective 3: Site safety** | **Objective 4: Late delivery** |
| ![Objective 3](images/objective_3_compare_x_and_y.png) | ![Objective 4](images/objective_4_compare_x_and_y.png) |

**Objective 5: Price anomalies**

![Objective 5](images/objective_5_compare_x_and_y.png)

---

## 17. Evaluate Model

| Objective | Test ROC AUC | Reading |
|-----------|:------------:|---------|
| 1. Cost overrun | **0.83** | Strong. Range across resampled projects: 0.75 to 0.87 |
| 2. Schedule delay | **0.80** | Strong |
| 3. Site safety | **0.86** | Strong |
| 4. Late delivery | 0.60 | Weak. Use as a rough guide only |
| 5. Price anomalies | n/a | Finds all 62 planted anomalies in a review list of 150 orders |

![Objective 1 evaluation](images/objective_1_evaluate_model.png)

> **Honest note:** the test set is only 16 projects, so exact scores can move. The rankings are more reliable than any single number. For Objective 5, a simple rule (price above 2x the material norm) also finds all 62, so the model's real value is on subtler cases that real invoices may hold.

---

## 18. Business Analysis

What does a model actually save? Compare reviewing the riskiest slice of items against reviewing at random.

| Objective | Review budget | Share of the loss found | Random review |
|-----------|:-------------:|:-----------------------:|:-------------:|
| 1. Overrun dollars | Top 40% | **58%** | 40% |
| 2. Delay days | Top 40% | **63%** | 40% |
| 3. Lost safety days | Top 40% | **83%** | 40% |
| 5. Money paid above the norm | Top 5% of orders | **88%** | 5% |

![Business analysis](images/business_analysis.png)

---

## 19. Save Model, Scaler and Features

Everything the app needs is saved with `joblib`: for each model, the model, the scaler, the encoder and a features file (column order, categories, alert cut-off). The price model also saves its price reference tables.

```python
joblib.dump(model, "construction_models/cost_overrun_model.joblib")
joblib.dump(scaler, "construction_models/cost_overrun_scaler.joblib")
joblib.dump(encoder, "construction_models/cost_overrun_encoder.joblib")
joblib.dump(features, "construction_models/cost_overrun_features.joblib")
```

A round-trip test reloads every file and checks it reproduces the same predictions.

---

## 20. Download Model, Scaler and Features

All 21 saved files are zipped into one `construction_models.zip` and copied to the Downloads folder. The step re-opens the zip, confirms it is not corrupt, and loads every file again to prove nothing was damaged.

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

---

## 23. Conclusion

Cost, schedule and safety risk can be ranked well enough to point managers at the slice of work that matters most, and a short review list catches most of the pricing risk. Late delivery is the exception: better delivery data (carrier, port, supplier lead times) is the way to improve it.

**Limits to keep in mind**
- Test sets are small (16 unseen projects), so exact scores can move.
- Group differences show a link, not proof of cause. Impact figures are upper bounds.
- The rainy-season calendar and the 15% price tolerance are assumptions.
- Price anomalies are measured against planted anomalies, which are easier to find than subtle real ones.

**Next steps:** retrain quarterly, collect better delivery data, and track the recommendation measures to see whether the actions work.

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
