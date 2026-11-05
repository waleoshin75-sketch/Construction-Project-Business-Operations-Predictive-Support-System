# Construction Project Business Operations: Predictive Decision Support System

**Predicting cost overruns, schedule delays, serious safety incidents, late material deliveries and suspicious supplier prices.**

[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-ff4b4b?logo=streamlit&logoColor=white)](YOUR_STREAMLIT_LINK)
![Python](https://img.shields.io/badge/Python-3.x-3776ab?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-models-f7931e?logo=scikit-learn&logoColor=white)

**Try it:** [Open the live prediction app](YOUR_STREAMLIT_LINK)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Business Problem and Objectives](#business-problem-and-objectives)
3. [Dataset](#dataset)
4. [Tools and Libraries](#tools-and-libraries)
5. [Data Quality Audit](#data-quality-audit)
6. [Data Cleaning](#data-cleaning)
7. [Feature Engineering](#feature-engineering)
8. [Star Schema](#star-schema)
9. [Exploratory Data Analysis](#exploratory-data-analysis)
10. [Analytical Base Table](#analytical-base-table)
11. [Encoding Categorical Columns](#encoding-categorical-columns)
12. [Separating X and y](#separating-x-and-y)
13. [Train and Test Split](#train-and-test-split)
14. [Feature Scaling](#feature-scaling)
15. [Model Training](#model-training)
16. [Predicting on the Test Set](#predicting-on-the-test-set)
17. [Comparing Predictions and Actuals](#comparing-predictions-and-actuals)
18. [Model Evaluation](#model-evaluation)
19. [Business Analysis](#business-analysis)
20. [Saving the Model, Scaler and Features](#saving-the-model-scaler-and-features)
21. [Key Findings and Insights](#key-findings-and-insights)
22. [Recommendations](#recommendations)
23. [Conclusion](#conclusion)
24. [Prediction App](#prediction-app)
25. [Repository Structure and How to Run](#repository-structure-and-how-to-run)
26. [Author](#author)

---

## Project Overview

A construction firm runs many projects at once, and problems show up late: a cost line that quietly overruns, a task that slips, an incident that turns serious, a delivery that never arrives, an invoice with a padded price.

This project builds **five predictive models** that tell managers where to look first, and wraps them in a simple Streamlit app anyone can use.

## Business Problem and Objectives

| # | Objective | What is predicted | Model |
|---|-----------|-------------------|-------|
| 1 | Control project cost overruns | Will a cost line overrun its budget by more than 15%? | Logistic Regression |
| 2 | Reduce schedule delays | Will a task finish more than 5 days late? | Logistic Regression |
| 3 | Improve site safety | Is an incident serious (lost-time injury or fatality)? | Logistic Regression |
| 4 | Make material procurement more reliable | Will a material order arrive late? | Logistic Regression |
| 5 | Detect price and invoice anomalies | Is the unit price far from the material norm? | Isolation Forest |

## Dataset

Six tables in a star-style layout: two dimension tables and four fact tables, plus an answer key used only to check Objective 5.

| Table | Type | Raw rows | Clean rows |
|-------|------|---------:|-----------:|
| `dim_projects` | Dimension | 84 | 80 |
| `dim_vendors` | Dimension | 48 | 45 |
| `fact_cost_ledger` | Fact | 2,465 | 2,413 |
| `fact_tasks` | Fact | 4,045 | 3,985 |
| `fact_material_orders` | Fact | 3,035 | 3,000 |
| `fact_safety_incidents` | Fact | 768 | 750 |

## Tools and Libraries

Python, pandas, NumPy, scikit-learn, matplotlib, joblib and Streamlit. Everything is done in Python.

## Data Quality Audit

Before changing anything, I measured the problems: missing values, business-rule breaks (end before start, negative costs), orphan project codes, mixed spellings and date styles. Nothing was altered at this stage, so every later fix can be traced back to a measured problem.

## Data Cleaning

The raw data was deliberately messy: five date formats, currencies written as `12.5M` or `USD 12,500`, country names like `USA` and `Deutschland`, ratings like `4/5` and `five`, duplicate rows and orphan project codes. Each fix is logged in a cleaning log, and the cleaned tables pass a verification step (no duplicate keys, no impossible dates, no unmapped labels).

Price spikes were **kept on purpose**, because they are what Objective 5 must find.

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
# Each messy date style has its own separator, so every format can be read exactly.
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

## Feature Engineering

New columns turn raw records into signals a model can use. Each feature is logged with its meaning. The targets are created here too.

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

> The rainy-season calendar (Nigeria and India June to September, Brazil December to March, UK October to January) is a domain assumption, not something taken from the data.

## Star Schema

Every fact table is linked to the dimension tables (projects, vendors, dates). A join test confirms that a many-to-one join never adds or loses rows.

![Star schema](images/star_schema.png)

## Exploratory Data Analysis

Nine charts answer the business questions before any modelling: how big the overruns are and where, what drives delays, where serious incidents cluster, what makes deliveries late, and how prices behave. The headline numbers: **47%** of cost lines overran by more than 15%, **41%** of tasks finished late, **25%** of incidents were serious and **46%** of orders arrived late.

## Analytical Base Table

One modelling table is built per objective. Each table holds **only what is known at the moment of the decision**. Anything that is a result of the outcome (actual cost, actual dates, severity, delivery date, the answer key) is treated as leakage and left out.

## Encoding Categorical Columns

A One-Hot Encoder turns text columns into 0/1 columns. Missing text gets its own `Unknown` label so no row is lost, and a label never seen before encodes as all zeros instead of crashing the app. The encoders are saved so new rows are encoded the same way.

## Separating X and y

For each objective, the target column becomes `y` and the allowed inputs become `X`. Checks confirm the target is not hiding inside `X`.

## Train and Test Split

The split keeps **whole projects together** (`GroupShuffleSplit`), so the test set holds 16 projects the model has never seen. A random row split would let the model memorise a project and look better than it really is.

## Feature Scaling

A median imputer plus `StandardScaler` for numeric columns, and a most-frequent imputer for 0/1 columns. Everything is fitted on the training rows only.

## Model Training

For each classification objective, four models were compared from the simplest to the strongest, using GroupKFold cross-validation (whole projects held out in each fold) and average precision. **Logistic Regression won all four times**, so the final models are simple and explainable. The alert cut-off for each model was chosen on out-of-fold training predictions, never on the test set.

| Objective | Alert cut-off |
|-----------|--------------:|
| 1 Cost overrun | 0.36 |
| 2 Schedule delay | 0.28 |
| 3 Site safety | 0.37 |
| 4 Late delivery | 0.46 |

Objective 5 uses an **Isolation Forest** (500 trees, 5% contamination) on four price features. It needs no labels: it learns what a normal order looks like and scores how far each order is from it.

## Predicting on the Test Set

Each model scores the unseen test projects. Probabilities are turned into alerts with the cut-offs above.

## Comparing Predictions and Actuals

Predicted and actual outcomes are compared side by side to see where the models agree and where they miss.

| Objective 1 | Objective 2 |
|:-:|:-:|
| ![Objective 1](images/objective_1_compare_x_and_y.png) | ![Objective 2](images/objective_2_compare_x_and_y.png) |
| **Objective 3** | **Objective 4** |
| ![Objective 3](images/objective_3_compare_x_and_y.png) | ![Objective 4](images/objective_4_compare_x_and_y.png) |

![Objective 5](images/objective_5_compare_x_and_y.png)

## Model Evaluation

| Objective | Test ROC AUC | Reading |
|-----------|-------------:|---------|
| 1 Cost overrun | **0.83** | Strong. Range across resampled projects: 0.75 to 0.87 |
| 2 Schedule delay | **0.80** | Strong |
| 3 Site safety | **0.86** | Strong |
| 4 Late delivery | **0.60** | Weak: use as a rough guide only |
| 5 Price anomalies | n/a | Finds all 62 planted anomalies in a list of 150 orders |

![Objective 1 evaluation](images/objective_1_evaluate_model.png)

> Test sets are small (16 unseen projects), so exact scores can move. The rankings are more reliable than single numbers.

## Business Analysis

Reviewing only the riskiest slice of items finds far more of the loss than reviewing at random:

- **58%** of overrun dollars, **63%** of delay days and **83%** of lost safety days, against 40% for a random review of the same size.
- The price-anomaly list finds **88%** of money paid above the norm in just **5%** of orders.

![Business analysis](images/business_analysis.png)

## Saving the Model, Scaler and Features

Everything the app needs is saved with `joblib`: for each model, the model, scaler, encoder and a features file (column order, categories, alert cut-off). The price model also saves its price reference tables.

```python
joblib.dump(model, "model/cost_overrun_model.joblib")
joblib.dump(scaler, "model/cost_overrun_scaler.joblib")
joblib.dump(encoder, "model/cost_overrun_encoder.joblib")
joblib.dump(features, "model/cost_overrun_features.joblib")
```

## Key Findings and Insights

- **Problems are common, not rare.** Control has to be part of routine work.
- **Contract type drives overruns.** Cost Plus lines overran 66% of the time, Fixed Price 36%. Materials lines overran 69% against 21% for Overheads.
- **Dollars follow budget size.** The largest 40% of budgets hold 76% of overrun dollars.
- **Understaffed tasks slip.** Crews of 18 or fewer were delayed 60% of the time, the rest 34%.
- **The rainy season roughly doubles delay risk.** 75% of rainy-season tasks were delayed against 36% in the dry season.
- **Height work without PPE is the most dangerous condition.** 59% of those incidents were serious, against 23% otherwise. Falls from height are the serious ones, and experience protects.
- **International orders arrive late more often.** 48% against 29% for local orders.
- **A small set of orders carries most of the overpricing.** And a simple rule (price above 2x the material norm) finds the planted anomalies as well as the Isolation Forest does, so start with the rule.

## Recommendations

1. **Enforce PPE on every job at height.** Low effort, large effect on serious injuries.
2. **Set a minimum crew size for critical tasks.** Crews of 18 or fewer are the high-risk group.
3. **Check invoices against the material norm before payment.** Start with the 2x rule, then review the model's top 150.
4. **Add cost-control incentives to Cost Plus contracts.**
5. **Run a weekly overrun review on the largest budget lines**, topped up by the model for risky smaller lines.
6. **Add rainy-season buffers and crew cover to schedules.**
7. **Control materials prices**, mainly in Nigeria and India.
8. **Buffer or localise material deliveries.**
9. **Add supervision on night shifts** and pair new workers with experienced ones.
10. **Use the models as ranking tools and retrain every quarter.**
11. **Capture better delivery and invoice data at the source.**

## Conclusion

Cost, schedule and safety risk can be ranked well enough to focus a manager's attention on the slice of work that matters most. Late delivery is the exception, and better delivery data (carrier, port, supplier lead times) is the way to improve it. Impact figures compare groups in historical data, so they show a link, not proof of cause.

**Limits to keep in mind:** small test sets, the rainy-season calendar and the 15% price tolerance are assumptions, and price anomalies are measured against planted anomalies, which are easier to find than subtle real ones.

## Prediction App

A Streamlit app brings all five models together. Each page takes plain inputs, works out the engineered features itself, and returns a probability, a Low / Medium / High risk level, the main reasons behind the score and a suggested action. The price page also scores a whole CSV of orders at once.

**[Open the live app](YOUR_STREAMLIT_LINK)**

## Repository Structure and How to Run

```
construction-prediction-system/
├── app.py                   # Streamlit app
├── requirements.txt
├── model/                   # saved models, scalers, encoders, features (.joblib)
├── images/                  # charts used in this README
├── notebook/                # the full step-by-step Python scripts
└── README.md
```

```bash
pip install -r requirements.txt
streamlit run app.py
```

> Use the same scikit-learn version the models were trained with, so the saved files load correctly.

## Author

**Adewale Oshinukan**, Data Scientist, Lagos, Nigeria
[GitHub](https://github.com/waleoshin75-sketch) | [Portfolio](https://waleoshin75-sketch.github.io/web-portfolio/)
