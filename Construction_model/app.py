"""
CONSTRUCTION BUSINESS OPERATIONS PREDICTIVE SUPPORT SYSTEM
-----------------------------------------------------------
A simple Streamlit app with five models:
  1. Cost overruns         - how likely is a cost line to overrun its budget by more than 15%?
  2. Schedule delays       - how likely is a task to finish more than 5 days late?
  3. Site safety           - how likely is an incident to be serious (lost-time injury or fatality)?
  4. Material procurement  - how likely is a material order to arrive late?
  5. Price anomalies       - is the unit price on an order suspiciously far from the norm?

How to run (in the VS Code terminal, in the folder that holds this file):
    pip install streamlit pandas numpy scikit-learn joblib
    streamlit run app.py

This file expects a folder called  construction_models  next to it, holding the
.joblib files you downloaded (models, scalers, encoders, features, price reference).
"""

from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ======================================================================================
# 1. PAGE SETUP AND SETTINGS
# ======================================================================================
st.set_page_config(page_title="Construction Predictive Support System", layout="wide")

MODEL_DIR = Path(__file__).parent / "construction_model"

CLASSIFIERS = ["cost_overrun", "schedule_delay", "site_safety", "late_delivery"]
PAGES = ["Home", "1. Cost overruns", "2. Schedule delays", "3. Site safety",
         "4. Material delivery", "5. Price anomalies"]

# Rainy months by country. This is a business ASSUMPTION used when the models were built.
RAINY_MONTHS = {"Nigeria": [6, 7, 8, 9], "India": [6, 7, 8, 9],
                "Brazil": [12, 1, 2, 3], "United Kingdom": [10, 11, 12, 1]}

PRICE_TOLERANCE = 1.15      # up to 15% above the material norm counts as normal price variation


# ======================================================================================
# 2. LOAD THE SAVED MODELS (done once, then remembered by Streamlit)
# ======================================================================================
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


# ======================================================================================
# 3. SHARED HELPER FUNCTIONS
# ======================================================================================
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


# ======================================================================================
# 4. PAGE: HOME
# ======================================================================================
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


# ======================================================================================
# 5. PAGE 1: COST OVERRUNS
# ======================================================================================
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


# ======================================================================================
# 6. PAGE 2: SCHEDULE DELAYS
# ======================================================================================
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


# ======================================================================================
# 7. PAGE 3: SITE SAFETY
# ======================================================================================
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


# ======================================================================================
# 8. PAGE 4: MATERIAL DELIVERY
# ======================================================================================
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


# ======================================================================================
# 9. PAGE 5: PRICE ANOMALIES
# ======================================================================================
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


# ======================================================================================
# 10. SIDEBAR MENU AND PAGE ROUTER
# ======================================================================================
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