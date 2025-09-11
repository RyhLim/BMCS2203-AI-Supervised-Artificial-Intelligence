import streamlit as st
import pandas as pd
import joblib
import json
from streamlit import rerun

# Load dataset
@st.cache_data # caches output to avoid reloading on every interaction
def load_dataset():
    df = pd.read_csv("crime_district.csv")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["weekday"] = df["date"].dt.day_name()
    df = df[df["state"].str.lower() != "malaysia"]  # remove aggregate 
    return df

df = load_dataset()

# Load artifacts -> Load models
with open("artifacts/evaluation_summary.json") as f:
    eval_summary = json.load(f)
    
# loop every model name and load them   
reg_models = {} 

for k in eval_summary["regression"].keys(): 
    reg_models[k] = joblib.load(f"artifacts/regression_{k}.joblib")

clf_models = {}
for k in eval_summary["classification"].keys():
    clf_models[k] = joblib.load(f"artifacts/classification_{k}.joblib")
    
# Sidebar: dropdowns
st.sidebar.subheader("Choose Models")
selected_reg = st.sidebar.selectbox("Select Regression Model", list(reg_models.keys()))
selected_clf = st.sidebar.selectbox("Select Classification Model", list(clf_models.keys()))

reg_model = reg_models[selected_reg]
clf_model = clf_models[selected_clf]

# Sidebar: Summary
st.sidebar.subheader("Crime Prediction Summary")

st.sidebar.subheader("Regression Models")
for k, v in eval_summary["regression"].items():
    st.sidebar.write(f"- {k}: R2={v['R2']:.3f}, MAE={v['MAE']:.2f}, RMSE={v['RMSE']:.2f}")

st.sidebar.subheader("Classification Models")
for k, v in eval_summary["classification"].items():
    st.sidebar.write(f"- {k}: Accuracy={v['Accuracy']:.3f}, F1_macro={v['F1_macro']:.3f}")

st.sidebar.markdown("---")
st.sidebar.write(f"**Best Regression Model:** {eval_summary['best_regression']}")
st.sidebar.write(f"**Best Classification Model:** {eval_summary['best_classification']}")
st.sidebar.subheader("Severity Classes")
st.sidebar.write(", ".join(eval_summary["classes"]))

# Input Form
def input_form(df, prefix):
    col1, col2 = st.columns(2)

    states = ["All"] + sorted(df["state"].dropna().unique())
    categories = ["All"] + sorted(df["category"].dropna().unique())

    with col1:
        state = st.selectbox("State", states, key=f"{prefix}_state")

        if state == "All":
            districts = ["All"]
        else:
            districts = ["All"] + sorted(df[df["state"] == state]["district"].dropna().unique())
        district = st.selectbox("District", districts, key=f"{prefix}_district")

        category = st.selectbox("Category", categories, key=f"{prefix}_category")

        if category == "All":
            types = ["All"]
        else:
            types = ["All"] + sorted(df[df["category"] == category]["type"].dropna().unique())
        type_ = st.selectbox("Type", types, key=f"{prefix}_type")

    with col2:
        year = st.number_input("Year", min_value=int(df["year"].min()),
                               max_value=2100, value=2025, key=f"{prefix}_year")
        month = st.number_input("Month", min_value=1, max_value=12, value=1, key=f"{prefix}_month")
        weekday = st.selectbox("Weekday", df["weekday"].unique(), key=f"{prefix}_weekday")

    if st.button("Reset Form", key=f"{prefix}_reset"):
        for k in list(st.session_state.keys()):
            if k.startswith(prefix):
                del st.session_state[k]
        rerun()

    return {
        "state": state,
        "district": district,
        "category": category,
        "type": type_,
        "year": year,
        "month": month,
        "weekday": weekday
    }
    return sample
    
# Helpers
def expand_inputs(df, selection):
    """Expand 'All' selections into multiple prediction rows"""
    states = [selection["state"]] if selection["state"] != "All" else sorted(df["state"].unique())
    results = []

    for state in states:
        districts = (
            [selection["district"]] if selection["district"] != "All"
            else sorted(df[df["state"] == state]["district"].unique())
        )
        for district in districts:
            categories = (
                [selection["category"]] if selection["category"] != "All"
                else sorted(df["category"].unique())
            )
            for category in categories:
                types = (
                    [selection["type"]] if selection["type"] != "All"
                    else sorted(df[df["category"] == category]["type"].unique())
                )
                for type_ in types:
                    results.append({
                        "state": state,
                        "district": district,
                        "category": category,
                        "type": type_,
                        "year": selection["year"],
                        "month": selection["month"],
                        "weekday": selection["weekday"],
                    })
    return pd.DataFrame(results)

# Main Tabs
tabs = st.tabs(["Dataset Preview", "Predict Crime Count", "Predict Severity"])

with tabs[0]:
    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    st.markdown("### Crime Trends by Year")
    trend = df.groupby("year")["crimes"].sum().reset_index()
    st.line_chart(trend.set_index("year"))

    st.markdown("### Crime Distribution by State")
    state_dist = df.groupby("state")["crimes"].sum().sort_values(ascending=False).reset_index()
    st.bar_chart(state_dist.set_index("state"))

with tabs[1]:
    st.subheader("Predict Crime Counts (Regression)")
    selection = input_form(df, prefix="reg")
    if st.button("Predict Crime Count", key="predict_reg"):
        samples = expand_inputs(df, selection)
        preds = reg_model.predict(samples)
        samples["predicted_crimes"] = preds.astype(int)
        st.dataframe(samples)

with tabs[2]:
    st.subheader("Predict Severity Level (Classification)")
    selection = input_form(df, prefix="clf")
    if st.button("Predict Severity", key="predict_clf"):
        samples = expand_inputs(df, selection)
        preds = clf_model.predict(samples)

        # map numeric predictions 0,1,2 to labels (low, medium, high)
        class_labels = eval_summary["classes"] 
        samples["predicted_severity"] = [class_labels[p] for p in preds]

        st.dataframe(samples)

        # show prediction result summary
        st.success(f"Prediction: {samples['predicted_severity'].iloc[0]}")