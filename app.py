from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Travel Insurance Predictor", page_icon="✈️", layout="centered")

MODEL_PATH = Path(__file__).parent / "travel_insurance_model.pkl"
FEATURES = [
    "Age",
    "Employment Type",
    "AnnualIncome",
    "FamilyMembers",
    "ChronicDiseases",
    "FrequentFlyer",
    "EverTravelledAbroad",
]

st.title("✈️ Travel Insurance Prediction")
st.write(
    "Enter a customer's details to see the prediction from your trained Random Forest model. "
    "This is a demonstration, not a guarantee of future purchasing behaviour."
)

if not MODEL_PATH.exists():
    st.error("The trained model file is missing: travel_insurance_model.pkl")
    st.markdown("""
    **Connect your trained model first:**
    1. Open your `TRAVEL_INS(1).IPYNB` notebook.
    2. Run the model-export cell shown in `README.md` after `grid_search_rf.fit(...)` has completed.
    3. Copy `travel_insurance_model.pkl` into the same folder as `app.py`.
    4. Restart the app.
    """)
    st.stop()

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the model file: {exc}")
    st.info("Export the model using the same Python/scikit-learn environment in which it was trained.")
    st.stop()

st.subheader("Customer details")
with st.form("customer_form"):
    age = st.slider("Age", min_value=18, max_value=80, value=30)
    employment_label = st.selectbox(
        "Employment type",
        ["Government Sector", "Private Sector / Self Employed"],
    )
    annual_income = st.number_input(
        "Annual income (₹)", min_value=0, max_value=100_000_000, value=800_000, step=50_000
    )
    family_members = st.number_input(
        "Number of family members", min_value=1, max_value=30, value=4, step=1
    )
    chronic_diseases = st.radio("Has chronic disease?", ["No", "Yes"], horizontal=True)
    frequent_flyer = st.radio("Frequent flyer?", ["No", "Yes"], horizontal=True)
    travelled_abroad = st.radio("Travelled abroad before?", ["No", "Yes"], horizontal=True)
    submitted = st.form_submit_button("Predict purchase likelihood", type="primary")

if submitted:
    # These mappings reproduce the encoding used in the uploaded notebook.
    customer = pd.DataFrame([{
        "Age": int(age),
        "Employment Type": 1 if employment_label == "Government Sector" else 0,
        "AnnualIncome": float(annual_income),
        "FamilyMembers": int(family_members),
        "ChronicDiseases": 1 if chronic_diseases == "Yes" else 0,
        "FrequentFlyer": 1 if frequent_flyer == "Yes" else 0,
        "EverTravelledAbroad": 1 if travelled_abroad == "Yes" else 0,
    }], columns=FEATURES)

    try:
        prediction = int(model.predict(customer)[0])
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.caption("Check that the exported model was trained on the seven features in the README, with the same encodings.")
        st.stop()

    st.divider()
    st.subheader("Model result")
    if prediction == 1:
        st.success("Prediction: Customer is likely to purchase travel insurance (class 1).")
    else:
        st.info("Prediction: Customer is unlikely to purchase travel insurance (class 0).")

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(customer)[0]
            classes = list(model.classes_)
            if 1 in classes:
                positive_probability = float(probabilities[classes.index(1)])
                st.metric("Estimated purchase probability", f"{positive_probability:.1%}")
                st.progress(max(0.0, min(1.0, positive_probability)))
                st.caption("This is the model's estimated probability, not a guarantee. Probability calibration has not been verified here.")
        except Exception:
            st.caption("The model returned a class prediction; probability display was unavailable.")

    st.caption("Model output is for learning and demonstration. Do not use it as the sole basis for customer decisions.")
