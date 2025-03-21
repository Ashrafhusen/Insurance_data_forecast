import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/insurance_pricing_model.pkl')


st.set_page_config(page_title="Insurance Pricing Predictor", layout="wide")
st.title("Insurance Pricing Forecast")
st.markdown("Use this tool to estimate the likelihood of a customer purchasing an insurance policy based on their profile.")

st.markdown("---")


st.subheader("👤 Demographic Information")
with st.container():
    col1, col2, col3 = st.columns(3)
    region = col1.slider("Region Code (V1)", 1, 10, 5)
    age_group = col2.selectbox("Age Group (V2)", [1, 2, 3, 4, 5, 6])
    gender = col3.radio("Gender (V3)", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")

    marital_status = col1.radio("Marital Status (V4)", [0, 1], format_func=lambda x: "Single" if x == 0 else "Married")
    job_class = col2.slider("Job Class (V5)", 0, 5, 2)
    education = col3.slider("Education Level (V6)", 0, 5, 2)


st.subheader(" Household & Asset Ownership")
with st.container():
    col4, col5, col6 = st.columns(3)
    car_owner = col4.radio("Owns Car? (V7)", [0, 1])
    owns_home = col5.radio("Owns Home? (V8)", [0, 1])
    credit_card_holder = col6.radio("Credit Card Holder? (V9)", [0, 1])

    num_children = col4.slider("Number of Children (V10)", 0, 5, 1)
    household_size = col5.slider("Household Size (V11)", 1, 10, 2)
    years_with_company = col6.slider("Years with Company (V12)", 0, 30, 5)


st.subheader(" Financial & Behavioral Attributes")
with st.container():
    col7, col8, col9 = st.columns(3)
    V13 = col7.slider("Spending Score (V13)", 0, 5, 2)
    V14 = col8.slider("Savings Score (V14)", 0, 5, 2)
    V15 = col9.slider("Debt Index (V15)", 0, 10, 4)

    V16 = col7.slider("Loan Count (V16)", 0, 5, 1)
    V17 = col8.slider("Bank Visits per Year (V17)", 0, 12, 3)
    V18 = col9.slider("Online Logins per Month (V18)", 0, 20, 5)


st.subheader(" Product Ownership (V19–V85)")
st.caption("Select 0 or 1 for whether the customer owns each product (e.g., home insurance, fire policy, savings account, etc.)")
product_inputs = {}
with st.expander("Expand to input detailed product ownership info"):
    cols = st.columns(5)
    for i in range(19, 86):
        col = cols[(i - 19) % 5]
        product_inputs[f"V{i}"] = col.selectbox(f"Product V{i}", options=[0, 1], key=f"V{i}")


st.markdown("---")
if st.button("🚀 Predict Insurance Likelihood"):
    input_data = {
        "V1": region, "V2": age_group, "V3": gender, "V4": marital_status,
        "V5": job_class, "V6": education, "V7": car_owner, "V8": owns_home,
        "V9": credit_card_holder, "V10": num_children, "V11": household_size,
        "V12": years_with_company, "V13": V13, "V14": V14, "V15": V15,
        "V16": V16, "V17": V17, "V18": V18
    }
    input_data.update(product_inputs)

    df = pd.DataFrame([input_data])
    prediction = model.predict(df)

    st.success(f"✅ Predicted Likelihood of Insurance Purchase: **{prediction[0]:.2f}**")
    st.progress(int(min(prediction[0] * 100, 100)))

    st.info("A likelihood closer to 1.0 indicates a higher chance that this customer will buy an insurance policy.")

