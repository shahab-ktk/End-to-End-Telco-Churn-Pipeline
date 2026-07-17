import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

st.set_page_config(
    page_title="TelcoPulse: Enterprise Churn AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='color: #1e3a8a;'>📊 TelcoPulse: Enterprise Retention Console</h1>", unsafe_allow_html=True)
st.markdown("Use this workspace to calculate real-time customer attrition probability risks using our operational machine learning backend pipeline.")

st.sidebar.markdown("### 👥 Demographic Profile")
senior_citizen_label = st.sidebar.selectbox("Is Senior Citizen?", ["No", "Yes"])
senior_citizen = 1 if senior_citizen_label == "Yes" else 0
partner = st.sidebar.selectbox("Has Partner?", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents?", ["No", "Yes"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 💳 Account & Billing Parameters")
contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.sidebar.selectbox("Paperless Billing?", ["Yes", "No"])
payment_method = st.sidebar.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 Continuous Financial Fields")
tenure = st.sidebar.slider("Account Tenure (Months)", min_value=1, max_value=72, value=12)
monthly_charges = st.sidebar.number_input("Monthly Charges ($)", min_value=10.0, max_value=150.0, value=70.0, step=0.5)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=10.0, max_value=8500.0, value=840.0, step=1.0)


st.markdown("### 🛠️ Core Network & Digital Utility Subscriptions")
col1, col2, col3 = st.columns(3)

with col1:
    internet_service = st.selectbox("Internet Service Provider", ["Fiber optic", "DSL", "No"])
    multiple_lines = st.selectbox("Multiple Phone Lines", ["No", "Yes", "No phone service"])
    online_security = st.selectbox("Online Security Add-on", ["No", "Yes", "No internet service"])

with col2:
    online_backup = st.selectbox("Online Backup Utility", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection Coverage", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Technical Assistance Plan", ["No", "Yes", "No internet service"])

with col3:
    streaming_tv = st.selectbox("Streaming TV Service", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies Service", ["No", "Yes", "No internet service"])

st.markdown("---")


if st.button("🚀 Calculate Retention Analytics", type="primary", use_container_width=True):
    with st.spinner("Processing customer profiles across transformation matrix arrays..."):
        try:
            
            customer_payload = CustomData(
                SeniorCitizen=senior_citizen, Partner=partner, Dependents=dependents,
                tenure=tenure, MultipleLines=multiple_lines, InternetService=internet_service,
                OnlineSecurity=online_security, OnlineBackup=online_backup,
                DeviceProtection=device_protection, TechSupport=tech_support,
                StreamingTV=streaming_tv, StreamingMovies=streaming_movies, 
                Contract=contract, PaperlessBilling=paperless_billing,
                PaymentMethod=payment_method, MonthlyCharges=monthly_charges,
                TotalCharges=total_charges
            )
            
            # 2. Extract input variables as a clean dataframe payload
            raw_input_df = customer_payload.get_data_as_data_frame()
            
            # 3. Pass data frame row into your verified prediction pipeline logic
            pipeline = PredictPipeline()
            predicted_class, predicted_probability = pipeline.predict_live_input(raw_input_df)
            
            # Extract numerical float value out of array and handle format safely
            risk_percentage = float(predicted_probability[0] * 100) if hasattr(predicted_probability, '__len__') else float(predicted_probability * 100)
            final_class = int(predicted_class[0]) if hasattr(predicted_class, '__len__') else int(predicted_class)
            
            # 4. Display Results Panels dynamically based on threshold risks
            st.markdown("### 🔮 Predictive Inference Results")
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                if final_class == 1:
                    st.error("🚨 **Classification Status: HIGH CHURN RISK**")
                else:
                    st.success("✅ **Classification Status: STABLE ACCOUNT**")
                
                st.metric(
                    label="Calculated Attrition Probability", 
                    value=f"{risk_percentage:.2f}%"
                )
            
            with res_col2:
                # Add a visual progress tracker bar matching risk weights
                st.progress(min(max(risk_percentage / 100.0, 0.0), 1.0))
                
                # Contextual corporate recommendations
                st.markdown("** Retention Action Plan:**")
                if risk_percentage >= 70:
                    st.warning("👉 *Immediate Action Required:* Dispatch an auto-pay incentive credit or propose a long-term contract migration promotion to stabilize this user's pricing threshold.")
                elif 40 <= risk_percentage < 70:
                    st.info("👉 *Proactive Care Recommended:* Offer an ecosystem bundle enhancement (e.g., adding Online Security or Tech Support free for 3 months) to deepen user retention.")
                else:
                    st.markdown("✨ *Account is Healthy:* No immediate discount interventions needed. Maintain standard service operations.")
                    
        except Exception as e:
            st.error(f"❌ Pipeline Execution Error: {e}")
