"""
ConnectTel Customer Churn Prediction Dashboard
------------------------------------------------
A professional Streamlit dashboard that predicts telecom customer churn
risk using a trained XGBoost model.

Required files (same folder as this script):
    - app.py
    - best_xgboost_model.pkl
    - columns.pkl
    - requirements.txt

Run locally:
    streamlit run app.py

Deploy on Streamlit Community Cloud:
    Push this folder to a public GitHub repo, then deploy from
    https://share.streamlit.io pointing at app.py.
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path

# Resolve paths relative to this script's own folder — not the process's
# working directory. Streamlit Community Cloud runs the app with the repo
# root as cwd, so a bare "best_xgboost_model.pkl" fails if app.py lives in
# a subfolder. This makes it work regardless of where the script sits.
BASE_DIR = Path(__file__).resolve().parent

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Telecom Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =====================================================================
# CONSTANTS
# =====================================================================
HIGH_RISK_THRESHOLD = 0.75
MEDIUM_RISK_THRESHOLD = 0.50

# TODO: replace with the actual evaluation numbers from your notebook
MODEL_METRICS = {
    "Accuracy": "80.2%",
    "ROC-AUC Score": "0.84",
    "Algorithm": "XGBoost",
}

RISK_STYLES = {
    "High":   {"color": "#DC2626", "bg": "#FEE2E2", "icon": "🔴", "class": "risk-high"},
    "Medium": {"color": "#D97706", "bg": "#FEF3C7", "icon": "🟠", "class": "risk-medium"},
    "Low":    {"color": "#16A34A", "bg": "#DCFCE7", "icon": "🟢", "class": "risk-low"},
}


# =====================================================================
# STYLING
# =====================================================================
def inject_custom_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        .main-header {
            background: linear-gradient(135deg, #0F2C59 0%, #1E4C9A 55%, #2E7FE0 100%);
            padding: 2.2rem 2rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 1.6rem;
            box-shadow: 0 8px 24px rgba(15, 44, 89, 0.25);
        }
        .main-header h1 {
            margin: 0;
            font-size: 2.1rem;
            font-weight: 800;
        }
        .main-header p {
            margin: 0.4rem 0 0 0;
            opacity: 0.9;
            font-size: 1.02rem;
        }

        .section-title {
            font-weight: 700;
            font-size: 1.15rem;
            color: #0F2C59;
            margin: 1.2rem 0 0.6rem 0;
        }

        .card {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            border: 1px solid #EEF1F6;
            height: 100%;
        }

        .risk-badge {
            display: inline-block;
            padding: 0.55rem 1.4rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 1.05rem;
            border: 2px solid;
        }
        .risk-high   { background: #FEE2E2; color: #DC2626; border-color: #DC2626; }
        .risk-medium { background: #FEF3C7; color: #D97706; border-color: #D97706; }
        .risk-low    { background: #DCFCE7; color: #16A34A; border-color: #16A34A; }

        .gauge-container {
            width: 100%;
            background: #E5E7EB;
            border-radius: 999px;
            height: 26px;
            overflow: hidden;
            margin: 0.4rem 0 0.3rem 0;
        }
        .gauge-fill {
            height: 100%;
            border-radius: 999px;
            text-align: right;
            transition: width 0.6s ease;
        }

        .app-footer {
            text-align: center;
            padding: 1.4rem 0 0.6rem 0;
            margin-top: 2.2rem;
            border-top: 1px solid #E5E7EB;
            color: #6B7280;
            font-size: 0.85rem;
        }

        section[data-testid="stSidebar"] {
            background: #F7F9FC;
            border-right: 1px solid #E5E7EB;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# MODEL / ARTIFACT LOADING
# =====================================================================
@st.cache_resource(show_spinner="Loading prediction model...")
def load_artifacts():
    """Load the trained model and the training column schema."""
    try:
        model = joblib.load(BASE_DIR / "best_xgboost_model.pkl")
        columns = joblib.load(BASE_DIR / "columns.pkl")
        return model, columns, None
    except FileNotFoundError as e:
        return None, None, f"Missing file: {e.filename}"
    except Exception as e:
        return None, None, f"Failed to load model artifacts: {e}"
# =====================================================================
# HEADER
# =====================================================================
def render_header():
    st.markdown(
        """
        <div class="main-header">
            <h1>📊 Telecom Customer Churn Prediction Dashboard</h1>
            <p>AI-powered early warning system for customer retention &middot; Developed by Abhishek Panchal</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# SIDEBAR — CUSTOMER INPUTS
# =====================================================================
def render_sidebar():
    st.sidebar.markdown("## 👤 Customer Information")
    st.sidebar.caption("Fill in the customer's profile, then run the prediction.")

    inputs = {}

    with st.sidebar.expander("👤 Demographics", expanded=True):
        inputs["gender"] = st.selectbox("Gender", ["Female", "Male"])
        inputs["senior"] = st.selectbox("Senior Citizen", ["No", "Yes"])
        inputs["partner"] = st.selectbox("Partner", ["No", "Yes"])
        inputs["dependents"] = st.selectbox("Dependents", ["No", "Yes"])
        inputs["tenure"] = st.slider("Tenure (Months)", 0, 72, 12)

    with st.sidebar.expander("📞 Phone & Internet", expanded=False):
        inputs["phone_service"] = st.selectbox("Phone Service", ["No", "Yes"])
        inputs["multiple_lines"] = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        inputs["internet_service"] = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )

    with st.sidebar.expander("🛡️ Add-on Services", expanded=False):
        inputs["online_security"] = st.selectbox(
            "Online Security", ["No", "Yes", "No internet service"]
        )
        inputs["online_backup"] = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"]
        )
        inputs["device_protection"] = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"]
        )
        inputs["tech_support"] = st.selectbox(
            "Tech Support", ["No", "Yes", "No internet service"]
        )
        inputs["streaming_tv"] = st.selectbox(
            "Streaming TV", ["No", "Yes", "No internet service"]
        )
        inputs["streaming_movies"] = st.selectbox(
            "Streaming Movies", ["No", "Yes", "No internet service"]
        )

    with st.sidebar.expander("💳 Billing & Contract", expanded=False):
        inputs["contract"] = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        inputs["paperless"] = st.selectbox("Paperless Billing", ["No", "Yes"])
        inputs["payment"] = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )
        inputs["monthly_charges"] = st.number_input(
            "Monthly Charges ($)", min_value=0.0, value=70.0, step=1.0
        )
        inputs["total_charges"] = st.number_input(
            "Total Charges ($)", min_value=0.0, value=1000.0, step=10.0
        )

    st.sidebar.markdown("---")
    predict_clicked = st.sidebar.button(
        "🔍 Predict Customer Churn", width="stretch", type="primary"
    )

    show_debug = st.sidebar.checkbox(
        "🔧 Show technical feature vector (for developers)", value=False
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📈 Model Performance")
    m1, m2 = st.sidebar.columns(2)
    m1.metric("Accuracy", MODEL_METRICS["Accuracy"])
    m2.metric("ROC-AUC", MODEL_METRICS["ROC-AUC Score"])
    st.sidebar.caption(f"Algorithm: {MODEL_METRICS['Algorithm']}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## ℹ️ About This Project")
    st.sidebar.info(
        """
**Predictive Customer Churn Model**

**Algorithm:** XGBoost

**Dataset:** IBM Telco Customer Churn

**Developed by:** Abhishek Panchal
        """
    )

    return inputs, predict_clicked, show_debug


# =====================================================================
# FEATURE ENGINEERING + ONE-HOT ENCODING
# =====================================================================
def build_feature_frame(inputs, columns):
    input_data = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    # Numerical features
    input_data["tenure"] = inputs["tenure"]
    input_data["MonthlyCharges"] = inputs["monthly_charges"]
    input_data["TotalCharges"] = inputs["total_charges"]

    # Engineered features
    if inputs["tenure"] == 0:
        input_data["TotalChargesPerTenure"] = 0
    else:
        input_data["TotalChargesPerTenure"] = (
            inputs["total_charges"] / inputs["tenure"]
        )

    services = [
        inputs["phone_service"],
        inputs["multiple_lines"],
        inputs["online_security"],
        inputs["online_backup"],
        inputs["device_protection"],
        inputs["tech_support"],
        inputs["streaming_tv"],
        inputs["streaming_movies"],
    ]
    input_data["ServiceCount"] = sum(1 for s in services if s == "Yes")
    input_data["LongTermCustomer"] = 1 if inputs["tenure"] >= 24 else 0

    # One-hot encoding
    def activate(column_name):
        if column_name in input_data.columns:
            input_data[column_name] = 1

    activate(f"gender_{inputs['gender']}")
    activate(f"SeniorCitizen_{inputs['senior']}")
    activate(f"Partner_{inputs['partner']}")
    activate(f"Dependents_{inputs['dependents']}")

    activate(f"PhoneService_{inputs['phone_service']}")
    activate(f"MultipleLines_{inputs['multiple_lines']}")
    activate(f"InternetService_{inputs['internet_service']}")

    activate(f"OnlineSecurity_{inputs['online_security']}")
    activate(f"OnlineBackup_{inputs['online_backup']}")
    activate(f"DeviceProtection_{inputs['device_protection']}")
    activate(f"TechSupport_{inputs['tech_support']}")

    activate(f"StreamingTV_{inputs['streaming_tv']}")
    activate(f"StreamingMovies_{inputs['streaming_movies']}")

    activate(f"Contract_{inputs['contract']}")
    activate(f"PaperlessBilling_{inputs['paperless']}")
    activate(f"PaymentMethod_{inputs['payment']}")

    return input_data


# =====================================================================
# PREDICTION
# =====================================================================
def run_prediction(model, input_data):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    return prediction, probability


def get_risk_tier(probability):
    if probability >= HIGH_RISK_THRESHOLD:
        return "High"
    elif probability >= MEDIUM_RISK_THRESHOLD:
        return "Medium"
    return "Low"


# =====================================================================
# RESULTS DASHBOARD
# =====================================================================
def render_results(prediction, probability, inputs, input_data, model, show_debug):
    st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)

    tier = get_risk_tier(probability)
    style = RISK_STYLES[tier]
    confidence = max(probability, 1 - probability) * 100

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        verdict = "Likely to Churn" if prediction == 1 else "Likely to Stay"
        st.markdown(f"#### {'⚠️' if prediction == 1 else '✅'} {verdict}")
        st.markdown(
            f'<span class="risk-badge {style["class"]}">{style["icon"]} {tier} Risk</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='margin-top:0.8rem; color:#6B7280;'>Model confidence: "
            f"<b>{confidence:.1f}%</b></p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Churn Probability**")
        pct = probability * 100
        st.markdown(
            f"""
            <div class="gauge-container">
                <div class="gauge-fill" style="width:{pct:.1f}%; background:{style['color']};"></div>
            </div>
            <p style="font-weight:700; font-size:1.3rem; color:{style['color']}; margin:0;">
                {pct:.2f}%
            </p>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">🧾 Customer Summary</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Tenure", f"{inputs['tenure']} mo")
    s2.metric("Monthly Charges", f"${inputs['monthly_charges']:.2f}")
    s3.metric("Contract", inputs["contract"])
    s4.metric("Internet", inputs["internet_service"])

    st.markdown('<div class="section-title">💡 Recommended Action</div>', unsafe_allow_html=True)
    if tier == "High":
        st.error(
            "**High Risk — Immediate Action Recommended**\n\n"
            "- Offer a loyalty discount or retention incentive\n"
            "- Have a retention specialist contact the customer directly\n"
            "- Provide complimentary premium technical support"
        )
    elif tier == "Medium":
        st.warning(
            "**Medium Risk — Proactive Engagement Recommended**\n\n"
            "- Send targeted promotional offers\n"
            "- Recommend upgrading to a longer-term contract\n"
            "- Highlight underused services/add-ons"
        )
    else:
        st.success(
            "**Low Risk — Customer Is Stable**\n\n"
            "- No immediate action required\n"
            "- Continue delivering consistent service quality\n"
            "- Consider periodic satisfaction check-ins"
        )

    render_shap_section(model, input_data)

    st.markdown('<div class="section-title">📄 Export</div>', unsafe_allow_html=True)
    render_download_section(prediction, probability, inputs)

    if show_debug:
        with st.expander("🔧 View Engineered Feature Vector (Technical / Advanced)", expanded=True):
            st.caption(
                "This is the exact one-hot-encoded feature row passed into the "
                "XGBoost model, useful for debugging or demonstrating the pipeline."
            )
            st.dataframe(input_data.T.rename(columns={0: "value"}), width="stretch")


# =====================================================================
# SHAP EXPLANATION (per-prediction feature contributions)
# =====================================================================
@st.cache_resource(show_spinner=False)
def get_shap_explainer(_model):
    import shap
    return shap.TreeExplainer(_model)


def render_shap_section(model, input_data):
    st.markdown(
        '<div class="section-title">🔍 Top Features Influencing This Prediction</div>',
        unsafe_allow_html=True,
    )
    try:
        import shap  # noqa: F401  (raises a clean ImportError if not installed)

        explainer = get_shap_explainer(model)
        shap_values = explainer.shap_values(input_data)

        # Some SHAP/XGBoost combinations return a list of per-class arrays,
        # others return a single array for the positive class directly.
        if isinstance(shap_values, list):
            values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            values = shap_values[0]

        contrib = pd.Series(values, index=input_data.columns)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index[:6])
        max_abs = top.abs().max() or 1

        for feature, value in top.items():
            color = "#DC2626" if value > 0 else "#16A34A"
            width = abs(value) / max_abs * 100
            direction = "↑ increases risk" if value > 0 else "↓ decreases risk"
            st.markdown(
                f"""
                <div style="margin-bottom:0.55rem;">
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#374151;">
                        <span>{feature}</span><span style="color:{color}; font-weight:600;">{direction}</span>
                    </div>
                    <div class="gauge-container" style="height:12px;">
                        <div class="gauge-fill" style="width:{width:.1f}%; background:{color};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    except ImportError:
        st.info(
            "Install the `shap` package (`pip install shap`, already listed in "
            "requirements.txt) to see per-prediction feature explanations here."
        )
    except Exception as e:
        st.info(f"Feature explanation unavailable for this prediction ({e}).")


# =====================================================================
# DOWNLOAD PREDICTION SUMMARY
# =====================================================================
def render_download_section(prediction, probability, inputs):
    summary = dict(inputs)
    summary["Prediction"] = "Churn" if prediction == 1 else "No Churn"
    summary["Churn Probability (%)"] = round(probability * 100, 2)
    summary["Risk Tier"] = get_risk_tier(probability)

    summary_df = pd.DataFrame([summary])
    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📄 Download Prediction Summary (CSV)",
        data=csv_bytes,
        file_name="churn_prediction_summary.csv",
        mime="text/csv",
        width="stretch",
    )


# =====================================================================
# FOOTER
# =====================================================================
def render_footer():
    st.markdown(
        """
        <div class="app-footer">
            Developed by Abhishek Panchal<br>
            Diploma in Computer Engineering &amp; IoT<br>
            Machine Learning Project using XGBoost
        </div>
        """,
        unsafe_allow_html=True,
    )


# =====================================================================
# MAIN
# =====================================================================
def main():
    inject_custom_css()
    render_header()

    model, columns, load_error = load_artifacts()
    if load_error:
        st.error(
            f"⚠️ {load_error}\n\n"
            "Make sure `best_xgboost_model.pkl` and `columns.pkl` are in the "
            "same folder as this script."
        )
        st.stop()

    inputs, predict_clicked, show_debug = render_sidebar()

    if predict_clicked:
        try:
            input_data = build_feature_frame(inputs, columns)
            prediction, probability = run_prediction(model, input_data)
            st.session_state["result"] = {
                "prediction": prediction,
                "probability": probability,
                "inputs": inputs,
                "input_data": input_data,
            }
        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
            st.stop()

    if "result" in st.session_state:
        r = st.session_state["result"]
        render_results(
            r["prediction"], r["probability"], r["inputs"], r["input_data"],
            model, show_debug,
        )
    else:
        st.info(
            "👈 Fill in the customer's details in the sidebar, then click "
            "**🔍 Predict Customer Churn** to see the risk assessment."
        )

    render_footer()


if __name__ == "__main__":
    main()
