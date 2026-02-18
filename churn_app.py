import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
import io
import tempfile

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER


# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Churn Analytics System", layout="wide")


# -------------------------------------------------------
# PREMIUM GRADIENT THEME
# -------------------------------------------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #111827);
}
.big-title {font-size: 36px; font-weight: 700; color: #f8fafc;}
.subtitle {color: #94a3b8;}
.prediction-card {
    padding: 25px;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    font-size: 20px;
    font-weight: 600;
}
.churn {border-left: 6px solid #ef4444;}
.no-churn {border-left: 6px solid #22c55e;}
.stButton>button {
    width: 100%;
    height: 45px;
    border-radius: 10px;
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    font-weight: 600;
    border: none;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
@st.cache_resource
def load_model():
    with open("customer_churn_model.pkl", "rb") as f:
        model_data = pickle.load(f)
    with open("encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return model_data["model"], encoders, model_data["features_names"]

model, encoders, feature_names = load_model()


# -------------------------------------------------------
# PREPROCESS FUNCTION
# -------------------------------------------------------
def preprocess_data(df):
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    df.fillna(0, inplace=True)

    for col in encoders:
        if col in df.columns:
            df[col] = encoders[col].transform(df[col])

    return df


# -------------------------------------------------------
# PDF GENERATION
# -------------------------------------------------------
def generate_pdf(prediction, proba, fig, gauge):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        name='CenterTitle',
        parent=styles['Heading1'],
        alignment=TA_CENTER
    )

    elements.append(Paragraph("Corporate Churn Risk Report", title_style))
    elements.append(Spacer(1, 0.3 * inch))

    risk_level = "High" if prediction == 1 else "Low"

    summary_data = [
        ["Customer Risk Level", risk_level],
        ["Churn Probability", f"{proba[1]*100:.2f}%"],
        ["Retention Probability", f"{proba[0]*100:.2f}%"]
    ]

    table = Table(summary_data, colWidths=[220, 220])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.4 * inch))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp1:
        fig.write_image(tmp1.name)
        elements.append(Image(tmp1.name, width=400, height=300))

    elements.append(Spacer(1, 0.4 * inch))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
        gauge.write_image(tmp2.name)
        elements.append(Image(tmp2.name, width=400, height=300))

    doc.build(elements)
    return buffer.getvalue()


# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
st.sidebar.image("customerchurn2.png", width=120)
st.sidebar.markdown("### Customer Churn Prediction")
st.sidebar.caption("AI Risk Prediction System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Single Prediction", "Bulk Prediction", "Analytics"]
)


# =======================================================
# SINGLE PREDICTION
# =======================================================
if page == "Single Prediction":

    st.markdown('<div class="big-title">Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Random Forest Model</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        tenure = st.slider("Tenure", 0, 72, 12)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        monthly_charges = st.number_input("Monthly Charges", 0.0, 150.0, 50.0)

    with col2:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)
        partner = st.selectbox("Partner", ["Yes", "No"])

    if st.button("Analyze Risk"):

        input_dict = {
            'gender': gender,
            'SeniorCitizen': 0,
            'Partner': partner,
            'Dependents': "No",
            'tenure': tenure,
            'PhoneService': "Yes",
            'MultipleLines': "No",
            'InternetService': internet_service,
            'OnlineSecurity': "No",
            'OnlineBackup': "No",
            'DeviceProtection': "No",
            'TechSupport': "No",
            'StreamingTV': "No",
            'StreamingMovies': "No",
            'Contract': contract,
            'PaperlessBilling': "Yes",
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }

        df = pd.DataFrame([input_dict])
        df = preprocess_data(df)
        df = df[feature_names]

        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0]

        if prediction == 1:
            st.markdown('<div class="prediction-card churn">High Risk of Churn</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="prediction-card no-churn">Customer Likely to Stay</div>', unsafe_allow_html=True)

        colA, colB = st.columns(2)

        with colA:
            fig = go.Figure(data=[go.Pie(
                labels=["Stay", "Churn"],
                values=[proba[0], proba[1]],
                hole=0.6
            )])
            st.plotly_chart(fig, use_container_width=True)

        with colB:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba[1]*100,
                title={'text': "Churn Risk (%)"},
                gauge={'axis': {'range': [0, 100]}}
            ))
            st.plotly_chart(gauge, use_container_width=True)

        pdf_data = generate_pdf(prediction, proba, fig, gauge)

        st.download_button(
            "Download Executive PDF Report",
            pdf_data,
            "corporate_churn_report.pdf",
            "application/pdf"
        )


# =======================================================
# BULK PREDICTION
# =======================================================
elif page == "Bulk Prediction":

    st.title("Bulk Customer Prediction")

    file = st.file_uploader("Upload CSV File", type=["csv"])

    if file:
        data = pd.read_csv(file)
        data = preprocess_data(data)

        missing_cols = [col for col in feature_names if col not in data.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}")
            st.stop()

        data = data[feature_names]

        preds = model.predict(data)
        data["Prediction"] = preds

        total_customers = len(data)
        total_churn = (data["Prediction"] == 1).sum()
        total_stay = (data["Prediction"] == 0).sum()
        churn_rate = (total_churn / total_customers) * 100

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", total_customers)
        col2.metric("Predicted Churn", total_churn)
        col3.metric("Predicted Stay", total_stay)
        col4.metric("Churn Rate (%)", f"{churn_rate:.2f}%")

        summary_df = pd.DataFrame({
            "Category": ["Stay", "Churn"],
            "Count": [total_stay, total_churn]
        })

        fig = px.pie(summary_df, values="Count", names="Category", hole=0.6)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(data.head())

        csv = data.to_csv(index=False).encode("utf-8")
        st.download_button("Download Results", csv, "bulk_predictions.csv", "text/csv")


# =======================================================
# ANALYTICS
# =======================================================
else:

    st.title("Model Analytics")

    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        fig = px.bar(
            importance_df.head(10),
            x="Importance",
            y="Feature",
            orientation="h"
        )

        st.plotly_chart(fig, use_container_width=True)
