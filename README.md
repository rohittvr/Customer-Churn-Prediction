# Customer Churn Prediction & Analytics System

A production-ready Machine Learning dashboard that predicts customer churn risk and provides actionable business insights through interactive analytics and executive reporting.

Built using **Streamlit, Scikit-learn, Plotly, and ReportLab**, this project demonstrates end-to-end ML deployment from model training to a business-facing dashboard.

---

## Live Demo

[Click here to open ](https://customer-churn-prediction-k9b4krn2hcfqrdk95mcbcq.streamlit.app/)


---

## Project Overview

Customer churn significantly impacts revenue and long-term business growth. Identifying customers at high risk of leaving enables companies to implement proactive retention strategies.

This system:

- Predicts churn probability
- Classifies customers as High or Low risk
- Supports bulk risk prediction via CSV upload
- Generates executive PDF reports
- Visualizes churn distribution and feature importance

---

## Machine Learning Model

**Algorithm:** Random Forest Classifier  
**Dataset:** Telco Customer Churn Dataset  
**Target Variable:** Churn (Yes / No)  
**Features Used:** 19 customer attributes  

### Data Preprocessing:
- Label Encoding for categorical variables
- Missing value handling
- Feature alignment validation
- Bulk file sanitization

---

## Application Features

###  1. Single Customer Prediction
- Interactive customer input form
- Churn probability calculation
- Risk gauge visualization
- Probability donut chart
- Downloadable executive PDF report

###  2. Bulk CSV Prediction
- Upload customer dataset
- Predict churn for multiple records
- Summary metrics:
  - Total Customers
  - Predicted Churn
  - Predicted Stay
  - Churn Rate (%)
- Distribution visualization
- Download results as CSV

###  3. Model Analytics
- Feature importance visualization
- Business insight interpretation

---

## Business Value

This system enables organizations to:

- Identify high-risk customers
- Reduce revenue loss
- Improve customer retention strategies
- Support data-driven decision-making
- Analyze churn patterns at scale

---

##  Technology Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- Plotly
- ReportLab
- Kaleido

---

##  Project Structure

```
customer-churn-dashboard/
│
├── churn_app.py              # Main Streamlit application
├── customer_churn_model.pkl  # Trained ML model
├── encoders.pkl              # Saved encoders for preprocessing
├── customerchurn2.png        # Logo / branding image
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## Run Locally

### Clone Repository

[Click here to view the repository](https://github.com/rohittvr/language-detector-and-translator.git)

### Install Dependencies

pip install -r requirements.txt


### Run Application

streamlit run churn_app.py


Open in browser:

http://localhost:8501


---

## Executive Reporting

The application generates a professional PDF report including:

- Risk classification
- Churn probability values
- Visual charts
- Business interpretation summary

---

## Future Improvements

- Confusion Matrix and ROC Curve dashboard
- SHAP-based model explainability
- Role-based authentication
- Database integration
- Automated reporting system
- Cloud deployment with custom domain

---

## Author

Rajput Rohit  
Artificial Intelligence & Machine Learning  

---

## ⭐ If You Found This Useful

Consider giving the repository a star.
