
import streamlit as st
import pandas as pd
import joblib

st.title("Loan Approval Prediction System")

model = joblib.load("loan_approval_model.pkl")

gender = st.selectbox("Gender",["Male","Female"])
married = st.selectbox("Married",["Yes","No"])
dependents = st.selectbox("Dependents",["0","1","2","3+"])
education = st.selectbox("Education",["Graduate","Not Graduate"])
self_emp = st.selectbox("Self Employed",["Yes","No"])
income = st.number_input("Applicant Income",0)
coincome = st.number_input("Coapplicant Income",0)
loan_amount = st.number_input("Loan Amount",0)
term = st.number_input("Loan Term",360)
credit = st.selectbox("Credit History",[1.0,0.0])
area = st.selectbox("Property Area",["Urban","Rural","Semiurban"])

if st.button("Predict"):
    df = pd.DataFrame({
        "Gender":[gender],
        "Married":[married],
        "Dependents":[dependents],
        "Education":[education],
        "Self_Employed":[self_emp],
        "ApplicantIncome":[income],
        "CoapplicantIncome":[coincome],
        "LoanAmount":[loan_amount],
        "Loan_Amount_Term":[term],
        "Credit_History":[credit],
        "Property_Area":[area]
    })
    pred = model.predict(df)
    st.success("Loan Approved" if pred[0]==1 else "Loan Rejected")
