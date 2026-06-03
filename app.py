import streamlit as st
import pandas as pd
import joblib
import os

st.title("Loan Approval Prediction System")

@st.cache_resource
def load_or_train_model():
    try:
        model = joblib.load("loan_approval_model.pkl")
        # Validate the model with a dummy prediction to catch version mismatch errors
        dummy_df = pd.read_csv('loan_train.csv', nrows=1)
        dummy_df = dummy_df.drop(['Loan_ID', 'Loan_Status'], axis=1)
        model.predict(dummy_df)
        return model
    except Exception as e:

        # If loading fails (e.g. version mismatch), retrain using local data
        from sklearn.model_selection import train_test_split
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder, LabelEncoder
        from sklearn.ensemble import RandomForestClassifier

        df = pd.read_csv('loan_train.csv')
        df = df.drop('Loan_ID', axis=1)

        X = df.drop('Loan_Status', axis=1)
        y = LabelEncoder().fit_transform(df['Loan_Status'])

        cat = X.select_dtypes(include='object').columns
        num = X.select_dtypes(exclude='object').columns

        pre = ColumnTransformer([
            ('num', SimpleImputer(strategy='median'), num),
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='most_frequent')),
                ('oh', OneHotEncoder(handle_unknown='ignore'))
            ]), cat)
        ])

        model = Pipeline([
            ('pre', pre),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
        ])

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(Xtr, ytr)

        joblib.dump(model, 'loan_approval_model.pkl')
        return model

model = load_or_train_model()

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
