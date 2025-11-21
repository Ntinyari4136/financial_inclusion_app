import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("bank_model.pkl")
features = joblib.load("feature_names.pkl")

st.title("💰 Financial Inclusion Prediction App")
st.write("Predict whether an individual has a bank account based on demographic and household data.")

st.subheader("Enter Customer Details")

# Raw inputs from user
year = st.number_input("Year", value=2018)
household_size = st.number_input("Household size", value=1)
age = st.number_input("Age of respondent", value=25)
country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
location_type = st.selectbox("Location type", ["Rural", "Urban"])
cellphone_access = st.selectbox("Cellphone access", ["Yes", "No"])
gender = st.selectbox("Gender", ["Male", "Female"])
education = st.selectbox("Education level", [
    "No formal education", "Primary education", "Secondary education", 
    "Vocational/Specialised training"
])
job_type = st.selectbox("Job type", [
    "Self employed", "Formally employed Private", "Formally employed Government",
    "Government Dependent", "Informally employed", "Farming and Fishing",
    "No Income", "Other Income", "Remittance Dependent"
])

# Convert to dataframe for model
input_dict = {
    "year": year,
    "household_size": household_size,
    "age_of_respondent": age,
    # One-hot encoding manually
    "country_Kenya": int(country=="Kenya"),
    "country_Rwanda": int(country=="Rwanda"),
    "country_Tanzania": int(country=="Tanzania"),
    "country_Uganda": int(country=="Uganda"),
    "location_type_Urban": int(location_type=="Urban"),
    "cellphone_access_Yes": int(cellphone_access=="Yes"),
    "gender_of_respondent_Male": int(gender=="Male"),
    "education_level_No formal education": int(education=="No formal education"),
    "education_level_Primary education": int(education=="Primary education"),
    "education_level_Secondary education": int(education=="Secondary education"),
    "education_level_Vocational/Specialised training": int(education=="Vocational/Specialised training"),
    "job_type_Self employed": int(job_type=="Self employed"),
    "job_type_Formally employed Private": int(job_type=="Formally employed Private"),
    "job_type_Formally employed Government": int(job_type=="Formally employed Government"),
    "job_type_Government Dependent": int(job_type=="Government Dependent"),
    "job_type_Informally employed": int(job_type=="Informally employed"),
    "job_type_Farming and Fishing": int(job_type=="Farming and Fishing"),
    "job_type_No Income": int(job_type=="No Income"),
    "job_type_Other Income": int(job_type=="Other Income"),
    "job_type_Remittance Dependent": int(job_type=="Remittance Dependent")
}

# Fill missing features with 0 (in case your model has extra columns)
input_df = pd.DataFrame([input_dict])
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[features]  # Ensure column order matches model

# Prediction
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    result = "✔️ Has a Bank Account" if pred==1 else "❌ Does NOT Have a Bank Account"
    st.subheader("Prediction Result")
    st.success(result)
