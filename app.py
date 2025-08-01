import streamlit as st
import joblib
import pandas as pd
import re

# Load preprocessing pipeline and best model
pipeline = joblib.load("models/preprocessing_pipeline.pkl")
model = joblib.load("models/best_model.pkl")  # Make sure best_model.pkl is saved after training

# Streamlit app configuration
st.set_page_config(page_title="Rental Price Predictor", layout="centered")
st.title("🏡 Rental Price Recommendation System")
st.markdown("Enter the property details to predict the monthly rent:")

# Utility function to clean currency/sqft inputs
def clean_numeric(value):
    if isinstance(value, str):
        value = re.sub(r"[^\d.]", "", value)
        value = value.replace(",", "")
    try:
        return float(value)
    except:
        return None

# Input Fields
house_format = st.selectbox("Select House Format", [
    "1 RK", "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK",
    "6 BHK", "7 BHK", "8 BHK", "9 BHK", "10 BHK", "12 BHK"
])

house_type = st.selectbox("Select House Type", [
    "Studio Apartment", "Independent Floor", "Independent House",
    "Apartment", "Villa", "Penthouse"
])

house_size = st.text_input("House Size (e.g., 1100 sqft)")
location = st.text_input("Location (Area/Colony)")
city = st.selectbox("City", ["Delhi", "Mumbai", "Pune"])
numBathrooms = st.number_input("Number of Bathrooms", step=1, min_value=1)
security_deposit = st.text_input("Security Deposit (e.g., ₹25000)")
Status = st.selectbox("Status", ["Furnished", "Semi-Furnished", "Unfurnished"])

# Predict Button
if st.button("Predict Rent"):
    try:
        # Prepare input DataFrame
        input_df = pd.DataFrame([{
            "house_format": house_format,
            "house_type": house_type,
            "house_size": clean_numeric(house_size),
            "location": location,
            "city": city,
            "numBathrooms": int(numBathrooms),
            "SecurityDeposit": clean_numeric(security_deposit),
            "Status": Status
        }])

        # Transform and predict
        transformed = pipeline.transform(input_df)
        prediction = model.predict(transformed)[0]

        st.success(f"💰 Estimated Monthly Rent: ₹{round(prediction):,}")
    
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
