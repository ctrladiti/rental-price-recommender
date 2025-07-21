import streamlit as st
import joblib
import pandas as pd
from src.preprocessing_sql import preprocess_input_data  # Preprocess function
from src.db_connect import load_data_from_sql  # Optional: to get schema/column names

# Load model
model = joblib.load("models/best_model.pkl")

# Streamlit UI
st.set_page_config(page_title="Rental Price Recommender", layout="centered")
st.title("🏠 Rental Price Recommender")
st.markdown("Enter the details below to estimate the rental price:")

# Input fields
house_type = st.selectbox("Property Type", [
    "Studio Apartment", "Independent Floor", "Independent House", "Apartment", "Villa", "Penthouse"
])

house_format = st.selectbox("House Format (e.g., BHK)", [
    "1 RK", "1 BHK", "2 BHK", "3 BHK", "4 BHK", "5 BHK",
    "6 BHK", "7 BHK", "8 BHK", "9 BHK", "10 BHK", "12 BHK"
])

status = st.selectbox("Furnishing Status", ["Furnished", "Semi-Furnished", "Unfurnished"])

house_size = st.number_input("House Size (in sqft)", min_value=100, max_value=5000, step=50)
location = st.text_input("Location")
city = st.selectbox("City", ["Delhi", "Mumbai", "Pune"])
num_bathrooms = st.selectbox("Number of Bathrooms", [1, 2, 3, 4])
security_deposit = st.number_input("Security Deposit (₹)", min_value=0)

# Create DataFrame
input_data = pd.DataFrame([{
    "house_type": house_type,
    "house_format": house_format,
    "Status": status,
    "house_size": house_size,
    "location": location,
    "city": city,
    "numBathrooms": num_bathrooms,
    "SecurityDeposit": security_deposit
}])

# Prediction
if st.button("Predict Rent"):
    try:
        X_preprocessed, _ = preprocess_input_data(input_data)
        prediction = model.predict(X_preprocessed)[0]
        st.success(f"Estimated Rent: ₹{round(prediction):,}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
