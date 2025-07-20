import joblib
import numpy as np
import pandas as pd

def predict_rent(input_data: pd.DataFrame):
    pipeline = joblib.load("models/preprocessing_pipeline.pkl")
    model = joblib.load("models/best_model.pkl")
    
    processed_data = pipeline.transform(input_data)

    predictions = model.predict(processed_data)
    return predictions

if __name__ == "__main__":
    sample = pd.DataFrame([{
    'id': 9999,
    'house_type': '2 BHK Apartment',
    'house_size': "1200 sq ft",
    'location': 'Kothrud',
    'city': 'Pune',
    'latitude': 18.5204,
    'longitude': 73.8567,
    'numBathrooms': 2,
    'SecurityDeposit': 4000,
    'furnishing': 'Semi-Furnished'
}])

    prediction = predict_rent(sample)
    print(f"🏠 Predicted Rent: ₹{prediction[0]:,.2f}")
