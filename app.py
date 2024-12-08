import streamlit as st
import pickle
import pandas as pd

# Load the pre-trained model
with open("rf_obesity.pkl", "rb") as model_file:
    obesity_model = pickle.load(model_file)

# Streamlit app title and description
st.title("Obesity Prediction App")
st.subheader("Provide your details to estimate your obesity level.")

# User inputs
user_age = st.number_input("Enter your age (years):", min_value=1, max_value=120, value=25)

# Height input
feet = st.number_input("Height - Feet:", min_value=1, max_value=8, value=5)
inches = st.number_input("Height - Inches:", min_value=0, max_value=11, value=7)
height_in_meters = ((feet * 12) + inches) * 0.0254  # Convert height to meters

# Weight input
weight_lbs = st.number_input("Weight (lbs):", min_value=20, max_value=500, value=150)
weight_in_kg = weight_lbs * 0.453592  # Convert to kilograms

# Lifestyle and habits
family_history = st.radio("Family History of Obesity:", ["Yes", "No"])
calorie_food = st.radio("Do you frequently consume high-calorie food?", ["Yes", "No"])
veggie_frequency = st.slider("Vegetable consumption frequency (1=Low, 3=High):", 1, 3, 2)
meals_per_day = st.slider("Number of meals per day:", 1, 4, 3)
snack_habits = st.radio("Eating snacks between meals:", ["Always", "Frequently", "Sometimes", "Never"])
smoking = st.radio("Do you smoke?", ["Yes", "No"])
water_intake = st.slider("Daily water consumption (liters):", 1.0, 3.0, 2.0, step=0.1)
calorie_monitor = st.radio("Do you monitor calorie intake?", ["Yes", "No"])
activity_days = st.slider("Physical activity (days/week):", 0.0, 7.0, 3.0, step=0.5)
tech_time = st.slider("Time spent using technology (hours/day):", 0.0, 24.0, 5.0, step=0.5)
alcohol_use = st.radio("Frequency of alcohol consumption:", ["No", "Sometimes", "Frequently"])
transport_mode = st.radio(
    "Primary mode of transportation:",
    ["Walking", "Public Transport", "Car", "Bicycle", "Motorbike"]
)

# Data preparation for the model
features = {
    "Age": [user_age],
    "Height": [height_in_meters],
    "Weight": [weight_in_kg],
    "FH_yes": [1 if family_history == "Yes" else 0],
    "FAVC_yes": [1 if calorie_food == "Yes" else 0],
    "FCVC": [veggie_frequency],
    "NCP": [meals_per_day],
    "CAEC_Always": [1 if snack_habits == "Always" else 0],
    "CAEC_Frequently": [1 if snack_habits == "Frequently" else 0],
    "CAEC_Sometimes": [1 if snack_habits == "Sometimes" else 0],
    "SMOKE_yes": [1 if smoking == "Yes" else 0],
    "CH2O": [water_intake],
    "SCC_yes": [1 if calorie_monitor == "Yes" else 0],
    "FAF": [activity_days],
    "TUE": [tech_time],
    "CALC_no": [1 if alcohol_use == "No" else 0],
    "CALC_sometimes": [1 if alcohol_use == "Sometimes" else 0],
    "CALC_frequently": [1 if alcohol_use == "Frequently" else 0],
    "MTRANS_Walking": [1 if transport_mode == "Walking" else 0],
    "MTRANS_Public Transport": [1 if transport_mode == "Public Transport" else 0],
    "MTRANS_Car": [1 if transport_mode == "Car" else 0],
    "MTRANS_Bicycle": [1 if transport_mode == "Bicycle" else 0],
    "MTRANS_Motorbike": [1 if transport_mode == "Motorbike" else 0],
}

# Convert to DataFrame
user_data = pd.DataFrame(features)

# Ensure all required features are in the input data
for feature in obesity_model.feature_names_in_:
    if feature not in user_data.columns:
        user_data[feature] = 0

# Prediction button and result display
if st.button("Predict Obesity Level"):
    user_data = user_data[obesity_model.feature_names_in_]
    prediction = obesity_model.predict(user_data)
    prediction_label = [
        "Normal Weight",
        "Obesity Type I",
        "Obesity Type II",
        "Obesity Type III",
        "Overweight Level I",
        "Overweight Level II",
    ][prediction[0].argmax()]
    st.success(f"Predicted Obesity Level: {prediction_label}")
