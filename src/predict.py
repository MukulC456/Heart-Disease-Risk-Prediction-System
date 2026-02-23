import joblib
import numpy as np

# Load trained model
MODEL_PATH = "models/heart_disease_model.pkl"
model = joblib.load(MODEL_PATH)

def predict_heart_disease(
    age,
    sex,
    resting_bp,
    cholesterol,
    fasting_blood_sugar,
    max_heart_rate,
    chest_pain_type,
    exercise_induced_angina,
    resting_ecg,
    st_depression,
    slope,
    num_major_vessels
):
    """
    Predict heart disease risk using trained ML model.
    """

    input_data = np.array([[
        age,
        sex,
        resting_bp,
        cholesterol,
        fasting_blood_sugar,
        max_heart_rate,
        chest_pain_type,
        exercise_induced_angina,
        resting_ecg,
        st_depression,
        slope,
        num_major_vessels
    ]])

    prediction = model.predict(input_data)

    return "High Risk" if prediction[0] == 1 else "Low Risk"


if __name__ == "__main__":
    print("Heart Disease Risk Prediction (Offline Test)")
    print("------------------------------------------")

    # Example test values (from medical tests)
    result = predict_heart_disease(
        age=55,
        sex=1,
        resting_bp=145,
        cholesterol=260,
        fasting_blood_sugar=1,
        max_heart_rate=150,
        chest_pain_type=2,
        exercise_induced_angina=1,
        resting_ecg=1,
        st_depression=2.3,
        slope=1,
        num_major_vessels=2
    )

    print(f"Prediction Result: {result}")