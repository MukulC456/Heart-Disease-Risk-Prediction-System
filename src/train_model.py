import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os

# 1. Setup absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "data", "processed_data.csv")
model_dir = os.path.join(base_dir, "..", "models")
model_path = os.path.join(model_dir, "heart_disease_model.pkl")

# Ensure the 'models' directory exists
os.makedirs(model_dir, exist_ok=True)

# 2. Load the processed data
try:
    df = pd.read_csv(data_path)
    print("Processed data loaded successfully!")

    X = df.drop("target", axis=1)
    y = df["target"]

    # 3. Split and Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 4. Save the trained model
    joblib.dump(model, model_path)
    print(f"Model is trained and saved to: {model_path}")

except FileNotFoundError:
    print(f"Error: Could not find {data_path}. Did you run data_preprocessing.py first?")