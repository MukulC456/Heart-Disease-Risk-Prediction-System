import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# 1. Get the directory where THIS script is saved (the 'src' folder)
base_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the project root, then into the 'data' folder
# This matches your folder structure: Heart Disease Risk Prediction System/data/raw_data.csv
data_path = os.path.join(base_dir, "..", "data", "raw_data.csv")
output_path = os.path.join(base_dir, "..", "data", "processed_data.csv")

# Load raw data
try:
    df = pd.read_csv(data_path)
    print("File loaded successfully!")
    
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df["target"] = y.values

    # Save to the data/processed folder
    processed_df.to_csv(output_path, index=False)
    print(f"Processed data saved successfully to: {output_path}")

except FileNotFoundError:
    print(f"Error: Could not find the file at {data_path}")
    print("Check if 'raw_data.csv' exists inside your 'data' folder.")