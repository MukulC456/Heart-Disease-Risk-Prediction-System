import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)

# 1. Setup absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "data", "processed_data.csv")
model_path = os.path.join(base_dir, "..", "models", "heart_disease_model.pkl")

try:
    # 2. Load the processed data
    df = pd.read_csv(data_path)
    X = df.drop("target", axis=1)
    y = df["target"]

    # 3. Load the saved model
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        y_pred = model.predict(X)

        # 4. Calculate the 4 Key Metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred) # Adding F1 as it balances precision and recall
        cm = confusion_matrix(y, y_pred)

        # 5. Print Metrics Summary
        print("\n" + "="*40)
        print("      HEART DISEASE MODEL EVALUATION")
        print("="*40)
        print(f"1. Accuracy:         {accuracy:.4f}")
        print(f"2. Precision:        {precision:.4f}")
        print(f"3. Recall:           {recall:.4f}")
        print(f"4. F1-Score:         {f1:.4f}")
        print("-" * 40)
        print("Confusion Matrix (Raw Counts):")
        print(cm)
        print("="*40 + "\n")

        # 6. Generate and Save Visual Confusion Matrix
        fig, ax = plt.subplots(figsize=(7, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Healthy", "Heart Disease"])
        disp.plot(cmap=plt.cm.RdPu, ax=ax) # Using a distinct color map
        
        plt.title("Confusion Matrix: Heart Disease Prediction")
        plot_path = os.path.join(base_dir, "confusion_matrix_visual.png")
        plt.savefig(plot_path)
        plt.close() # Close plot to free up memory
        
        print(f"Visual report saved to: {plot_path}")
        
    else:
        print(f"Error: Model not found at {model_path}. Please run train_model.py first!")

except FileNotFoundError:
    print(f"Error: Could not find processed data at {data_path}")
except Exception as e:
    print(f"An error occurred: {e}")