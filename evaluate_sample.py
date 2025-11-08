
import joblib
import numpy as np
import pandas as pd

# Example input (high-level attributes)
example = {
    "Product_Type": "DIA",
    "On_Off_Net": "Off-Net",
    "Site_Class": "Multi Site",
    "Vendor_Required": "Yes",
    "Region": "East",
    "Bundle_Count": 4
}

def encode_row(d, encoders):
    row = d.copy()
    for col, le in encoders.items():
        row[col] = le.transform([row[col]])[0]
    return row

def main():
    model = joblib.load("model_decision_tree.pkl")
    encoders = joblib.load("encoders.pkl")

    row = encode_row(example, encoders)
    X = pd.DataFrame([row])
    proba = model.predict_proba(X)[0][1]
    pred = model.predict(X)[0]

    print("Input:", example)
    print(f"Predicted Fallout class: {int(pred)} (1=Likely Fallout)")
    print(f"Fallout Probability: {proba:.3f}")

if __name__ == "__main__":
    main()
