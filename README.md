
# Order Fallout Prediction (Decision Tree)

Supervised learning demo for Telecom OSS/BSS provisioning risk. Uses high‑level order attributes to predict whether an order will "fall out" into manual handling.

## Features
- Product_Type (Ethernet, DIA, Business Internet, Managed Router, OTT Bundle)
- On_Off_Net (On-Net, Off-Net)
- Site_Class (Single Site, Multi Site)
- Vendor_Required (Yes, No)
- Region (East, Central, West)
- Bundle_Count (1..5)

## Files
- `data_generate.py` — creates `order_fallout_prediction.csv`
- `train_decision_tree.py` — trains model, saves metrics + artifacts
- `evaluate_sample.py` — runs a single prediction on an example row
- `order_fallout_prediction.csv` — synthetic dataset
- Artifacts after training:
  - `model_decision_tree.pkl`
  - `encoders.pkl`
  - `feature_importances.csv`
  - `tree_plot.png`
  - `metrics_report.txt`

## Quickstart (VS Code / Windows PowerShell)

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# generate dataset
python data_generate.py

# train and evaluate
python train_decision_tree.py

# run a sample prediction
python evaluate_sample.py
```

## Notes
- The dataset is synthetic with clear patterns for demonstration purposes.
- In production, connect real order data and retrain regularly.
