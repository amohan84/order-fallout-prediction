
import numpy as np
import pandas as pd

def generate_dataset(num_rows=300, seed=42, csv_path="order_fallout_prediction.csv"):
    rng = np.random.default_rng(seed)
    product_types = ["Ethernet", "DIA", "Business Internet", "Managed Router", "OTT Bundle"]
    on_off_net = ["On-Net", "Off-Net"]
    site_class = ["Single Site", "Multi Site"]
    vendor_required = ["Yes", "No"]
    regions = ["East", "Central", "West"]

    df = pd.DataFrame({
        "Product_Type": rng.choice(product_types, num_rows),
        "On_Off_Net": rng.choice(on_off_net, num_rows),
        "Site_Class": rng.choice(site_class, num_rows),
        "Vendor_Required": rng.choice(vendor_required, num_rows),
        "Region": rng.choice(regions, num_rows),
        "Bundle_Count": rng.integers(1, 6, num_rows)
    })

    # Label logic (for demo clarity):
    # Off-Net + Vendor Required + Bundle_Count > 3  => likely fallout
    score = (df["On_Off_Net"].eq("Off-Net")).astype(int) \
          + (df["Vendor_Required"].eq("Yes")).astype(int) \
          + (df["Bundle_Count"] > 3).astype(int)
    df["Fallout"] = (score >= 2).astype(int)

    df.to_csv(csv_path, index=False)
    return df

if __name__ == "__main__":
    df = generate_dataset()
    print(f"Wrote {len(df)} rows to order_fallout_prediction.csv")
