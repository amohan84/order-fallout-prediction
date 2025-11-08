
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt

DATA_PATH = Path("order_fallout_prediction.csv")
MODEL_PATH = Path("model_decision_tree.pkl")
ENCODERS_PATH = Path("encoders.pkl")
FEATURE_IMPORTANCES_PATH = Path("feature_importances.csv")
TREE_PNG_PATH = Path("tree_plot.png")
REPORT_TXT_PATH = Path("metrics_report.txt")

def main():
    df = pd.read_csv(DATA_PATH)

    # --- add noise to labels to avoid perfect separability ---
    #noise_rate = 0.06  # 6% of rows
    #mask = (pd.Series(np.random.rand(len(df))) < noise_rate)
    #df.loc[mask, "Fallout"] = 1 - df.loc[mask, "Fallout"].astype(int)


    # Encode categoricals
    encoders = {}
    cat_cols = ["Product_Type", "On_Off_Net", "Site_Class", "Vendor_Required", "Region"]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df.drop(columns=["Fallout"])
    y = df["Fallout"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(max_depth=2, min_samples_leaf=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    # Save artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)

    fi = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    fi.to_csv(FEATURE_IMPORTANCES_PATH, index=False)

    # Plot tree
    plt.figure(figsize=(16, 10))
    plot_tree(model, feature_names=X.columns, class_names=["NoFallout","Fallout"], filled=False, rounded=True)
    plt.tight_layout()
    plt.savefig(TREE_PNG_PATH, dpi=200)
    plt.close()

    # Write text report
    with open(REPORT_TXT_PATH, "w") as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    print("=== Training Complete ===")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    print(f"\nSaved model → {MODEL_PATH}")
    print(f"Saved encoders → {ENCODERS_PATH}")
    print(f"Saved feature importances → {FEATURE_IMPORTANCES_PATH}")
    print(f"Saved tree plot → {TREE_PNG_PATH}")
    print(f"Saved metrics → {REPORT_TXT_PATH}")

if __name__ == "__main__":
    main()
