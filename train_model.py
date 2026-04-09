import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("merged_lesion_EDSS_patient_info.csv")

#filter progressive patients
def progression(edss):
    if edss <= 1.5:
        return 0
    elif edss >= 2:
        return 1
    else:
        return None

df["progression"] = df["EDSS"].apply(progression)
df = df.dropna(subset=["progression"])

df_prog = df[df["progression"] == 1].copy()

# Create severity labels
def edss_severity(edss):
    if 2 <= edss <= 3:
        return 0
    elif 3.5 <= edss <= 5:
        return 1
    elif 5.5 <= edss <= 9.5:
        return 2
    else:
        return None

df_prog["severity"] = df_prog["EDSS"].apply(edss_severity)
df_prog = df_prog.dropna(subset=["severity"])

# Features
feature_cols = [
    "Lesion_Volume_mL",
    "EDSS",
    "Mean_Lesion_Size_mL",
    "Lesion_Count"
]

X = df_prog[feature_cols]
y = df_prog["severity"].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    class_weight="balanced",
    random_state=42,
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "ms_severity_model.pkl")

print("Model saved successfully!")