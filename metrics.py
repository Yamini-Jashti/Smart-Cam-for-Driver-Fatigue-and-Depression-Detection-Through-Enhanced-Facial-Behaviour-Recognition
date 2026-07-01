import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# ================= LOAD DATASET =================
DATA_PATH = "dataset/people_features.csv"
df = pd.read_csv(DATA_PATH)

# ================= AUTO LABELING =================
def assign_label(row):
    if row["EyeClosedSeconds"] >= 10:
        return "HIGH_RISK"
    elif row["EAR"] < 0.18:
        return "DROWSY"
    elif row["BlinkCount"] > 15 or row["YawnCount"] > 5:
        return "FATIGUE"
    else:
        return "NORMAL"

df["Label"] = df.apply(assign_label, axis=1)

print("\n===== LABEL DISTRIBUTION =====")
print(df["Label"].value_counts())

# ================= ADD SMALL NOISE =================
# Break perfect rule learning (important)

noise = 0.02

df["EAR"] += np.random.normal(0, noise, len(df))
df["MAR"] += np.random.normal(0, noise, len(df))
df["BlinkCount"] += np.random.randint(-2, 3, len(df))
df["YawnCount"] += np.random.randint(-1, 2, len(df))

# ================= FEATURES =================
X = df[[
    "EAR",
    "BlinkCount",
    "MAR",
    "YawnCount",
    "EyeClosedSeconds"
]]

y = df["Label"]

# ================= ENCODE LABELS =================
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# ================= TRAIN MODEL =================
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ================= PREDICTION =================
y_pred = model.predict(X_test)

# ================= METRICS =================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\n===== PERFORMANCE METRICS =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ================= CONFUSION MATRIX =================
print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, y_pred))

# ================= CLASSIFICATION REPORT =================
print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))