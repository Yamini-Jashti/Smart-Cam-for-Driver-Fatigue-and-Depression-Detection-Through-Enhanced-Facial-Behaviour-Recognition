import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,confusion_matrix   

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "dataset", "people_features.csv")

# ================= LOAD DATA =================
df = pd.read_csv(CSV_PATH)

# Remove duplicates
df = df.drop_duplicates()

# Drop missing values
df = df.dropna()

# ================= EMOTION MERGING =================
emotion_map = {
    "NEUTRAL": "NORMAL",
    "HAPPY": "NORMAL",
    "SURPRISED": "DISTRACTED",
    "ANGRY": "DISTRACTED",
    "SAD": "FATIGUE",
    "YAWNING": "FATIGUE",
    "TIRED": "DROWSY",
    "DROWSY": "DROWSY"
}

df["Emotion"] = df["Emotion"].map(emotion_map)
df = df.dropna(subset=["Emotion"])

# ================= HEAD DIRECTION ENCODING =================
df["HeadDirection"] = df["HeadDirection"].map({
    "Forward": 0,
    "Left": 1,
    "Right": 2,
    "Up": 3,
    "Down": 4
})

# ================= FEATURES =================
X = df[[
    "EAR",
    "MAR",
    "EyeClosedSeconds",
    "BlinkCount",
    "YawnCount",
    "HeadDirection"
]].values

y = df["Emotion"].values

# ================= LABEL ENCODING =================
encoder = LabelEncoder()
y = encoder.fit_transform(y)

np.save("emotion_classes.npy", encoder.classes_)

# ================= SCALE FEATURES =================
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, "feature_scaler.pkl")

# ================= RESHAPE FOR CNN-LSTM =================
X = X.reshape((X.shape[0], X.shape[1], 1))

# ================= TRAIN TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================= MODEL =================
model = Sequential()

# CNN layer
model.add(Conv1D(filters=32, kernel_size=2, activation='relu',
                 input_shape=(X_train.shape[1],1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))

# LSTM layer
model.add(LSTM(64))

# Dense layers
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(len(np.unique(y)), activation='softmax'))

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ================= EARLY STOPPING =================
early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=10,
    restore_best_weights=True
)

# ================= TRAIN =================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop]
)

# ================= SAVE MODEL =================
model.save("driver_state_model.keras")

# ================= EVALUATE =================
loss, acc = model.evaluate(X_test, y_test, verbose=0)

print(f"\nFinal Model Accuracy: {acc*100:.2f}%")

plt.figure()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()

plt.figure()

plt.plot(history.history['loss'],label="Training Loss")
plt.plot(history.history['val_loss'],label="Validation Loss")

plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()


y_pred = model.predict(X_test)

y_pred_classes = np.argmax(y_pred,axis=1)

cm = confusion_matrix(y_test,y_pred_classes)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

accuracy = accuracy_score(y_test,y_pred_classes)

precision = precision_score(y_test,y_pred_classes,average="weighted")

recall = recall_score(y_test,y_pred_classes,average="weighted")

f1 = f1_score(y_test,y_pred_classes,average="weighted")

print("Accuracy :",accuracy)
print("Precision:",precision)
print("Recall   :",recall)
print("F1 Score :",f1)

metrics = ["Accuracy","Precision","Recall","F1 Score"]

values = [accuracy,precision,recall,f1]

plt.figure()

plt.bar(metrics,values)

plt.title("Model Performance Metrics")

plt.ylabel("Score")

plt.ylim(0,1)

plt.show()

