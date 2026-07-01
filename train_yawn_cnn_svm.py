import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
import joblib
from sklearn.svm import LinearSVC


# ================= CONFIG =================
DATASET_PATH = "."

IMG_SIZE = 64
X, y = [], []

for label, folder in enumerate(["yawn", "no yawn"]):
    folder_path = os.path.join(DATASET_PATH, folder)

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(label)



X = np.array(X) / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)

print("✅ Dataset loaded:", X.shape, y.shape)

# ================= TRAIN / TEST SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= CNN FEATURE EXTRACTOR =================
cnn = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten()
])

cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Extract features
X_train_feat = cnn.predict(X_train)
X_test_feat = cnn.predict(X_test)

# ================= SVM CLASSIFIER =================

svm = LinearSVC(max_iter=10000)
svm.fit(X_train_feat, y_train)

# ================= EVALUATION =================
y_pred = svm.predict(X_test_feat)
acc = accuracy_score(y_test, y_pred)

print(f"✅ Yawning Model Accuracy: {acc*100:.2f}%")

# ================= SAVE MODELS =================
cnn.save("yawn_cnn_feature_extractor.h5")
joblib.dump(svm, "yawn_svm_model.pkl")

print("✅ Models saved successfully")
