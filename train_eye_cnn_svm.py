import os
import cv2
import numpy as np
import joblib

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

# ===============================
# PARAMETERS
# ===============================
IMG_SIZE = 64
DATASET_PATH = "CEW/dataset_B_Eye_Images"

CLOSED_DIR = os.path.join(DATASET_PATH, "closedEyes")
OPEN_DIR   = os.path.join(DATASET_PATH, "openEyes")

# ===============================
# LOAD DATASET
# ===============================
X = []
y = []

# closedEyes -> 0
for file in os.listdir(CLOSED_DIR):
    img_path = os.path.join(CLOSED_DIR, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X.append(img)
    y.append(0)

# openEyes -> 1
for file in os.listdir(OPEN_DIR):
    img_path = os.path.join(OPEN_DIR, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    X.append(img)
    y.append(1)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array(y)

print("✅ Dataset loaded")
print("X:", X.shape, "y:", y.shape)

# ===============================
# CNN MODEL
# ===============================
cnn = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

cnn.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===============================
# TRAIN CNN
# ===============================
cnn.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

# ===============================
# FEATURE EXTRACTOR
# ===============================
feature_extractor = Model(
    inputs=cnn.input,
    outputs=cnn.layers[-2].output
)

# ===============================
# EXTRACT FEATURES
# ===============================
features = feature_extractor.predict(X)

# ===============================
# SCALE FEATURES
# ===============================
scaler = StandardScaler()
features = scaler.fit_transform(features)

# ===============================
# LINEAR SVM (BEST FOR FEATURES)
# ===============================
svm = LinearSVC()
svm.fit(features, y)

# ===============================
# SAVE MODELS
# ===============================
feature_extractor.save("eye_cnn_feature_extractor.h5")
joblib.dump(svm, "eye_linear_svm.pkl")
joblib.dump(scaler, "eye_scaler.pkl")

print("✅ eye_cnn_feature_extractor.h5 saved")
print("✅ eye_linear_svm.pkl saved")
print("✅ eye_scaler.pkl saved")
print("🎉 Training completed successfully")
