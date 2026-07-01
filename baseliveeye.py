import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ===============================
# LOAD MODELS
# ===============================
eye_cnn = load_model("eye_cnn_feature_extractor.h5")
eye_svm = joblib.load("eye_linear_svm.pkl")
eye_scaler = joblib.load("eye_scaler.pkl")

# ===============================
# LOAD HAAR CASCADES
# ===============================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ===============================
# WEBCAM
# ===============================
cap = cv2.VideoCapture(0)
print("🚗 Live Driver Monitoring Started (Press Q to quit)")

# ===============================
# PREDICT FUNCTION
# ===============================
def predict_eye_state(eye_img):
    eye_img = cv2.resize(eye_img, (64, 64))
    eye_img = eye_img.reshape(1, 64, 64, 1) / 255.0

    features = eye_cnn.predict(eye_img, verbose=0)
    features = eye_scaler.transform(features)

    pred = eye_svm.predict(features)[0]
    return "OPEN" if pred == 1 else "CLOSED"

# ===============================
# LIVE LOOP
# ===============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            state = predict_eye_state(eye_img)

            color = (0, 255, 0) if state == "OPEN" else (0, 0, 255)
            cv2.rectangle(frame,
                          (x+ex, y+ey),
                          (x+ex+ew, y+ey+eh),
                          color, 2)

            cv2.putText(frame, state,
                        (x+ex, y+ey-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

            break  # only first eye

    cv2.imshow("Driver Eye Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===============================
# CLEANUP
# ===============================
cap.release()
cv2.destroyAllWindows()
