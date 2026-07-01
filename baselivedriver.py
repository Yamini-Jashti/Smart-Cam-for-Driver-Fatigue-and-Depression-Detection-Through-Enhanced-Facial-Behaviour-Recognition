import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import winsound   # Windows alert

# ===================== LOAD MODELS =====================
eye_cnn = load_model("eye_cnn_feature_extractor.h5")
eye_svm = joblib.load("eye_linear_svm.pkl")
eye_scaler = joblib.load("eye_scaler.pkl")

# ===================== HAAR CASCADES =====================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# ===================== PARAMETERS =====================
EYE_CONFIRM_FRAMES = 5       # eye must be closed for 5 frames
YAWN_CONFIRM_FRAMES = 5      # yawn must persist
eye_temp_counter = 0
yawn_counter = 0

# ===================== WEBCAM =====================
cap = cv2.VideoCapture(0)
print("🚗 Stable Driver Drowsiness Detection Started (Press Q to quit)")

# ===================== EYE PREDICTION =====================
def predict_eye_state(eye_img):
    eye_img = cv2.resize(eye_img, (64, 64))
    eye_img = eye_img.reshape(1,64,64,1) / 255.0
    features = eye_cnn.predict(eye_img, verbose=0)
    features = eye_scaler.transform(features)
    pred = eye_svm.predict(features)[0]
    return "OPEN" if pred == 1 else "CLOSED"

# ===================== MAIN LOOP =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    drowsy = False

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # ---------------- EYE DETECTION ----------------
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        final_eye_state = "OPEN"

        for (ex, ey, ew, eh) in eyes:
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            state = predict_eye_state(eye_img)

            if state == "CLOSED":
                eye_temp_counter += 1
            else:
                eye_temp_counter = 0

            if eye_temp_counter >= EYE_CONFIRM_FRAMES:
                final_eye_state = "CLOSED"
                drowsy = True

            color = (0,255,0) if final_eye_state=="OPEN" else (0,0,255)
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), color, 2)
            cv2.putText(roi_color, final_eye_state, (ex,ey-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            break   # one eye is enough

        # ---------------- YAWN DETECTION ----------------
        mx1 = x + w//4
        mx2 = x + 3*w//4
        my1 = y + int(0.6*h)
        my2 = y + int(0.9*h)

        mouth_height_ratio = (my2 - my1) / h
        mouth_width_ratio  = (mx2 - mx1) / w

        if mouth_height_ratio > 0.30 and mouth_width_ratio > 0.40:
            yawn_counter += 1
        else:
            yawn_counter = 0

        if yawn_counter >= YAWN_CONFIRM_FRAMES:
            yawn_state = "YAWN"
            drowsy = True
            y_color = (0,0,255)
        else:
            yawn_state = "NO YAWN"
            y_color = (0,255,0)

        cv2.rectangle(frame, (mx1,my1), (mx2,my2), y_color, 2)
        cv2.putText(frame, yawn_state, (mx1,my1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, y_color, 2)

        # ---------------- ALERT ----------------
        if drowsy:
            cv2.putText(frame, "DROWSY ALERT!", (40,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0,0,255), 3)
            winsound.Beep(1000, 200)

    cv2.imshow("Stable Driver Drowsiness Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
