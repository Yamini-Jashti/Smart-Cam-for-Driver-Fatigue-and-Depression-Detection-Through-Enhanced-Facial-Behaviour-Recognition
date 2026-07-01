import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import csv
import time
import os
import winsound

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "people_features.csv")

# ================= LOAD DATASET =================
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError("people_features.csv not found. Complete Step-2 first.")

df = pd.read_csv(DATASET_PATH)

# ================= MEDIAPIPE =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= LANDMARKS =================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [78, 81, 13, 311, 308, 402, 14]

NOSE = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
CHIN = 152
FOREHEAD = 10

# ================= THRESHOLDS =================
BLINK_THRESH = 0.20
YAWN_THRESH = 0.65

BLINK_ALERT = 30
YAWN_ALERT = 30
EYE_CLOSE_DANGER = 10
TIME_WINDOW = 60   # seconds

# ================= COUNTERS =================
blink_count = 0
yawn_count = 0
eye_closed = False
mouth_open = False
eye_close_start = None
eye_closed_seconds = 0
# ================= TIMER =================
minute_start = time.time()
BLINK_LIMIT = 30
YAWN_LIMIT = 30

# ================= FUNCTIONS =================
def eye_aspect_ratio(lm, eye):
    pts = np.array([(lm[i].x, lm[i].y) for i in eye])
    return (np.linalg.norm(pts[1]-pts[5]) +
            np.linalg.norm(pts[2]-pts[4])) / (2.0*np.linalg.norm(pts[0]-pts[3]))

def mouth_aspect_ratio(lm, mouth):
    pts = np.array([(lm[i].x, lm[i].y) for i in mouth])
    return np.linalg.norm(pts[2]-pts[6]) / np.linalg.norm(pts[0]-pts[4])

def get_head_direction(lm):
    if lm[NOSE].x < lm[LEFT_CHEEK].x:
        return "Left"
    elif lm[NOSE].x > lm[RIGHT_CHEEK].x:
        return "Right"
    elif lm[CHIN].y - lm[FOREHEAD].y > 0.25:
        return "Down"
    elif lm[FOREHEAD].y - lm[CHIN].y < -0.15:
        return "Up"
    return "Forward"

def detect_emotion(ear, mar, eye_closed_sec):
    if eye_closed_sec >= 10:
        return "DROWSY"
    if ear < 0.18:
        return "TIRED"
    if mar > 0.70:
        return "SURPRISED"
    if ear < 0.22 and mar < 0.35:
        return "ANGRY"
    if 0.18 <= ear <= 0.22:
        return "SAD"
    return "NEUTRAL"

def match_person(live_ear, live_mar, data, threshold=0.05):
    for _, row in data.iterrows():
        if abs(row["EAR"] - live_ear) < threshold and abs(row["MAR"] - live_mar) < threshold:
            return row["PersonID"]
    return None

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    alert_msg = ""
    person_msg = ""

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        ear = (eye_aspect_ratio(lm, LEFT_EYE) +
               eye_aspect_ratio(lm, RIGHT_EYE)) / 2
        mar = mouth_aspect_ratio(lm, MOUTH)
        head_dir = get_head_direction(lm)

        # ===== Eye closed logic =====
        if ear < BLINK_THRESH:
            if eye_close_start is None:
                eye_close_start = time.time()
            eye_closed_seconds = time.time() - eye_close_start
        else:
            if eye_close_start:
                blink_count += 1
            eye_close_start = None
            eye_closed_seconds = 0

        # ===== Yawn logic =====
        if mar > YAWN_THRESH:
            mouth_open = True
        else:
            if mouth_open:
                yawn_count += 1
            mouth_open = False

        emotion = detect_emotion(ear, mar, eye_closed_seconds)

        '''# ===== ALERTS =====
        if eye_closed_seconds >= EYE_CLOSE_DANGER:
            alert_msg = "🚨 DANGER: Eyes Closed"
            winsound.Beep(1200, 800)
        elif blink_count >= BLINK_ALERT:
            alert_msg = "⚠ FATIGUE: Excess Blinking"
            winsound.Beep(1000, 500)
        elif yawn_count >= YAWN_ALERT:
            alert_msg = "⚠ FATIGUE: Excess Yawning"
            winsound.Beep(1000, 500)
        else:
            alert_msg = "Driver State: NORMAL"'''
            # ===== ALERTS WITH 1 MIN WINDOW =====
        current_time = time.time()

        # reset every 60 seconds
        if current_time - minute_start >= TIME_WINDOW:
            blink_count = 0
            yawn_count = 0
            minute_start = current_time

        if eye_closed_seconds >= EYE_CLOSE_DANGER:
            alert_msg = "🚨 DANGER: Eyes Closed"
            winsound.Beep(1200, 800)
            time.sleep(1)

        elif blink_count >= BLINK_LIMIT:
            alert_msg = "⚠ FATIGUE: Excess Blinking in 1 Min"
            winsound.Beep(1000, 500)
            time.sleep(1)

        elif yawn_count >= YAWN_LIMIT:
            alert_msg = "⚠ FATIGUE: Excess Yawning in 1 Min"
            winsound.Beep(1000, 500)
            time.sleep(1)

        else:
            alert_msg = "Driver State: NORMAL"

            # ===== PERSON MATCH =====
        matched_id = match_person(ear, mar, df)

        if matched_id:
            person_msg = f"Matched ID: {matched_id}"
        else:
            new_id = f"P{len(df['PersonID'].unique()) + 1}"
            with open(DATASET_PATH, "a", newline="") as f:
                csv.writer(f).writerow([
                    new_id, ear, blink_count, mar, yawn_count,
                    eye_closed_seconds, head_dir, emotion, time.time()
                ])
            df = pd.read_csv(DATASET_PATH)
            person_msg = f"New Person Added: {new_id}"

        # ===== DISPLAY =====
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 40), 0, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (30, 70), 0, 0.8, (0,255,0), 2)
        cv2.putText(frame, f"Blink Count: {blink_count}", (30, 110), 0, 0.8, (255,255,0), 2)
        cv2.putText(frame, f"Yawn Count: {yawn_count}", (30, 150), 0, 0.8, (255,255,0), 2)
        cv2.putText(frame, f"Eyes Closed: {eye_closed_seconds:.1f}s", (30, 190), 0, 0.8, (0,0,255), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (30, 230), 0, 0.9, (255,255,0), 2)
        cv2.putText(frame, f"Head: {head_dir}", (30, 270), 0, 0.9, (255,255,0), 2)
        cv2.putText(frame, person_msg, (30, 310), 0, 1, (255,0,0), 3)
        cv2.putText(frame, alert_msg, (30, 350), 0, 1, (0,0,255), 3)

    cv2.imshow("STEP-3 Live Driver Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
