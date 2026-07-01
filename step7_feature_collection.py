import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os

# ================= PATH SETUP =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)
CSV_PATH = os.path.join(DATASET_DIR, "people_features.csv")

# Create CSV header once
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow([
            "PersonID",
            "EAR",
            "BlinkCount",
            "MAR",
            "YawnCount",
            "EyeClosedSeconds",
            "HeadDirection",
            "Emotion",
            "Timestamp"
        ])

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

LEFT_EYEBROW = 70
RIGHT_EYEBROW = 300

NOSE = 1
LEFT_CHEEK = 234
RIGHT_CHEEK = 454
CHIN = 152
FOREHEAD = 10

# ================= THRESHOLDS =================
BLINK_THRESH = 0.20
YAWN_THRESH = 0.65
EYE_CLOSE_TIME_LIMIT = 10
SAVE_INTERVAL = 2.0   # 20–30 samples per minute

# ================= COUNTERS =================
blink_count = 0
yawn_count = 0
eye_closed = False
mouth_open = False
eye_close_start = None
eye_closed_seconds = 0

last_save_time = 0

# ================= FUNCTIONS =================
def eye_aspect_ratio(lm, eye):
    pts = np.array([(lm[i].x, lm[i].y) for i in eye])
    return (np.linalg.norm(pts[1] - pts[5]) +
            np.linalg.norm(pts[2] - pts[4])) / (2.0 * np.linalg.norm(pts[0] - pts[3]))

def mouth_aspect_ratio(lm, mouth):
    pts = np.array([(lm[i].x, lm[i].y) for i in mouth])
    return np.linalg.norm(pts[2] - pts[6]) / np.linalg.norm(pts[0] - pts[4])

def eyebrow_distance(lm):
    l = np.array([lm[LEFT_EYEBROW].x, lm[LEFT_EYEBROW].y])
    r = np.array([lm[RIGHT_EYEBROW].x, lm[RIGHT_EYEBROW].y])
    return np.linalg.norm(l - r)

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

# ================= EMOTION LOGIC =================
def detect_emotion_strong(ear, mar, eye_closed_time, blink_event, yawn_event):

    if eye_closed_time >= 10:
        return "DROWSY"

    if ear < 0.18:
        return "TIRED"

    if blink_event:
        return "BLINK_DETECTED"

    if yawn_event:
        return "YAWNING"

    if ear < 0.22 and mar < 0.35:
        return "ANGRY"

    if 0.18 <= ear <= 0.22 and 0.35 <= mar <= 0.45:
        return "SAD"

    if mar > 0.70:
        return "SURPRISED"
    if 0.40 < mar <= 0.55:
        return "HAPPY"
    return "NEUTRAL"

# ================= INPUT =================
person_id = input("Enter Person ID (P1 / P2 / etc): ").strip()

# ================= CAMERA =================
cap = cv2.VideoCapture(0)
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blink_event = False
    yawn_event = False

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        ear = (eye_aspect_ratio(lm, LEFT_EYE) +
               eye_aspect_ratio(lm, RIGHT_EYE)) / 2
        mar = mouth_aspect_ratio(lm, MOUTH)
        head_dir = get_head_direction(lm)

        # ---- Blink detection ----
        if ear < BLINK_THRESH:
            eye_closed = True
            if eye_close_start is None:
                eye_close_start = time.time()
        else:
            if eye_closed:
                blink_count += 1
                blink_event = True
            eye_closed = False
            eye_close_start = None
            eye_closed_seconds = 0

        if eye_close_start:
            eye_closed_seconds = time.time() - eye_close_start

        # ---- Yawn detection ----
        if mar > YAWN_THRESH:
            mouth_open = True
        else:
            if mouth_open:
                yawn_count += 1
                yawn_event = True
            mouth_open = False

        emotion = detect_emotion_strong(
            ear,
            mar,
            eye_closed_seconds,
            blink_event,
            yawn_event
        )

        # ================= SAVE =================
        now = time.time()
        if now - last_save_time >= SAVE_INTERVAL:
            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow([
                    person_id,
                    round(ear, 3),
                    blink_count,
                    round(mar, 3),
                    yawn_count,
                    round(eye_closed_seconds, 2),
                    head_dir,
                    emotion,
                    time.strftime("%H:%M:%S")
                ])
            last_save_time = now

        # ================= DISPLAY =================
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(frame, f"Blink Count: {blink_count}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        cv2.putText(frame, f"Yawn Count: {yawn_count}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        cv2.putText(frame, f"Eye Closed: {eye_closed_seconds:.1f}s", (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(frame, f"Head: {head_dir}", (20, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)

        cv2.putText(frame, f"Emotion: {emotion}", (20, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("STEP-2 Data Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Data saved in:", CSV_PATH)
