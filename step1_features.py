import cv2
import mediapipe as mp
import numpy as np

# ================= MEDIAPIPE =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ================= LANDMARK INDEXES =================
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
BLINK_THRESH = 0.25
YAWN_THRESH = 0.60

# ================= DISPLAY HOLD =================
blink_hold = 0
yawn_hold = 0

# ================= FEATURE FUNCTIONS =================
def eye_aspect_ratio(lm, eye):
    pts = np.array([(lm[i].x, lm[i].y) for i in eye])
    return (np.linalg.norm(pts[1]-pts[5]) +
            np.linalg.norm(pts[2]-pts[4])) / (2.0*np.linalg.norm(pts[0]-pts[3]))

def mouth_aspect_ratio(lm, mouth):
    pts = np.array([(lm[i].x, lm[i].y) for i in mouth])
    return np.linalg.norm(pts[2]-pts[6]) / np.linalg.norm(pts[0]-pts[4])

def eyebrow_distance(lm):
    l = np.array([lm[LEFT_EYEBROW].x, lm[LEFT_EYEBROW].y])
    r = np.array([lm[RIGHT_EYEBROW].x, lm[RIGHT_EYEBROW].y])
    return np.linalg.norm(l - r)

def head_direction(lm):
    if lm[NOSE].x < lm[LEFT_CHEEK].x:
        return "Looking Left"
    if lm[NOSE].x > lm[RIGHT_CHEEK].x:
        return "Looking Right"
    if lm[CHIN].y - lm[FOREHEAD].y > 0.25:
        return "Looking Down"
    if lm[FOREHEAD].y - lm[CHIN].y < -0.15:
        return "Looking Up"
    return "Looking Forward"


def detect_emotion(ear, mar, brow):
    """
    ear  : Eye Aspect Ratio
    mar  : Mouth Aspect Ratio
    brow : Eyebrow distance
    """

    # ---------- TIRED ----------
    if ear < 0.18:
        return "Tired 😴"

    # ---------- DEPRESSED ----------
    if ear < 0.22 and brow < 0.045 and mar < 0.28:
        return "Depressed 😔"

    # ---------- SAD ----------
    if ear < 0.24 and mar < 0.35:
        return "Sad 😞"

    # ---------- ANGRY ----------
    if brow < 0.040 and ear > 0.23:
        return "Angry 😠"

    # ---------- SURPRISED ----------
    if mar > 0.65 and ear > 0.28:
        return "Surprised 😮"

    # ---------- HAPPY (SMALL SMILE) ----------
    if 0.30 < mar <= 0.45 and ear > 0.26:
        return "Happy 🙂"

    # ---------- NEUTRAL ----------
    return "Neutral 😐"


# ================= CAMERA =================
cap = cv2.VideoCapture(0)
print("STEP-1: Feature Extraction | Press Q to Quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        # ---- Feature values ----
        ear = (eye_aspect_ratio(lm, LEFT_EYE) +
               eye_aspect_ratio(lm, RIGHT_EYE)) / 2
        mar = mouth_aspect_ratio(lm, MOUTH)
        brow = eyebrow_distance(lm)

        emotion = detect_emotion(ear, mar, brow)
        head = head_direction(lm)

        # ---- Blink & Yawn ----
        if ear < BLINK_THRESH:
            blink_hold = 10
        if mar > YAWN_THRESH:
            yawn_hold = 10

        # ---- Display ----
        cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"MAR: {mar:.2f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"Emotion: {emotion}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        cv2.putText(frame, f"Head: {head}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

        if blink_hold > 0:
            cv2.putText(frame, "BLINK DETECTED", (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            blink_hold -= 1

        if yawn_hold > 0:
            cv2.putText(frame, "YAWN DETECTED", (20, 250),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            yawn_hold -= 1

    cv2.imshow("STEP-1: Facial Feature Extraction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
