import cv2

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
print("😮 Live Yawn Detection (Mouth Open Based)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Mouth region (lower face)
        mx1 = x + w//4
        mx2 = x + 3*w//4
        my1 = y + int(0.6*h)
        my2 = y + int(0.9*h)

        mouth = gray[my1:my2, mx1:mx2]

        # Mouth opening heuristic
        mouth_height = my2 - my1
        face_height = h

        ratio = mouth_height / face_height

        if ratio > 0.25:   # OPEN mouth
            state = "YAWN"
            color = (0, 0, 255)
        else:
            state = "NO YAWN"
            color = (0, 255, 0)

        cv2.rectangle(frame, (mx1, my1), (mx2, my2), color, 2)
        cv2.putText(frame, state, (mx1, my1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Live Yawn Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
