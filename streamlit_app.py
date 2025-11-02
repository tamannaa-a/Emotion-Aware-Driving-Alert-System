import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from utils import eye_aspect_ratio, play_beep_nonblocking

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("🚗 Real-Time Drowsiness Detection System")

# Mediapipe initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# EAR Threshold + Frame Counter Settings
EAR_THRESHOLD = 0.25
CLOSED_FRAMES_REQUIRED = 15  # ~1 sec if camera runs at ~15 FPS
closed_frames = 0

# Mediapipe eye landmark IDs (left + right)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Video input
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = None

if run:
    cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("⚠️ Unable to access the camera.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    results = face_mesh.process(frame_rgb)

    ear = 1.0  # default if face not detected

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Extract left + right eye points
        left_eye_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE])
        right_eye_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE])

        # EAR calculation
        left_EAR = eye_aspect_ratio(left_eye_points)
        right_EAR = eye_aspect_ratio(right_eye_points)
        ear = (left_EAR + right_EAR) / 2.0

        # Display EAR on screen
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw landmarks
        for point in left_eye_points:
            cv2.circle(frame, tuple(point.astype(int)), 2, (0, 255, 255), -1)
        for point in right_eye_points:
            cv2.circle(frame, tuple(point.astype(int)), 2, (255, 255, 0), -1)

    # ------------------------------
    # ✅ DROWSINESS LOGIC (BLINK-SAFE)
    # ------------------------------
    if ear < EAR_THRESHOLD:
        closed_frames += 1
    else:
        closed_frames = 0

    # Trigger only if eyes are closed for long duration
    if closed_frames >= CLOSED_FRAMES_REQUIRED:
        cv2.putText(frame, "DROWSINESS ALERT!", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        play_beep_nonblocking()
    else:
        cv2.putText(frame, "Awake", (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    FRAME_WINDOW.image(frame, channels="BGR")

if cap:
    cap.release()
