# streamlit_app.py

import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from utils import play_beep_nonblocking

st.set_page_config(page_title="Drowsiness Detection", layout="wide")

st.title("🚘 Real-Time Drowsiness Detection System")
st.write("Eyes closed for long = Drowsiness alert with beep sound.")

# ========== CONFIG ==========
EAR_THRESHOLD = 0.25
CLOSED_FRAMES_REQUIRED = 15    # approx 0.5 sec at ~30 FPS
# ============================

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Mediapipe Eye Indexes
LEFT_EYE = [159, 145, 33, 133]
RIGHT_EYE = [386, 374, 263, 362]

def euclidean_dist(a, b):
    return np.linalg.norm(a - b)

def eye_aspect_ratio(landmarks, eye_idx, img_w, img_h):
    p1 = np.array([landmarks[eye_idx[0]].x * img_w, landmarks[eye_idx[0]].y * img_h])
    p2 = np.array([landmarks[eye_idx[1]].x * img_w, landmarks[eye_idx[1]].y * img_h])
    p3 = np.array([landmarks[eye_idx[2]].x * img_w, landmarks[eye_idx[2]].y * img_h])
    p4 = np.array([landmarks[eye_idx[3]].x * img_w, landmarks[eye_idx[3]].y * img_h])

    ear = (euclidean_dist(p1, p2)) / (2 * euclidean_dist(p3, p4))
    return ear

stframe = st.empty()

closed_frames = 0
alert_active = False

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("❌ Camera not detected!")
else:
    st.success("✅ Camera connected successfully!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("❗ Cannot read from camera")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    h, w = frame.shape[:2]

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        leftEAR = eye_aspect_ratio(lm, LEFT_EYE, w, h)
        rightEAR = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        EAR = (leftEAR + rightEAR) / 2.0

        # Eye State Logic
        if EAR < EAR_THRESHOLD:
            closed_frames += 1
        else:
            closed_frames = 0
            alert_active = False

        # Drowsiness Condition
        if closed_frames >= CLOSED_FRAMES_REQUIRED and not alert_active:
            alert_active = True
            play_beep_nonblocking()

            cv2.putText(frame, "DROWSINESS ALERT!", (60, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, f"EAR: {EAR:.3f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Eye labels
        state = "Closed" if EAR < EAR_THRESHOLD else "Open"
        cv2.putText(frame, f"Eyes: {state}", (20, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    stframe.image(frame, channels="BGR")

cap.release()
face_mesh.close()
