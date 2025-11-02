# streamlit_app.py
import cv2
import numpy as np
import streamlit as st
import time
from utils import play_beep_nonblocking, BEEP_PATH
import mediapipe as mp

st.set_page_config(page_title="Emotion-aware Driving Alert (EAR)", layout="centered")

# -------------------------
# Tunable parameters
# -------------------------
EAR_THRESH = 0.24          # tune if too sensitive
LONG_CLOSED_FRAMES = 20    # eyes must stay closed ~1 sec
SHOW_FPS = True
DRAW_LANDMARKS = True
# -------------------------

# MediaPipe setup
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Landmark indices for EAR
LEFT_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_IDX = [362, 385, 387, 263, 373, 380]

st.title("🚗 Emotion-aware Driving Alert — Eye State (MediaPipe EAR)")

col1, col2 = st.columns([3,1])
frame_display = col1.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="RGB")
info_box = col2.empty()

info_box.markdown("""
### Instructions
- Allow camera access.
- Good lighting, no sunglasses.
- Tune EAR and CLOSE duration if needed.
- Beep plays only when eyes are closed **long enough**, not for blinks.
""")

status_text = st.empty()
ear_text = st.empty()

# OpenCV Camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Camera not detected. Close other apps or try another device.")
    st.stop()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# EAR helper functions
def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def compute_ear(landmarks, idxs, img_w, img_h):
    try:
        coords = []
        for i in idxs:
            lm = landmarks[i]
            coords.append((lm.x * img_w, lm.y * img_h))
        p1, p2, p3, p4, p5, p6 = coords
        A = euclid(p2, p6)
        B = euclid(p3, p5)
        C = euclid(p1, p4)
        if C == 0:
            return None
        return (A + B) / (2.0 * C)
    except:
        return None

# loop state
closed_frames = 0
alarm_on = False
ear_val = None

with mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as face_mesh:

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            status_text.error("Failed to read frame.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame_rgb.shape[:2]

        results = face_mesh.process(frame_rgb)

        ear_val = None
        eye_state = "No face"
        color = (0, 255, 0)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # EAR for both eyes
            ear_l = compute_ear(lm, LEFT_IDX, img_w, img_h)
            ear_r = compute_ear(lm, RIGHT_IDX, img_w, img_h)

            if ear_l is not None and ear_r is not None:
                ear_val = (ear_l + ear_r) / 2.0

                # --- NEW LOGIC: No beep for blinks ---
                if ear_val < EAR_THRESH:
                    closed_frames += 1
                    eye_state = "Closed"
                    color = (0, 0, 255)

                    # ALARM ONLY if eyes stay closed long enough
                    if closed_frames >= LONG_CLOSED_FRAMES and not alarm_on:
                        alarm_on = True
                        play_beep_nonblocking()

                else:
                    # Eyes open -> reset everything  
                    closed_frames = 0
                    alarm_on = False
                    eye_state = "Open"
                    color = (0, 255, 0)

                # draw landmarks
                if DRAW_LANDMARKS:
                    mp_drawing.draw_landmarks(
                        image=frame_rgb,
                        landmark_list=results.multi_face_landmarks[0],
                        connections=mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 200, 150), thickness=1, circle_radius=1
                        ),
                    )
                    for idx in LEFT_IDX + RIGHT_IDX:
                        p = lm[idx]
                        cx, cy = int(p.x * img_w), int(p.y * img_h)
                        cv2.circle(frame_rgb, (cx, cy), 2, (255, 255, 0), -1)
            else:
                eye_state = "Face detected, EAR failed"
        else:
            closed_frames = 0
            alarm_on = False

        # overlay text
        overlay = frame_rgb.copy()
        t = f"Eye: {eye_state}"
        if ear_val is not None:
            t += f" | EAR: {ear_val:.3f}"
        t += f" | ClosedFrames: {closed_frames}"
        cv2.putText(overlay, t, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # FPS
        if SHOW_FPS:
            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0
            prev_time = now
            cv2.putText(
                overlay,
                f"FPS: {fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )

        # stream to UI
        frame_display.image(overlay, channels="RGB")
        status_text.markdown(f"**Status:** {t}")

        if ear_val is not None:
            ear_text.markdown(f"**EAR (avg): {ear_val:.3f}**")
        else:
            ear_text.markdown("**EAR (avg): -**")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
