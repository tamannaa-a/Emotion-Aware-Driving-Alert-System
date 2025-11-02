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
EAR_THRESH = 0.24         # typical range 0.20 - 0.27. Raise if false-positive closed->open
CONSEC_FRAMES = 10        # frames (at ~15-30 FPS). 10 ~ 0.3-1s depending on fps
SHOW_FPS = True
DRAW_LANDMARKS = True
# -------------------------

# MediaPipe setup
mp_face = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# These landmark indices work well for MediaPipe face_mesh
# Left eye: p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
# Right eye: p1=362, p2=385, p3=387, p4=263, p5=373, p6=380
LEFT_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_IDX = [362, 385, 387, 263, 373, 380]

st.title("🚗 Emotion-aware Driving Alert System")

col1, col2 = st.columns([3,1])
frame_display = col1.image(np.zeros((480, 640, 3), dtype=np.uint8), channels="RGB")
info_box = col2.empty()

# show instructions
info_box.markdown("""
**Instructions**
- Allow camera access when prompted.
- For best results: face the camera straight, good lighting, no sunglasses.
- Tune `EAR_THRESH` and `CONSEC_FRAMES` if detection is too sensitive.
- `beep.wav` is used for alarm (placed in repo root).
""")

# status widgets
status_text = st.empty()
ear_text = st.empty()

# OpenCV capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Could not open camera. Make sure no other app is using it and device index is correct.")
    st.stop()

# Ensure we have a small frame for speed
CAP_WIDTH = 640
CAP_HEIGHT = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)

# EAR helper
def euclid(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def compute_ear(landmarks, idxs, img_w, img_h):
    """
    landmarks: list of mp normalized landmarks
    idxs: list of 6 indices [p1,p2,p3,p4,p5,p6]
    returns EAR scalar or None
    """
    try:
        coords = []
        for i in idxs:
            lm = landmarks[i]
            coords.append((lm.x * img_w, lm.y * img_h))
        p1, p2, p3, p4, p5, p6 = coords
        # vertical distances
        A = euclid(p2, p6)
        B = euclid(p3, p5)
        # horizontal distance
        C = euclid(p1, p4)
        if C == 0:
            return None
        ear = (A + B) / (2.0 * C)
        return ear
    except Exception:
        return None

# main loop state
closed_frames = 0
alarm_on = False
last_beep_time = 0.0
BEEP_MIN_COOLDOWN = 2.0   # seconds between beeps when alarm remains ON

with mp_face.FaceMesh(static_image_mode=False,
                      max_num_faces=1,
                      refine_landmarks=True,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as face_mesh:

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            status_text.error("Failed to read frame from camera.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]

        # MediaPipe expects RGB
        results = face_mesh.process(frame_rgb)

        ear_val = None
        eye_state = "No face"
        color = (0, 255, 0)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            # compute eyes EAR
            ear_l = compute_ear(lm, LEFT_IDX, img_w, img_h)
            ear_r = compute_ear(lm, RIGHT_IDX, img_w, img_h)
            if ear_l is not None and ear_r is not None:
                ear_val = (ear_l + ear_r) / 2.0
                # determine open/closed
                if ear_val < EAR_THRESH:
                    closed_frames += 1
                    eye_state = "Closed"
                    color = (0, 0, 255)
                else:
                    closed_frames = 0
                    alarm_on = False
                    eye_state = "Open"
                    color = (0, 255, 0)

                # Draw landmarks if requested
                if DRAW_LANDMARKS:
                    mp_drawing.draw_landmarks(
                        image=frame_rgb,
                        landmark_list=results.multi_face_landmarks[0],
                        connections=mp_face.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,200,150), thickness=1, circle_radius=1)
                    )

                    # highlight eye points
                    for ix in LEFT_IDX + RIGHT_IDX:
                        p = lm[ix]
                        cx, cy = int(p.x*img_w), int(p.y*img_h)
                        cv2.circle(frame_rgb, (cx, cy), 2, (255,255,0), -1)

                # Alarm condition
                if closed_frames >= CONSEC_FRAMES:
                    # avoid continuous spamming; use cooldown
                    now = time.time()
                    if not alarm_on or (now - last_beep_time) >= BEEP_MIN_COOLDOWN:
                        alarm_on = True
                        last_beep_time = now
                        play_beep_nonblocking()
            else:
                eye_state = "Face but EAR failed"
        else:
            closed_frames = 0
            alarm_on = False

        # overlay info on frame
        overlay = frame_rgb.copy()
        text = f"Eye: {eye_state}"
        if ear_val is not None:
            text += f" | EAR: {ear_val:.3f}"
        text += f" | ClosedFrames: {closed_frames}"
        cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        # FPS
        if SHOW_FPS:
            now = time.time()
            fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
            prev_time = now
            cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        # Convert back for Streamlit display
        frame_display.image(overlay, channels="RGB")

        # Update side info
        status_text.markdown(f"**Status**: {text}")
        if ear_val is not None:
            ear_text.markdown(f"**EAR (avg)**: {ear_val:.3f}")
        else:
            ear_text.markdown("**EAR (avg)**: -")

        # short delay - necessary so Streamlit UI remains interactive
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
