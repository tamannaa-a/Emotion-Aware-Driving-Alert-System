import numpy as np
import simpleaudio as sa
import threading

def eye_aspect_ratio(eye):
    # EAR formula
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def play_beep_nonblocking():
    def _play():
        try:
            wave_obj = sa.WaveObject.from_wave_file("beep.wav")
            wave_obj.play()
        except Exception as e:
            print("Beep Error:", e)

    threading.Thread(target=_play, daemon=True).start()
