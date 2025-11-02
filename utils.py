# utils.py
import os
import threading

# Try winsound (Windows)
try:
    import winsound
    _HAS_WINSOUND = True
except:
    _HAS_WINSOUND = False

BEEP_PATH = os.path.join(os.path.dirname(__file__), "beep.wav")

def play_beep_blocking():
    if _HAS_WINSOUND and os.path.exists(BEEP_PATH):
        winsound.PlaySound(BEEP_PATH, winsound.SND_FILENAME)
    elif _HAS_WINSOUND:
        winsound.Beep(2000, 700)
    else:
        pass

def play_beep_nonblocking():
    threading.Thread(target=play_beep_blocking, daemon=True).start()
