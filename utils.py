# utils.py
import os
import threading

# Try winsound (Windows only)
try:
    import winsound
    _HAS_WINSOUND = True
except:
    _HAS_WINSOUND = False

BEEP_PATH = os.path.join(os.path.dirname(__file__), "beep.wav")

def play_beep_blocking():
    """Play beep using winsound for Windows."""
    if _HAS_WINSOUND:
        if os.path.exists(BEEP_PATH):
            winsound.PlaySound(BEEP_PATH, winsound.SND_FILENAME)
        else:
            winsound.Beep(2000, 700)  # fallback beep
    else:
        # No beep available on non-Windows systems
        pass

def play_beep_nonblocking():
    """Play beep in a background thread."""
    threading.Thread(target=play_beep_blocking, daemon=True).start()
