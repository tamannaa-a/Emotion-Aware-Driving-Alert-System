# utils.py
import os
import threading

# Try using winsound (only available on Windows)
try:
    import winsound
    _HAS_WINSOUND = True
except:
    _HAS_WINSOUND = False

BEEP_PATH = os.path.join(os.path.dirname(__file__), "beep.wav")

def play_beep_blocking():
    """Play beep sound (blocking) using winsound."""
    if _HAS_WINSOUND:
        if os.path.exists(BEEP_PATH):
            winsound.PlaySound(BEEP_PATH, winsound.SND_FILENAME)
        else:
            # Fallback if beep.wav missing
            winsound.Beep(2000, 700)
    else:
        # Non-Windows: no beep
        pass

def play_beep_nonblocking():
    """Play beep in separate thread so UI does not freeze."""
    threading.Thread(target=play_beep_blocking, daemon=True).start()
