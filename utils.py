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
    """Play beep using winsound (Windows) or do nothing."""
    if _HAS_WINSOUND and os.path.exists(BEEP_PATH):
        winsound.PlaySound(BEEP_PATH, winsound.SND_FILENAME)
    elif _HAS_WINSOUND:
        winsound.Beep(2000, 700)  # Fallback beep
    else:
        # On platforms without winsound, Streamlit app will use st.audio fallback
        pass

def play_beep_nonblocking():
    """Run beep in thread."""
    threading.Thread(target=play_beep_blocking, daemon=True).start()
