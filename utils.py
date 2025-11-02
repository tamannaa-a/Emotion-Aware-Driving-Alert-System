# utils.py
import os
import io
import threading

# Try to import simpleaudio (preferred), else winsound (Windows), else None
try:
    import simpleaudio as sa
    _HAS_SIMPLEAUDIO = True
except Exception:
    _HAS_SIMPLEAUDIO = False

try:
    import winsound
    _HAS_WINSOUND = True
except Exception:
    _HAS_WINSOUND = False

# Path to beep file (make sure beep.wav is in repo root)
BEEP_PATH = os.path.join(os.path.dirname(__file__), "beep.wav")

def _load_beep_bytes():
    """Return bytes of beep.wav or None if not present."""
    if not os.path.exists(BEEP_PATH):
        return None
    with open(BEEP_PATH, "rb") as f:
        return f.read()

_BEEP_BYTES = _load_beep_bytes()

def play_beep_blocking():
    """
    Play beep synchronously. Tries simpleaudio -> winsound -> no-op.
    Use in a separate thread when you don't want to block app.
    """
    if _BEEP_BYTES is None:
        # no beep file; try winsound default beep
        if _HAS_WINSOUND:
            winsound.Beep(2000, 700)  # freq, duration(ms)
        return

    if _HAS_SIMPLEAUDIO:
        try:
            wave_obj = sa.WaveObject(_BEEP_BYTES, num_channels=1, bytes_per_sample=2, sample_rate=44100)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            return
        except Exception:
            # fallback below
            pass

    if _HAS_WINSOUND and os.name == "nt":
        try:
            # winsound.PlaySound expects a filename or alias; use file path
            winsound.PlaySound(BEEP_PATH, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
        except Exception:
            pass

    # Last fallback: cannot play locally; no-op
    return

def play_beep_nonblocking():
    """Play beep in a non-blocking thread (safe to call from main loop)."""
    t = threading.Thread(target=play_beep_blocking, daemon=True)
    t.start()
    return
