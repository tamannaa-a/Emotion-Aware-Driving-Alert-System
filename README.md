
# 🚗 Emotion-Aware Driving Alert System (Drowsiness Detection)

This project is an **AI-powered driver drowsiness detection system** that monitors the driver's eyes in real time using a webcam.  
If the driver’s eyes remain closed for a long period (indicating sleepiness), the system **triggers an audible alert (beep)** to prevent accidents.

---

## 🧠 Overview

The system uses **MediaPipe FaceMesh** to detect facial landmarks and computes the **Eye Aspect Ratio (EAR)** to determine whether the eyes are open or closed.  
When the eyes remain closed for several consecutive frames, it plays a **beep sound alert** using `winsound`.

This system can serve as a foundation for **in-vehicle fatigue monitoring**, **driver assistance**, and **safety alert systems**.

---

## ⚙️ Features

✅ Real-time face and eye detection using webcam  
✅ EAR (Eye Aspect Ratio)–based drowsiness detection  
✅ Continuous beep alert when eyes are closed for too long  
✅ Works on Windows (using `winsound`)  
✅ Adjustable sensitivity and thresholds  
✅ Streamlit-based clean web interface  
✅ Lightweight and deployable on Render or local systems  

---

## 🧩 Tech Stack

| Component | Technology Used |
|------------|-----------------|
| **Frontend UI** | Streamlit |
| **Backend Logic** | Python + OpenCV + MediaPipe |
| **Audio Alert** | Winsound (for Windows) |
| **Face Landmark Detection** | MediaPipe FaceMesh |
| **Visualization** | Streamlit + OpenCV Frames |
| **Language** | Python 3.10+ |

---

## 🧮 Working Principle

1. **Camera Capture:** The webcam captures real-time frames.
2. **RGB Conversion:** The frame is converted from BGR → RGB for MediaPipe processing.
3. **Face Detection:** MediaPipe FaceMesh detects face landmarks.
4. **Eye Landmarks:** Eye points are extracted from the face mesh.
5. **EAR Calculation:** The Eye Aspect Ratio (EAR) is computed for both eyes.
6. **Drowsiness Detection:**  
   - If EAR < threshold for several consecutive frames → eyes considered **closed**.  
   - When eyes stay closed beyond the preset limit → **beep alarm is triggered**.
7. **Alert Generation:** Continuous beeps sound until the eyes open again.

---

## 🧠 Formula Used

**Eye Aspect Ratio (EAR):**

\[
EAR = \frac{||p2 - p6|| + ||p3 - p5||}{2 \times ||p1 - p4||}
\]

Where:
- p1–p6 are the 6 key landmarks around the eye.

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/tamannaa-a/Drowsiness-Detection.git
cd Drowsiness-Detection
