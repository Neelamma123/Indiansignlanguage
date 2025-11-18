import streamlit as st
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

# --------------------------------------------
# Streamlit Page Setup
# --------------------------------------------
st.set_page_config(page_title="Sign Language Recognition", layout="wide")

st.markdown("""
# 🤘 INDIAN SIGN LANGUAGE RECOGNITION
""")

# --------------------------------------------
# Sidebar Controls
# --------------------------------------------
st.sidebar.title("⚙️ Settings")
voice_enabled = st.sidebar.checkbox("🔊 Enable Voice Output", value=True)
show_crop = st.sidebar.checkbox("📸 Show Cropped Image", value=True)
show_history = st.sidebar.checkbox("📜 Show Prediction History", value=True)

# --------------------------------------------
# Load Model + Labels
# --------------------------------------------
detector = HandDetector(detectionCon=0.35, minTrackCon=0.35, maxHands=2)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

with open("Model/labels.txt") as f:
    labels = [line.strip().split(" ")[-1] for line in f.readlines()]

# --------------------------------------------
# JavaScript Voice Output (works always)
# --------------------------------------------
def speak_js(text):
    js_code = f"""
        <script>
            var msg = new SpeechSynthesisUtterance("{text}");
            msg.rate = 1;
            msg.pitch = 1;
            speechSynthesis.speak(msg);
        </script>
    """
    st.components.v1.html(js_code, height=0)

# --------------------------------------------
# Streamlit Display Areas
# --------------------------------------------
FRAME_COL, INFO_COL = st.columns([3, 2])

frame_holder = FRAME_COL.empty()
crop_holder = FRAME_COL.empty()

pred_box = INFO_COL.empty()
conf_box = INFO_COL.empty()
fps_box = INFO_COL.empty()

history_box = INFO_COL.empty()
prediction_history = []

prev_pred = ""   # Avoid repeating voice

# --------------------------------------------
# Start Webcam Button
# --------------------------------------------
start = st.button("🚀 Start Webcam")
stop = st.button("🛑 Stop Webcam")

if "running" not in st.session_state:
    st.session_state.running = False

if start:
    st.session_state.running = True

if stop:
    st.session_state.running = False

# --------------------------------------------
# Webcam Loop
# --------------------------------------------
if st.session_state.running:

    cap = cv2.VideoCapture(0)
    prev_time = time.time()

    while st.session_state.running:
        ret, img = cap.read()
        if not ret:
            st.error("❌ Unable to access webcam")
            break

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        final_label = "No Hand"
        confidence = 0

        # -------------------------------------------------
        # HAND DETECTION (one or two hands merged)
        # -------------------------------------------------
        if hands:

            if len(hands) == 2:
                # Combine two hands into one bounding box
                x1, y1, w1, h1 = hands[0]["bbox"]
                x2, y2, w2, h2 = hands[1]["bbox"]

                x = min(x1, x2)
                y = min(y1, y2)
                w = max(x1 + w1, x2 + w2) - x
                h = max(y1 + h1, y2 + h2) - y
            else:
                x, y, w, h = hands[0]["bbox"]

            # Prepare white canvas
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size != 0:
                aspect = h / w

                if aspect > 1:
                    k = imgSize / h
                    wCal = math.ceil(w * k)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    gap = (imgSize - wCal) // 2
                    imgWhite[:, gap:gap + wCal] = imgResize

                else:
                    k = imgSize / w
                    hCal = math.ceil(h * k)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    gap = (imgSize - hCal) // 2
                    imgWhite[gap:gap + hCal, :] = imgResize

                prediction, index = classifier.getPrediction(imgWhite)
                final_label = labels[index]
                confidence = float(max(prediction) * 100)

                if show_crop:
                    crop_holder.image(cv2.cvtColor(imgCrop, cv2.COLOR_BGR2RGB))

            # Draw on frame
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 3)
            cv2.putText(imgOutput, final_label, (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

        # -------------------------------------------------
        # FPS
        # -------------------------------------------------
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now

        # -------------------------------------------------
        # Update Streamlit UI
        # -------------------------------------------------
        frame_holder.image(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB))

        pred_box.markdown(f"### 🔮 Prediction: **{final_label}**")
        conf_box.markdown(f"### 🎯 Confidence: **{confidence:.1f}%**")
        fps_box.markdown(f"### ⚡ FPS: **{int(fps)}**")

        # -------------------------------------------------
        # History
        # -------------------------------------------------
        if final_label != "No Hand":
            prediction_history.append(final_label)
            if show_history:
                history_box.write(prediction_history[-15:])

        # -------------------------------------------------
        # Browser Voice Output (very reliable)
        # -------------------------------------------------
        if (
            voice_enabled
            and final_label != "No Hand"
            and confidence > 70
            and final_label != prev_pred
        ):
            speak_js(final_label)
            prev_pred = final_label

        time.sleep(0.002)

    cap.release()
