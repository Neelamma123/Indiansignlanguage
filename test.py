import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Initialize camera and modules
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # detect up to two hands
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

# ✅ Read labels directly from your labels.txt file
with open("Model/labels.txt") as f:
    labels = [line.strip().split(' ')[-1] for line in f.readlines()]

print(f"✅ Loaded {len(labels)} labels: {labels}")

while True:
    success, img = cap.read()
    if not success:
        print("⚠ Camera not accessible!")
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img)  # Detect hands

    if hands:
        # ✅ If two hands detected → combine them into one region
        if len(hands) == 2:
            x1, y1, w1, h1 = hands[0]['bbox']
            x2, y2, w2, h2 = hands[1]['bbox']

            # Find one bounding box that covers both hands
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y
        else:
            # Only one hand
            x, y, w, h = hands[0]['bbox']

        # ✅ Create white background
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the region covering one or both hands
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            aspectRatio = h / w

            # Fit the cropped image into white background
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # ✅ Get prediction for combined gesture
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

            if 0 <= index < len(labels):
                label = labels[index]
                # Draw prediction on the output image
                cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                              (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
                cv2.putText(imgOutput, label, (x, y - 26),
                            cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
                cv2.rectangle(imgOutput, (x - offset, y - offset),
                              (x + w + offset, y + h + offset), (255, 0, 255), 4)
            else:
                print(f"⚠ Predicted index {index} out of range for labels length {len(labels)}")

            # Optional: show intermediate views
            cv2.imshow("CombinedCrop", imgCrop)
            cv2.imshow("WhiteBG", imgWhite)

    # Display final output
    cv2.imshow("Image", imgOutput)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()