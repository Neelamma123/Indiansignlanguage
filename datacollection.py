import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize camera and hand detector
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # detect up to 2 hands

offset = 20
imgSize = 300
folder = "Data/Yes"

# create folder if not exist
if not os.path.exists(folder):
    os.makedirs(folder)

counter = 0

while True:
    success, img = cap.read()
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
    hands, img = detector.findHands(img)  # detect one or two hands

    if hands:
        # Get combined bounding box of both hands
        x_min = min([hand['bbox'][0] for hand in hands])
        y_min = min([hand['bbox'][1] for hand in hands])
        x_max = max([hand['bbox'][0] + hand['bbox'][2] for hand in hands])
        y_max = max([hand['bbox'][1] + hand['bbox'][3] for hand in hands])

        # Final bounding box for both hands together
        x, y, w, h = x_min, y_min, x_max - x_min, y_max - y_min

        # Create white image for resizing
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Crop the region with both hands (with offset)
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size == 0:
            continue

        aspectRatio = h / w

        # Resize proportionally to fit both hands in white background
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

        # Display cropped and processed images
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == ord("s") and hands:
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved Image {counter}")

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
