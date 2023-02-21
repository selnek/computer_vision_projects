# opencv-python 4.7.0.68
# mediapipe 0.9.1.0
# osascript 2020.12.3

import numpy as np
import cv2
import time
import HandTrackingModule as htm
import math
import osascript

wCam, hCam = 1280, 720

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
p_time = 0
thumb_tip = 4
index_tip = 8
bbx_offset = 20

detector = htm.HandDetector(max_hands=1)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lm_list, bounding_box = detector.findPosition(img)

    if lm_list:

        # drawing fingers and line
        x1, y1 = lm_list[thumb_tip][1], lm_list[thumb_tip][2]
        x2, y2 = lm_list[index_tip][1], lm_list[index_tip][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)

        # drawing bounding_box
        cv2.rectangle(img, (bounding_box[0] - bbx_offset, bounding_box[1] - bbx_offset),
                      (bounding_box[2] + bbx_offset, bounding_box[3] + bbx_offset), (255, 255, 255), 2)

        # length of the line between fingers
        length = math.hypot(x2 - x1, y2 - y1)

        # scale volume to length
        volume = np.interp(length, [50, 300], [0, 100])

        # set volume
        osascript.osascript(f"set volume output volume {volume}")

        if length < 50:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    c_time = time.time()
    fps = 1 / (c_time - p_time)
    p_time = c_time

    img = cv2.flip(img, 1)
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX,
                2, (0, 255, 0), 2)
    cv2.namedWindow('Image', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Image", img)
    cv2.resizeWindow('Image', 900, 500)

    # close with ESC key
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
