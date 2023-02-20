import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

pTime = 0
cTime = 0

cap = cv2.VideoCapture(0)
detector = htm.HandDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img, draw=True)
    lmList = detector.findPosition(img, draw=True)
    if len(lmList) != 0:
        print(lmList[4])

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    img = cv2.flip(img, 1)
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 255, 0), 3)
    cv2.namedWindow('Image', cv2.WINDOW_KEEPRATIO)
    cv2.imshow("Image", img)
    cv2.resizeWindow('Image', 900, 500)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
