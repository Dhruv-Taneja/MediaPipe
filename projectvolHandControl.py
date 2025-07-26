import cv2 as cv
import time
import numpy as np
import hand_tracking_module as htm
import math
import os

cap = cv.VideoCapture(0)
ptime = 0
ctime = 0

detector = htm.handDetector()
volume = 0
def set_volume_mac(volume_percent):
    """Set system volume on macOS using AppleScript."""
    volume_percent = max(0, min(100, int(volume_percent)))
    os.system(f"osascript -e 'set volume output volume {volume_percent}'")

print("macOS system volume control enabled via AppleScript")

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index tip
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv.circle(img, (x1, y1), 15, (255, 0, 255), cv.FILLED)
        cv.circle(img, (x2, y2), 15, (255, 0, 255), cv.FILLED)
        cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv.circle(img, (cx, cy), 15, (255, 0, 255), cv.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        # print(int(length))

        # Map distance to volume level (0–100)
        volume = np.interp(length, [40, 400], [0, 100])
        volume = int(volume)
        print(int(length), volume)
        set_volume_mac(volume)

        if length < 50:
            cv.circle(img, (cx, cy), 15, (0, 255, 0), cv.FILLED)
    cv.rectangle(img, (50, 90), (85, 440), (0, 0, 0), 3)
    cv.rectangle(img, (50, 40+int(400 - (volume * 3.5))), (85, 440), (0, 255, 0), cv.FILLED)   
    cv.putText(img, f'Volume: {volume}%', (10, 500), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv.putText(img, f'FPS: {int(fps)}', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv.imshow("Image", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()