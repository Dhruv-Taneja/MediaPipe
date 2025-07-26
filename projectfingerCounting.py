import cv2
import time
import os
import hand_tracking_module as htm

cap = cv2.VideoCapture(0)

overlay_w, overlay_h = 320, 240

folderPath = "/Users/dhruvtaneja/Desktop/Document/python/MediaPipe/handImages"

def numeric_sort(filename):
    digits = ''.join(filter(str.isdigit, filename))
    return int(digits) if digits else -1  # fallback if no number is present

myList = sorted(
    [f for f in os.listdir(folderPath) if f.endswith(('.jpg', '.jpeg', '.png'))],
    key=numeric_sort
)

overlayList = [
    cv2.imread(os.path.join(folderPath, imPath))
    for imPath in myList
    if cv2.imread(os.path.join(folderPath, imPath)) is not None
]
print(f"Loaded {len(overlayList)} overlay images.")

detector = htm.handDetector(detectionCon=0.75)
tipIds = [4, 8, 12, 16, 20]
pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=True)

    if lmList:
        fingers = [
            1 if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1] else 0
        ] + [
            1 if lmList[tipIds[i]][2] < lmList[tipIds[i] - 2][2] else 0
            for i in range(1, 5)
        ]
        totalFingers = fingers.count(1)

        if 0 <= totalFingers < len(overlayList):
            overlay = cv2.resize(overlayList[totalFingers], (overlay_w, overlay_h))
            img[0:overlay_h, 0:overlay_w] = overlay

        cv2.rectangle(img, (20, 275), (170, 475), (211, 255, 211), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 420),
                    cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70),
                cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()