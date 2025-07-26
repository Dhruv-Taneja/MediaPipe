import cv2 as cv
import time
import os
import hand_tracking_module as htm
import numpy as np

brushThickness=15
eraserThickness=80

folderPath = "/Users/dhruvtaneja/Desktop/Document/python/MediaPipe/paint"
mylist= sorted(
    [f for f in os.listdir(folderPath) if f.endswith(('.jpg', '.jpeg', '.png'))],
    key=lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else -1
)
# print(f"Loaded {len(mylist)} paint images.")
# print(mylist)
overLayList = []
for imPath in mylist:
    img = cv.imread(os.path.join(folderPath, imPath))
    overLayList.append(img)
# print(len(overLayList))

header = overLayList[0]
drawColor=(0,0,255)
cap = cv.VideoCapture(0)
cap.set(4, 1080)
cap.set(3, 1920)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0, 0
imageCanvas = np.zeros((1080, 1920, 3), np.uint8)
while True:
    #Import image
    success,img=cap.read()
    img=cv.flip(img, 1)

    #Find hands
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False) 
    if len(lmList) !=0:
        # print(lmList)

        x1,y1= lmList[8][1:]
        x2,y2= lmList[12][1:]

        #Check which finger is up
        fingers=detector.fingersUp(lmList)
        # print(fingers)
        #Selection mode(2 fingers up)
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            # print("Selection Mode")
            cv.rectangle(img, (x1 , y1 - 25), (x2, y2 + 25),drawColor, cv.FILLED)
            if y1 < 198:
                if  400< x1 < 750:
                    header = overLayList[0]
                    drawColor = (0, 0, 255)
                elif 800 < x1 < 1150:
                    header = overLayList[1]
                    drawColor = (255, 50, 50)
                elif 1200 < x1 < 1550:  
                    header = overLayList[2]
                    drawColor = (0, 255, 0)
                elif 1600 < x1 < 1980:
                    header = overLayList[3]
                    drawColor = (0, 0, 0)
                elif 0 < x1 < 400:
                    header = overLayList[4]
                    drawColor = (255, 255, 255)


        #Drawing mode(1 finger up)
        if fingers[1] and not fingers[2]:
            # print("Drawing Mode") 
            cv.circle(img, (x1, y1), 15, drawColor, cv.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            if drawColor == (0, 0, 0):
                cv.line(img,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv.line(imageCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)

            cv.line(img,(xp,yp),(x1,y1),drawColor,brushThickness)
            cv.line(imageCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp, yp = x1, y1
    imgGray=cv.cvtColor(imageCanvas,cv.COLOR_BGR2GRAY)
    _,imgInv=cv.threshold(imgGray,50,255,cv.THRESH_BINARY_INV)
    imgInv=cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)
    img=cv.bitwise_and(img,imgInv)
    img=cv.bitwise_or(imageCanvas,img)
    #setting the header image
    img[0:198,0:1980] = header
    # img=cv.addWeighted(img, 0.5, imageCanvas, 0.5, 0)
    cv.imshow("Image", img)
    # cv.imshow("Canvas", imageCanvas) 
    if cv.waitKey(1) & 0xFF == ord('q'):
        break