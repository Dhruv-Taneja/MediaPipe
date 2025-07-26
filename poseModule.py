import cv2 as cv
import mediapipe as mp
import time
import numpy as np
import math

class poseDetector():
    def __init__(self,mode=False,upBody=False,smooth=True,detectionCon=0.5,trackCon=0.5):
        self.mode=mode
        self.upbody=upBody
        self.smooth=smooth
        self.detectionCon=detectionCon
        self.trackCon=trackCon

        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,model_complexity=1,smooth_landmarks=self.smooth,min_detection_confidence=self.detectionCon,min_tracking_confidence=self.trackCon
)

    def findPose(self,img,draw=True):
        imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results=self.pose.process(imgRGB)
        # print(results.pose_landmarks) 
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self,img,draw=True):
        self.lmlist=[]
        if self.results.pose_landmarks:
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c=img.shape
                # print(id,lm)
                cx,cy=int(lm.x*w),int(lm.y*h)
                self.lmlist.append([id,cx,cy])
                if draw:
                    cv.circle(img,(cx,cy),5,(255,0,0),cv.FILLED)
        return self.lmlist
    
    def findangle(self,img,p1,p2,p3,draw=True):
        #Get the coordinates of the points
        x1,y1=self.lmlist[p1][1:]
        x2,y2=self.lmlist[p2][1:]
        x3,y3=self.lmlist[p3][1:]
        #Calculate the angle
        angle=math.atan2(y3-y2,x3-x2) - math.atan2(y1-y2,x1-x2)
        angle=abs(angle*180/math.pi)
        if angle > 180:
            angle = 360 - angle
        
        if draw:
            cv.line(img,(x1,y1),(x2,y2),(0,0,0),3)
            cv.line(img,(x2,y2),(x3,y3),(0,0,0),3)
            cv.circle(img,(x1,y1),10,(0,0,0),cv.FILLED)
            cv.circle(img,(x1,y1),15,(0,0,0),2)
            cv.circle(img,(x2,y2),10,(0,0,0),cv.FILLED)
            cv.circle(img,(x2,y2),15,(0,0,0),2)
            cv.circle(img,(x3,y3),10,(0,0,0),cv.FILLED)
            cv.circle(img,(x3,y3),15,(0,0,0),2)  
            # cv.putText(img, str(int(angle)), (x2 - 50, y2 + 60), cv.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 2)
        return angle
def main():
    cap=cv.VideoCapture("/Users/dhruvtaneja/Desktop/Document/python/MediaPIpe/videos/1.mp4")
    pTime=0
    cTime=0
    detector=poseDetector()
    while True:
        success,img=cap.read()
        img=detector.findPose(img)
        lmlist=detector.findPosition(img)
        if len(lmlist) !=0:
            print(lmlist)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime

        cv.putText(img, str(int(fps)),(70,50),cv.FONT_HERSHEY_PLAIN,3,(0,0,0))

        cv.imshow("Image",img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=="__main__":
      main()
