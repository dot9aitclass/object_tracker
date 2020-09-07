#import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from yuvtrack import Tracker

if __name__ == '__main__' :
   # ind,acty,actu,actv=Tracker()
    def AVAR(imCrop):
        dim=imCrop.shape[0]*imCrop.shape[1]
        color=[0,0,0]
        for y in range(imCrop.shape[0]):
            for x in range(imCrop.shape[1]):
                color[0]=color[0]+imCrop[y][x][0]
                color[1]=color[1]+imCrop[y][x][1]
                color[2]=color[2]+imCrop[y][x][2]
        #cv2.imshow("meh",imCrop)
        return int(color[0]/dim),int(color[1]/dim),int(color[2]/dim),dim
    cap=cv2.VideoCapture(2)
    for i in range(20):
        _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    r = cv2.selectROI(frame)
    cv2.destroyAllWindows()
    imCrop = hsv[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    y,u,v,area=AVAR(imCrop)
    cv2.destroyAllWindows()
    print (y,u,v)

    while True:

        # Take each frame
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        lower_color = np.array([0,u-20,v-20])
        upper_color = np.array([255,u+20,v+20])        
        mask = cv2.inRange(hsv, lower_color, upper_color)
        #res = cv2.bitwise_and(frame,frame, mask= mask)
        contour,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in contour:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(mask,(x,y),(x+w,y+h),[255,255,255],2)

        #thresh=cv2.Canny(res,100,200)
        cv2.imshow('result',mask)
        if cv2.waitKey(1)==27:
            break
    cv2.destroyAllWindows()
    for cnt in contour:
        print(cv2.contourArea(cnt))

