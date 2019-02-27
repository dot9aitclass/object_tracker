import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
 
if __name__ == '__main__' :
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
    cap=cv2.VideoCapture(0)
    for i in range(20):
        _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    r = cv2.selectROI(frame)
    cv2.destroyAllWindows()
    imCrop = hsv[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    y,u,v,area=AVAR(imCrop)
    while True:

        # Take each frame
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        print(y,u,v)
        lower_color = np.array([0,u-20,v-20])
        upper_color = np.array([255,u+20,v+20])        
        mask = cv2.inRange(hsv, lower_color, upper_color)
        res = cv2.bitwise_and(frame,frame, mask= mask)

        #thresh=cv2.Canny(res,100,200)
        cv2.imshow('result',mask)
        if cv2.waitKey(1)==27:
            break
    cv2.destroyAllWindows()
    year = [1960, 1970, 1980, 1990, 2000, 2010]
    pop_pakistan = [44.91, 58.09, 78.07, 107.7, 138.5, 170.6]
    pop_india = [449.48, 553.57, 696.783, 870.133, 1000.4, 1309.1]
    plt.plot(year, pop_pakistan, color='g')
    plt.plot(year, pop_india, color='orange')
    plt.xlabel('Countries')
    plt.ylabel('Population in million')
    plt.title('Pakistan India Population till 2010')
    plt.show()
