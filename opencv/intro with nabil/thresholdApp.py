import numpy as np
import cv2

#bech thel l cam
cap = cv2.VideoCapture(0)

while(1):

    #take each frame
    #cap.read() ===> bech tlansi l video
    _, frame = cap.read()

    #convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #define range of blue color in hsv
    lower_blue = np.array([110,100,100])
    upper_blue = np.array([130,255,255])

    #treshold the HSV image to get only the color blue
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    #Bitwise-AND mask and original image
    #bitwise_and takes 2 frame
    res = cv2.bitwise_and(frame, frame, mask = mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.DestroyAllWindows()