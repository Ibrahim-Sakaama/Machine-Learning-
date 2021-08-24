import numpy as np
import cv2


#---------------------------PART 1 -------------------------------------------#


id = input("Donner user ID:  ")
faceDetect = cv2.CascadeClassifier(r"C:\Users\hdsr\Downloads\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
#compteur pour le nbrs des images
sampleNum = 0

while(1):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #1.3 et 5 on les mets par default
    #5 ===> nbrs de faces à detecter
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y), (x+w,y+h),(0,0,255),2)
        sampleNum += 1

        #saving the captured face in the dataset folder
        cv2.imwrite("dataset/User."+str(id)+'.'+str(sampleNum)+".png",gray[y:y+h,x:x+w])
        cv2.waitKey(100)
    cv2.imshow('reconnaissance faciale',img)
    cv2.waitKey(100)

    #break if the sample number is more than 30
    if(sampleNum>30):
        break

cam.release()
cv2.destroyAllWindows() 