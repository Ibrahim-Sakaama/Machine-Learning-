import numpy as np
import cv2
#pour le temps
import time


#-----------------------------PART 3------------------------#


faceDetect=cv2.CascadeClassifier(r'C:\Users\hdsr\Downloads\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer/trainingData.yml')
id=0
#if you're going to use the app IP webcam
#import urllib.request
#url="PUT URL HERE"


import pyrebase

config = {
  "apiKey": "AIzaSyDDxNgYjV_pWjC40BiUnECtno1iZZ-BzLg",
  "authDomain": "facerecog-70287.firebaseapp.com",
  "databaseURL": "https://facerecog-70287-default-rtdb.firebaseio.com",
  "storageBucket": "facerecog-70287.appspot.com",
  #"serviceAccount": "path/to/serviceAccountCredentials.json"
}

firebase = pyrebase.initialize_app(config)

# Get a reference to the auth service
auth = firebase.auth()

# Log the user in
user = auth.sign_in_with_email_and_password("admin@gmail.com", "adminadmin")

# Get a reference to the database service
db = firebase.database()







while(1):
    #UNCOMMENT THIS IF YOU'RE going to use the app
    # #use urllib to get image from the IP camera
    # imgResponse = urllib.request.urlopen(url)
    # #Numpy to convert into array
    # img = np.array(bytearray(imgResponse.read()),dtype=np.uint8)
    # #decode the array to Opencv usable format
    # img = cv2.imdecode(img, -1)

    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=recognizer.predict(gray[y:y+h,x:x+w])
        if conf<=70:
            if(id==1):
                id='ibrahim'
        else:
            id='unknown'

            
        # data to save
        data = {
        "name": id,
        "time":time.ctime(),
        "confidence": conf
        }   

        # Pass the user's idToken to the push method
        results = db.push(data, user['idToken'])
        

        cv2.putText(img,str(id),(x,y+2),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
        cv2.putText(img,str(conf),(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
        #cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255)
    cv2.imshow('reconnaissance faciale',img)
    if(cv2.waitKey(1)==ord('q')):
        break

cam.release()
cv2.destroyAllWindows()


