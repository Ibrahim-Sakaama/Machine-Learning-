import cv2

#un video est une ensemble de photos
#VideoCapture(0) ===> lorqsu'on travaille avec cam du pc
#VideoCapture(1) ===> lorqsu'on travaille avec cam exterieur
cap = cv2.VideoCapture(0)

#Load the cascade
face_cascade = cv2.CascadeClassifier(r'C:\Users\hdsr\Downloads\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')

while(True):
   
    #ret var boolean has to be always True
    #convert into gray scale
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    #detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    #draw rectangle around the faces
    for(x, y, w,h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0),2)


    

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

#jusqu'a lundi

#systeme de pointage avec la reconnaissance faciale en utilisaant la biblotheque face_recognition
#il faut sauvegarde dans une base de donees (mongodb) apres la reconnaissance le nom, la date d'entree(jour,heure,minutes,secondes)
#datatime