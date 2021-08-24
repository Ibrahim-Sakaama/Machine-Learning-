import cv2





#Load the cascade
face_cascade = cv2.CascadeClassifier(r'C:\Users\hdsr\Downloads\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')

#REad the input image
img = cv2.imread('lena.jpg')

#convert into gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

#draw rectangle around the faces
for(x, y, w,h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0),2)

#Display the output
cv2.imshow('image', img)
cv2.waitKey(0)

