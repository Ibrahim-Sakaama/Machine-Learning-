import cv2
import numpy as np
import PIL
from PIL import Image
import os


#---------------------------------PART 2------------------------------#



#recognizer = cv2.createLBPHFaceRecognizer()
recognizer=cv2.face.LBPHFaceRecognizer_create()
path='dataset'

def getImagesWithID(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        #convert('L') ===> convertir au niveau de gris
        faceImg=Image.open(imagePath).convert('L')
        #unsigned integer 8
        faceNP=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNP)
        print(ID)
        IDs.append(ID)
        cv2.imshow('Training',faceNP)
        cv2.waitKey(10)
    return IDs,faces

IDs,faces = getImagesWithID(path)
recognizer.train(faces,np.array(IDs))
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()
