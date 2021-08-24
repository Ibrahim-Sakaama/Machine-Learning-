import cv2
import matplotlib.pyplot as plt

#lire image
#0 ==>read color image then convert it to gay mode
img = cv2.imread("tiger.jpg",0)

#-------------instead of 0 we can pass by these functions-----------------
#loads a color image
#cv2.IMREAD_COLOR

#loads image in gray scale mode
#cv2.IMREAD8GRAYSCALE

#loads image as such including alpha channel
#cv2.IMREAD_UNCHANGED

#afficher l'image
cv2.imshow('TigerNB',img)



#attendre pour que l'utilisteur appuye sur une touche
cv2.waitKey(0)

#detruire les fenetres 

cv2.DestroyAllWindows()