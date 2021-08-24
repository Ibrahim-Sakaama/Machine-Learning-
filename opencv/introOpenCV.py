import cv2

#--------IMPORTANT--------------#
#---------IF YOU HAVE AN IMAGE YOU'LL TREAT IT AS A MATRIX-------------#
#----------THINK OF IT AS A MATRIX-------------------------------------#

#imread() ==> pour lire une image
A=cv2.imread("lena.jpg")

#resize() ===> resize the image
#A=cv2.resize(A,(200,200))

#demi image gauche
#shape[1] ===> columns
demi_gauche = A[:,0:A.shape[1]//2]

#demi image droit
demi_droite = A[:,A.shape[1]//2:]

#demi image dessus
#shape[0] ===> ligne
demi_dessus = A[:A.shape[0]//2,:]

#demi image dessous
demi_dessous = A[A.shape[0]//2:,:]

#cvtColor() ====> pour transformer une image couleur en gris
gray = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)

print(A)
print(A.shape)
print(gray.shape)

#imshow() ==> afficher une image
#Ibrahim is the name title
#cv2.imshow("Ibrahim",A)
cv2.imshow("Gray",gray)
cv2.imshow("Demi Gauche", demi_gauche)
cv2.imshow("Demi Droite", demi_droite)
cv2.imshow("Demi dessus", demi_dessus)
cv2.imshow("Demi Dessous",demi_dessous)
cv2.waitKey(0)

