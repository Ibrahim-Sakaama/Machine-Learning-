import matplotlib.pyplot as plt

f=[1,4,7,9]

#plot() ===> pour le tracage des courbes
plt.plot(f)
#show() ===> pour l'affichage
#plt.show()



x = [7,8,9]
y=[5,15,14]
#plot() must have 2 arguments
plt.plot(x,y)
#naming the title,x,y axes
plt.xlabel("axe des x")
plt.ylabel("axe des y")
plt.title("le premier courbe")
#plt.show()



import numpy as np

def p1(x):
    return x**4-4*x+3*x

def p2(x):
    return np.sin(10*x)


#linspace(-3,3,200) ===> donner 200 valeurs entre -3 et 3
X = np.linspace(-3,3,200)
plt.plot(X,p1(X),X,p2(X))
#cropping the x axe that's between -1 and 1
plt.xlim(-1,1)
#cropping the y axe that's between -2 and 2
plt.ylim(-2,2)
#plt.show()


x = np.arange(0.,10,0.1)
a = np.cos(x)
b = np.sin(x)
c = np.exp(x/10)
d = np.exp(-x/10)

plt.plot(x,a,'b--',label='cosinus') #b:blue
plt.plot(x,b,'r--',label='sinus') #r:red
plt.plot(x,c,'g--',label='exp(x/10)') #g:green
plt.plot(x,d,'y--',label='exp(-x/10)') #y:yellow
plt.legend(loc="upper left")
plt.title("Courbe")
#plt.show()



labels = 'homme','femme','enfants'
sizes= [30,30,40]

#subplots() ===> retourne 2 variables, on le met par default
#subplots() ====> pour tracer des courbes en même page 
fig,ax1=plt.subplots()
#pie()  ====> tracer la courbe sous une forme d'une cercle
#autopct ===> donne les donnes en pourcentages
ax1.pie(sizes,labels=labels,autopct='%1.1f%%',startangle=90)
#'equals' ====> pour le startangle an même pt 
ax1.axis('equal')

plt.show()

a=plt.imread("lena.jpg")
print(a)
plt.imshow(a)
plt.show()


#pour afficher que lles yeux
y=a[240:276,240:357]
plt.imshow(y)
plt.show()



