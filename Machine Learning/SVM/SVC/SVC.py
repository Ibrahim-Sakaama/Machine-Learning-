import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\hdsr\Desktop\Internship\Machine Learning\SVM\SVC\a.csv",sep=';')

X=data.iloc[:,0:3]
Y=data.iloc[:,-1]
X=np.array(X)
Y=np.array(Y)

from matplotlib import pyplot as plt
#scatter ===> Pour faire la distribution des pts
plt.scatter(X[Y==0][:,0],X[Y==0][:,1],X[Y==0][:,2],color='r',label='0')
plt.scatter(X[Y==1][:,0],X[Y==1][:,1],X[Y==1][:,2],color='b',label='1')
plt.legend()
plt.show()

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3)

from sklearn import svm

#-----------------rbf ===> noyeau gossien------------------
#kernel c'est le noyeux
#rbf ===> noyeau gossien
#random_state ===> 
classifier = svm.SVC(kernel='rbf', gamma=0.01, random_state=1)

#Choisir 6 valeurs pour C, entre 10^(-2) et 10^(3)
C_range = np.logspace(-2,3,6)

#choisir 4 valeurs pour gamma, entre 10^(-2) et 10^(1)
gamma_range = np.logspace(-2,1,4)

#grille de parametres
param_grid = {'C':C_range, 'gamma': gamma_range}

from sklearn import model_selection

#initialiser une recherche sur grille
grid = model_selection.GridSearchCV(svm.SVC(kernel='rbf'),
                                    param_grid,
                                    cv=5 #5 fois de validation croisée (nombre de repetition de chaque)
                                    )

#faire tourner la recherche sur grille
grid.fit(X_train, Y_train)


#affciher les parametres optimaux
print((grid.best_params_))


#--------------Pour le noyeau lineaire--------------


classifier = svm.SVC(kernel='linear', gamma=0.01, random_state=1)

#Choisir 6 valeurs pour C, entre 10^(-2) et 10^(3)
C_range = np.logspace(-2,3,6)



#grille de parametres
param_grid = {'C':C_range}

from sklearn import model_selection

#initialiser une recherche sur grille
grid = model_selection.GridSearchCV(svm.SVC(kernel='linear'),
                                    param_grid,
                                    cv=5 #5 fois de validation croisée (nombre de repetition de chaque)
                                    )

#faire tourner la recherche sur grille
grid.fit(X_train, Y_train)


#affciher les parametres optimaux
print((grid.best_params_))




#------------Pour le noyeau polinomial-----------------


classifier = svm.SVC(kernel='poly', degree=4, random_state=1)

#Choisir 6 valeurs pour C, entre 10^(-2) et 10^(3)
C_range = np.logspace(-2,3,6)

#choisir 10 valeurs entre 1 et 10 
degree_range = np.linspace(1,10,10)

#grille de parametres
param_grid = {'C':C_range, 'degree': degree_range}

from sklearn import model_selection

#initialiser une recherche sur grille
grid = model_selection.GridSearchCV(svm.SVC(kernel='poly'),
                                    param_grid,
                                    cv=3 #5 fois de validation croisée (nombre de repetition de chaque)
                                    )

#faire tourner la recherche sur grille
grid.fit(X_train, Y_train)


#afficher les parametres optimaux
print((grid.best_params_))








