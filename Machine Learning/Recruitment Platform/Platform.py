import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\hdsr\Desktop\Internship\Machine Learning\Recruitment Platform\dataset.csv",sep=';')

x= data.iloc[:,:-1]
y=data.iloc[:,-1]


#--------------------Model Creation-----------------------

#-------------------------SVC-----------------------------


from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=0.3)

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




# #--------------Pour le noyeau lineaire--------------


# classifier = svm.SVC(kernel='linear', gamma=0.01, random_state=1)

# #Choisir 6 valeurs pour C, entre 10^(-2) et 10^(3)
# C_range = np.logspace(-2,3,6)



# #grille de parametres
# param_grid = {'C':C_range}

# from sklearn import model_selection

# #initialiser une recherche sur grille
# grid = model_selection.GridSearchCV(svm.SVC(kernel='linear'),
#                                     param_grid,
#                                     cv=5 #5 fois de validation croisée (nombre de repetition de chaque)
#                                     )

# #faire tourner la recherche sur grille
# grid.fit(X_train, Y_train)


# #affciher les parametres optimaux
# print((grid.best_params_))





# #------------Pour le noyeau polinomial-----------------


# classifier = svm.SVC(kernel='poly', degree=4, random_state=1)

# #Choisir 6 valeurs pour C, entre 10^(-2) et 10^(3)
# C_range = np.logspace(-2,3,6)

# #choisir 10 valeurs entre 1 et 10 
# degree_range = np.linspace(1,10,10)

# #grille de parametres
# param_grid = {'C':C_range, 'degree': degree_range}

# from sklearn import model_selection

# #initialiser une recherche sur grille
# grid = model_selection.GridSearchCV(svm.SVC(kernel='poly'),
#                                     param_grid,
#                                     cv=3 #5 fois de validation croisée (nombre de repetition de chaque)
#                                     )

# #faire tourner la recherche sur grille
# grid.fit(X_train, Y_train)


# #afficher les parametres optimaux
# print((grid.best_params_))

#-------------------------LAST THNIG TO DO IS SAVING THE MODEL----------------------------
#-------------------------saving the model---------------------------------------------
import pickle
# #dump() ===> pour sauvegarder le modele
# #dump() prend 2 parametres
pickle.dump(rbf,open('rbf.pkl','wb')) #on l'execute une seule fois

# #load() ===> pour importer le modele
#model = pickle.load(open("rbf.pkl",'rb'))

#--------------------------------------KNN--------------------------------------------


#------------Pour chercher la valeur de K (1st method)------------------
#------------- executer une seule fois----------------------------------
# from sklearn import neighbors,metrics
# param_grid={'n_neighbors':[3,5,7,9]}
# from sklearn import model_selection
# # # Créer un classifieur KNN avec recherche d'hyperparamètre par validation croisée
# #GridSearchCV ===> Pour chercher la valeur de K
# clf = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(),param_grid) # un classifieur kNN
# # param_grid, # hyperparamètres à tester
# # cv=5, # nombre de folds de validation croisée


# # Optimiser ce classifieur sur le jeu d'entraînement
# # clf.fit(X_train, y_train)
# clf.fit(X_train, Y_train)

# # Afficher le(s) hyperparamètre(s) optimaux
# print("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:")
# print(clf.best_params_)


#RESULT: {'n_neighbors': 3}

# k=[3,5,7,9]
# error =[]
# from sklearn import neighbors

# for i in k:
#     knn = neighbors.KNeighborsClassifier(n_neighbors=i)
#     knn.fit(X_train, Y_train)
#     y_pred = knn.predict(X_test)
#     error.append(np.mean(abs(y_pred - Y_test)))
# print(error)

# print(k[error.index(min(error))])
# error = 1-knn.score(X_test, y_test)
# print("Error: %f" % error)


from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,Y_train)
y_pred=knn.predict(X_test)
print(y_pred)

# [0.0, 0.0, 0.0, 0.0]
# 3

#-------------------------LAST THNIG TO DO IS SAVING THE MODEL----------------------------
#-------------------------saving the model---------------------------------------------
import pickle
# #dump() ===> pour sauvegarder le modele
# #dump() prend 2 parametres
pickle.dump(knn,open('knn.pkl','wb')) #on l'execute une seule fois

# #load() ===> pour importer le modele
#model = pickle.load(open("knn.pkl",'rb'))

#--------------------------LINEAR REGRESSION---------------------------------
x=np.c_[np.ones((x.shape[0])),x]
y=y[:,np.newaxis]
theta=np.zeros((x.shape[1]))
print(theta)
def sigmoid(x):
    return 1/(1+np.exp(-x))
def net_input(theta,x):
    return np.dot(x,theta)
def probabilite(theta,x):
    return sigmoid(net_input(theta,x))
def cost_function(theta,x,y):
    m=x.shape[0]
    total_cost=(-1/m)*np.sum(y*np.log(probabilite(theta,x))+(1-y)*np.log(1-probabilite(theta,x)))
    return total_cost
def gradient(theta,x,y):
    m=x.shape[0]
    return (1/m)*np.dot(x.T,sigmoid(net_input(theta,x))-y)
def fit(x,y,theta):
    opt_weight=fmin_tnc(func=cost_function,x0=theta,fprime=gradient,args=(x,y.flatten()))
    return opt_weight[0]
from scipy.optimize import fmin_tnc
a=fit(x,y,theta)
print(a)




#tester la precision du modele


import math
resultat = 1/(1+math.exp(-(a[0]+8*a[1]+9*a[2]+7*a[3])))
print(resultat)
if(resultat>0.5):
    print('classe 1')
else:
    print('classe 2')




y_pred=[]
for i in x:
    print(i)
    resultat= 1/(1+math.exp((-(a[0]+i[0]*a[1]+i[1]*a[2]+a[3]*i[2]))))
    print(resultat)
    if(resultat>0.5):
        y_pred.append(1)
    else:
        y_pred.append(0)
print(y_pred)
from sklearn import metrics
print(metrics.accuracy_score(y_pred,y))

#-------------------------LAST THNIG TO DO IS SAVING THE MODEL----------------------------
#-------------------------saving the model---------------------------------------------
import pickle
# #dump() ===> pour sauvegarder le modele
# #dump() prend 2 parametres
pickle.dump(lr,open('lr.pkl','wb')) #on l'execute une seule fois

# #load() ===> pour importer le modele
#model = pickle.load(open("knn.pkl",'rb'))




#---------------------------RANDOM FOREST CLASSIFIER--------------------------



import numpy as np
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
from sklearn import model_selection
n_estimators=[int(x) for x in np.linspace(start=80,stop=150,num=10)]
print(n_estimators)
random_grid = {'n_estimators':n_estimators}
# #RandomizedSearchCV() ===> return
# #estimator=rfr ====> 
# #param_distributions= random_grid ===>
# #n_iter=100 ====> faire 100 iterations 
# #cv=3 (cross validation) ====> 
# #verbose=2 ====>
# #random_state=42 ====>
# #n_jobs=-1 ====>
# rfr_random = model_selection.RandomizedSearchCV(estimator=rfr, param_distributions= random_grid, n_iter=100,cv=3,verbose=2, random_state=42,n_jobs=-1)
# rfr_random.fit(x,y)
# #best_params_ ===> will find the best param entre les valeurs
# print(rfr_random.best_params_)


#RESULT:   {'n_estimators': 80}



rfr = RandomForestRegressor(n_estimators=80)
rfr.fit(X_train,Y_train)
y_pred = rfr.predict(X_test)
import numpy as np
error = np.mean(abs(y_pred-Y_test))
print(error)
accuracy = (1-error)*100
print("precision du modele est {}%".format(accuracy))

#result
# [80, 87, 95, 103, 111, 118, 126, 134, 142, 150]
# 0.0
# precision du modele est 100.0%

#-------------------------LAST THNIG TO DO IS SAVING THE MODEL----------------------------
#-------------------------saving the model---------------------------------------------
import pickle
# #dump() ===> pour sauvegarder le modele
# #dump() prend 2 parametres
pickle.dump(rfr,open('rfr.pkl','wb')) #on l'execute une seule fois

# #load() ===> pour importer le modele
#model = pickle.load(open("rfr.pkl",'rb'))


#----------------------------------MATRICE DE CONFUSION--------------------------
#Construire la courbe ROC du modele optimise
#fpr ===> false positive rate, la sensibilité
#tpr ===> true positive rate, la specificité
#on peut enlever 'cv' a la fin du nom des vars, ca change rien
fpr_cv, tpr_cv, thr_cv = metrics.roc_curve(Y_test, y_pred)

#Calculer l'aire sous la Courbe ROC du modele optimise
auc_cv = metrics.auc(fpr_cv, tpr_cv)

from matplotlib import pyplot as plt


#creer une figure
fig = plt.figure(figsize=(12,6))

#afficher la courbe ROC precedente
plt.plot(fpr_cv, tpr_cv, '-', lw=2, label='AUC=%.2f'%auc_cv)

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('SVM LINEAR', fontsize=16)

#afficher la legende
plt.legend(loc="lower right", fontsize=14)


#affciher l'image
plt.show()


# #-------------------------LAST THNIG TO DO IS SAVING THE MODEL----------------------------
# #-------------------------saving the model---------------------------------------------
# import pickle
# # #dump() ===> pour sauvegarder le modele
# # #dump() prend 2 parametres
# pickle.dump(rfr,open('data.pkl','wb')) #on l'execute une seule fois

# # #load() ===> pour importer le modele
# model = pickle.load(open("data.pkl",'rb'))

