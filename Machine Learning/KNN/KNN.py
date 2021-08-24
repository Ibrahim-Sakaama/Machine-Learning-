import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\hdsr\Desktop\Internship\Machine Learning\KNN\a.csv",sep=';')

X=data.iloc[:,0:3]
Y=data.iloc[:,-1]
X=np.array(X)
Y=np.array(Y)

from matplotlib import pyplot as plt

from sklearn import model_selection
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=0.3)



from sklearn import neighbors,metrics
#------------Pour cherhcer la valeur de K (1st method)------------------
#------------- executer une seulr fois---------------------
# param_grid={'n_neighbors':[3,5,7,9]}
from sklearn import model_selection
# # Créer un classifieur kNN avec recherche d'hyperparamètre par validation croisée
#GridSearchCV ===> Pour chercher la valeur de K
# clf = model_selection.GridSearchCV(neighbors.KNeighborsClassifier(),param_grid) # un classifieur kNN
# param_grid, # hyperparamètres à tester
# cv=5, # nombre de folds de validation croisée

# )

# # Optimiser ce classifieur sur le jeu d'entraînement
# # clf.fit(X_train, y_train)
# clf.fit(X_train, y_train)

# # Afficher le(s) hyperparamètre(s) optimaux
# print ("Meilleur(s) hyperparamètre(s) sur le jeu d'entraînement:",)
# print (clf.best_params_)



#------------Pour chercher la valeur de K (2nd method)------------------


#---------------il ne faut pas calculer l'erreur, on va chercher la courbe AUC-----------
k=[3,5,7,9]
error =[]
from sklearn import neighbors

for i in k:
    knn = neighbors.KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    error.append(np.mean(abs(y_pred - y_test)))
print(error)

print(k[error.index(min(error))])
# error = 1-knn.score(X_test, y_test)
# print("Error: %f" % error)


from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=9)

knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)





#-------------------Construire la courbe ROC du modele optimise---------------
#fpr ===> false positive rate, la sensibilité
#tpr ===> true positive rate, la specificité
#on peut enlever 'cv' a la fin du nom des vars, ca change rien
# fpr_cv, tpr_cv, thr_cv = metrics.roc_curve(y_test, y_pred)

#Calculer l'aire sous la Courbe ROC du modele optimise
auc_cv = metrics.auc(fpr_cv, tpr_cv)


from matplotlib import pyplot as plt


#creer une figure
fig = plt.figure(figsize=(12,6))

#afficher la courbe ROC precedente
plt.plot(fpr_cv, tpr_cv, '-', lw=2, label='AUC=%.2f'%auc_cv)

plt.plot([0,1],[0,1],'r--')

plt.xlabel('False Positive Rate', fontsize=16)
plt.ylabel('True Positive Rate', fontsize=16)
plt.title('KNN', fontsize=16)

#afficher la legende
plt.legend(loc="lower right", fontsize=14)


#affciher l'image
plt.show()