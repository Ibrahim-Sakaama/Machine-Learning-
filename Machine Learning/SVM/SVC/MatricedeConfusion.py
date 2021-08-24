import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\hdsr\Desktop\Internship\Machine Learning\SVM\SVC\a.csv",sep=';')

X=data.iloc[:,0:3]
Y=data.iloc[:,-1]
X=np.array(X)
Y=np.array(Y)



from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.3)

from sklearn import svm




#--------------Pour le noyeau lineaire--------------

#random_state = 1 ==> pour prendre les meme vals de X_train et Y_train
classifier = svm.SVC(kernel='linear', random_state=1)
classifier.fit(X_train,Y_train)
y_pred=classifier.predict(X_test)

from sklearn import metrics
matrice = metrics.confusion_matrix(Y_test,y_pred)
print(matrice)




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















