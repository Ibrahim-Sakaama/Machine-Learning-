import pandas as pd


data = pd.read_csv("random.csv",sep=';')
print(data)
#head ==> donner les premiers 5 lignes
print(data.head(5))
#describe ===> description des donnees 
print(data.describe())
print("shape of the dataset ",data.shape)
#info ===> pour verifier s'il y a des valeurs manquantes
print(data.info)


#iloc[] ===> splitting Input and output data
X = data.iloc[:,0:12] #input
Y = data.iloc[:,-1] #output

from sklearn.model_selection import train_test_split


X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.2) 
print("XXXX ",X.shape)
print("YYYY ",Y.shape)
print("X_train", X_train.shape)
print("Y_train", Y_train.shape)
print("X_test", X_test.shape)
print("Y_test", Y_test.shape)




#----------------1st method: imposing the best estimator by the operator-----------------------

# import numpy as np
# A=np.array([[1,2,3],[4,5,6],[1,2,3],[4,5,6],[1,2,3],[4,5,6]])
# import pandas as pd
# DataFrame() ====> c'est un table/tableau
# index=[] ====> ligne
# table = pd.DataFrame(A,index=['a','b','c','d','e','f'], columns =['r','u','h'])
# print(table)
from sklearn.ensemble import RandomForestRegressor
# n_estimators=100 ==> nombres d'arbres
rfr = RandomForestRegressor(n_estimators=87)
rfr.fit(X_train,Y_train)
y_pred = rfr.predict(X_test)
import numpy as np
error = np.mean(abs(y_pred-Y_test))
print(error)
accuracy = (1-error)*100
print("precision du modele est {}%".format(accuracy))




#-------------2nd method: Trying to find/choosing the best n_estimator optimal-----------------
# import numpy as np
# from sklearn.ensemble import RandomForestRegressor
# rfr = RandomForestRegressor()
# from sklearn import model_selection
# n_estimators=[int(x) for x in np.linspace(start=80,stop=150,num=10)]
# print(n_estimators)
# random_grid = {'n_estimators':n_estimators}
# #RandomizedSearchCV() ===> return
# #estimator=rfr ====> 
# #param_distributions= random_grid ===>
# #n_iter=100 ====> faire 100 iterations 
# #cv=3 (cross validation) ====> 
# #verbose=2 ====>
# #random_state=42 ====>
# #n_jobs=-1 ====>
# rfr_random = model_selection.RandomizedSearchCV(estimator=rfr, param_distributions= random_grid, n_iter=100,cv=3,verbose=2, random_state=42,n_jobs=-1)
# rfr_random.fit(X,Y)
# #best_params_ ===> will find the best param entre les valeurs
# print(rfr_random.best_params_)












#saving the model
import pickle
#dump() ===> pour sauvegarder le modele
#dump() prend 2 parametres
# pickle.dump(rfr,open('data.pkl','wb')) #on l'execute une seule fois

#load() ===> pour importer le modele
model = pickle.load(open("data.pkl",'rb'))

predic = model.predict([[4,1,4,5,4,6,4,5,6,7,8,4]])
print(predic)


#---------------------pour voir l'importance de chaque donnee dans le prediction---------------------------


from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor(n_estimators=87)
rfr.fit(X_train,Y_train)
#feature_importances_ ===> calculer l'importance de chaque donne dans la prediction
importances=list(rfr.feature_importances_)
print(importances)
#features_list ====> columns names
features_list=list(X.columns)
print(features_list)
feature_importances=[(feature,round(importance,2)) for feature,importance in zip(features_list,importances)]
print(feature_importances)


#-------------------Putting it in a graph----------------------
feature_importances=sorted(feature_importances, reverse=True)
[print('variable:{} Importances: {}'.format(*pair)) for pair in feature_importances]

from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
x_values=list(range(len(importances)))
print(x_values)
plt.bar(x_values,importances,orientation='vertical')
plt.xticks(x_values,features_list,rotation='vertical')
plt.ylabel('Importances')
plt.xlabel('feature list')
plt.title('Variable Importance')
plt.show()



#---------comparing between real vars and predicted vars--------------
rfr = RandomForestRegressor()
rfr= model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
fig, ax = plt.subplots()
ax.plot(Y_train,'b',label='valeur réel')
ax.plot(y_pred,'r',label='valeur predict')
leg = ax.legend()
fig.savefig('Comparison.jpg')
plt.show()





