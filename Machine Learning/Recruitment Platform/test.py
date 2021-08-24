import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\hdsr\Desktop\Internship\Machine Learning\Recruitment Platform\dataset.csv",sep=';')

x= data.iloc[:,:-1]
y=data.iloc[:,-1]


#--------------------Model Creation-----------------------

#-------------------------SVC-----------------------------


from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(x, y, test_size=0.3)
from sklearn import neighbors
knn = neighbors.KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,Y_train)
y_pred=knn.predict(X_test)
