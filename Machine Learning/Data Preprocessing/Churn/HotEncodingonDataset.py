import pandas as pd
import numpy as np

data=pd.read_csv(r'C:\Users\hdsr\Desktop\Internship\Machine Learning\Data Preprocessing\Churn\Churn_Modelling.csv')

X=data.iloc[0:5,1:5]
Y=data.iloc[0:5,-1]
print(X)
print(Y)


#-----------------One Hot Encoding using pandas-----------------

# print(X)
# X=pd.get_dummies(X,prefix_sep='_',drop_first=True)
# print(X)

#get_dummies() ====> Convert categorical variable into dummy/indicator variables.

Geeography=pd.get_dummies(X['Geography'], prefix = 'geography')



Surname=pd.get_dummies(X['Surname'], prefix = 'Surname')
data=pd.concat([Geeography,Surname],axis=1)

X=X.drop(['Geography','Surname'],axis=1)
X=pd.concat([X,data],axis=1)
print(X)




#-----------------One Hot Encoding using sklearn---------------

# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# data=X.iloc[:,1]
# print(data)
# values=np.array(data)
# print(values)

# label_encoder=LabelEncoder()
# integer_encoder = label_encoder.fit_transform(values)
# print(integer_encoder)

# onehot_encoder=OneHotEncoder(sparse=False)
# integer_encoder= integer_encoder.reshape(len(integer_encoder),1)
# print(integer_encoder)
# one_encoder = onehot_encoder.fit_transform(integer_encoder)
# print(one_encoder)


