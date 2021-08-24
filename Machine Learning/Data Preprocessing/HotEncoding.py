#One Hot Encoding using pandas
import pandas as pd
import numpy as np

A=np.linspace(2, 10, num=6)
B=['dog','cat','bird','test','dog','bird']
d={'numeric':A, 'categorical':B}

df=pd.DataFrame(d)
print(df)

dfDummies=pd.get_dummies(df['categorical'], prefix = 'category')
print(dfDummies)
df = pd.concat([df,dfDummies],axis=1)
print(df)


#One Hot Encoding using sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

data = ['cold','cold','warm','cold','hot','hot','warm','test']
values=np.array(data)

print(values)
label_encoder = LabelEncoder()
#fit_transform() ===> il faut l'appliquer sur une matrice
#le mettre en binaire
integer_encoder=label_encoder.fit_transform(values)
print(integer_encoder)

onehot_encoder=OneHotEncoder(sparse=False)
integer_encoder = integer_encoder.reshape(len(integer_encoder),1)
print(integer_encoder)
one_encoder=onehot_encoder.fit_transform(integer_encoder)
print(one_encoder)





