import numpy as np
import pandas as pd



data = pd.read_excel(r"C:\Users\hdsr\Desktop\Internship\Machine Learning\Weather Random Forest\Data2017_ET0.xlsx")
print(data)
# #head ==> donner les premiers 5 lignes
print(data.head(5))
# #describe ===> description des donnees 
# print(data.describe())
# print("shape of the dataset ",data.shape)
# #info ===> pour verifier s'il y a des valeurs manquantes
# print(data.info)

#data cleaning/ nettoyage du data

year=data.iloc[:,1]
print(year)
print(type(year))

df = pd.DataFrame({'Date':pd.to_datetime(year)})
print(df)
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['hour'] = df['Date'].dt.hour
print(df)

#to drop row Date
df.drop(['Date'],axis=1, inplace=True)
data.drop(['Date'],axis=1,inplace=True)
data.drop(['Unnamed: 0'],axis=1,inplace=True)

#pour supprimer une ligne
#df.drop(0,axis=0,inplace=True)
print(df)
print(data)



#for DataFrame we use concat()
#for matrices we use concatinate()
#concatinate df and data

result = pd.concat([df,data], axis=1)
print(result)
print(result.columns)


#------------------------pour tracer une graphe du distribution (qui represente les valeurs NAN)
import seaborn as sns
#cmap='viridis' ===> Pour le couleur
#cbar=False ===> le module qui va chercher les vals NAN 
sns.heatmap(result.isnull(),yticklabels=False,cbar=False,cmap='viridis')
print(result.isnull().sum())
from matplotlib import pyplot as plt
plt.show()
#sns.barplot(x = "names", y = "values", data = result)

result['Year'].mean()
result['Year'].fillna(2017, inplace=True)
result['ET0 aver'].fillna(result['ET0 aver'].mean(), inplace=True)
#bech tala3 l result lkol 0
print(result.isnull().sum())




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



