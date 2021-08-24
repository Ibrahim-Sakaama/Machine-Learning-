import pandas as pd

donnees = pd.read_csv(r"C:\Users\hdsr\Desktop\Internship\Machine Learning\Logistic Regression\a.csv",sep=';')

X=donnees.iloc[:,:-1]

import numpy as np
X=np.array(X)
y=donnees.iloc[:,-1]
y=y[:,np.newaxis]

x=np.c_[np.ones((X.shape[0])),X]
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
# 8 ===> X1
# 9 ===> X2
# 7 ===> X3
resultat = 1/(1+math.exp(-(a[0]+8*a[1]+9*a[2]+7*a[3])))
print(resultat)
if(resultat>0.5):
    print('classe 1')
else:
    print('classe 2')


X=X[0:20,:]
y=y[0:20]
print(y)

y_pred=[]
for i in X:
    resultat= 1/(1+math.exp((-(a[0]+i[0]*a[1]+i[1]*a[2]+a[3]*i[2]))))
    print(resultat)
    if(resultat>0.5):
        y_pred.append(1)
    else:
        y_pred.append(0)
print(y_pred)
from sklearn import metrics
print(metrics.accuracy_score(y_pred,y))





