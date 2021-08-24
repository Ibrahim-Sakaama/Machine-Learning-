import pandas as pd
import numpy as np

data = pd.read_csv(r"C:\Users\hdsr\Desktop\Internship\Machine Learning\Logistic Regression\a.csv",sep=';')

X= data.iloc[:,:-1]
y=data.iloc[:,-1]

#C=[] ===> concat()
x=np.c_[np.ones((X.shape[0])),X]
print(x)
y=y[:,np.newaxis]
print(y)

theta = np.zeros((X.shape[1]))


#definition de la fonction sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))

print(sigmoid(x))

def net_input(theta,X):
    return np.dot(X,theta)

print(net_input(theta, X))

def probability(theta,x):
    return sigmoid(net_input(theta,X))

print(probability(theta, X))

#function cost ===> minimiser l'erreur/ obtimisation

def cost_function(theta,x,y):
    m = x.shape[0]
    total_cost = - (1/m) * np.sum(y*np.log(probability(theta,x))+(1-y)*np.log(1-probability(theta, x)))
    return total_cost


#la fonction gradient est la derive du fonction cost
def gradient(theta,x,y):
    m=x.shape[0]
    #x.T ====> le transpose
    return (1/m)*np.dot(x.T,sigmoid(net_input(theta,x))-y)


from scipy.optimize import fmin_tnc

def fit(x,y, theta):
    opt_weight = fmin_tnc(func=cost_function,x0=theta,fprime=gradient,args=(x,y.flatten()))
    return opt_weight[0]

a=fit(X,y,theta)
print(a)



