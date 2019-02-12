# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML # used to print out pretty pandas dataframes

import matplotlib.dates as dates
import matplotlib.lines as mlines

import sklearn.decomposition
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from patsy import dmatrices
import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.discrete.discrete_model as sm

import numba
import numpy as np
import argparse
import time

import profile

%load_ext memory_profiler
%matplotlib inline 


#Load data

#look at all admissions records, pulled from PostGreSQL database
admissions = pd.read_csv('sepsisPatients.csv', index_col=0)
admissions.head()

#Hotcode insurance types
df = pd.concat([admissions, pd.get_dummies(admissions['insurance'])], axis=1)

#fix spacing in names
df['Self_Pay']=df['Self Pay']

#Hotcode gender
dfFinal = pd.concat([df, pd.get_dummies(df['gender'])], axis=1)

dfFinal.head()

dfFinal.columns.values

# Read in the data & create matrices
y, X = dmatrices('hospital_expire_flag ~ Government + Medicare + Medicaid + Self_Pay + F + M + cleanage + timediff', dfFinal, return_type = 'dataframe')


# Sklearn output
model = LogisticRegression(fit_intercept = False, C = 1e9)
%time model.fit(X, y)
%prun model.fit(X, y)
%memit model.fit(X, y)
#%mprun model.fit(X,y)
mdl = model.fit(X, y)
model.coef_

#Statsmodels Logistic Regression Output
logit = sm.Logit(y, X)
%time sm.Logit(y, X)
%prun sm.Logit(y, X)
%memit sm.Logit(y, X)
result = logit.fit()
result.summary2()

#Write it to LaTeX
beginningtex = """\\documentclass{report}
\\usepackage{booktabs}
\\begin{document}"""
endtex = "\end{document}"

f = open('mylog1.tex', 'w')
f.write(beginningtex)
f.write(result.summary2().as_latex())
f.write(endtex)
f.close()

#StatsModels Documentation Referred to (see code references at end of document)

#Shape the data into numpy arrays for logistic regression
#Get rid of the labels, put them into arrays
y = y.values
X = X.values

#Numba doesn't work when the arrays are in (R, 1) format, so make Y a one dimensional array
Y = y.sum(axis=1)


@numba.jit(nopython=True, parallel=True)
def logistic_regression(Y, X, w, iterations, lr):
    for i in range(iterations):
        w -= (lr*(np.dot(((1.0 / (1.0 + np.exp(-Y * np.dot(X, w))) - 1.0) * Y), X)))/len(X)
    return w

#Adapted from official Numba documentation (see code references at end of doc)


 #Initialize w as a vector of X features
w = np.zeros(9)


#Run logistic regression
logistic_regression(Y, X, w, 1000, .001)


#Profile logistic regression
%time logistic_regression(Y, X, w, 1000, .001)
%prun logistic_regression(Y, X, w, 1000, .001)
%memit logistic_regression(Y, X, w, 1000, .001)



#Keras model (Neural Network Implementation)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Keras documentation referred to (look at end of doc)


import keras
from keras.regularizers import l1_l2
from keras.models import Sequential
from keras.layers import Dense, Activation


model = Sequential()
model.add(Dense(2,  # output dim is 2, one score per each class
                activation='softmax',
                kernel_regularizer=l1_l2(l1=0.0, l2=0.1),
                input_dim=(9)))
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
%time %prun %memit model.fit(X_train, y_train, epochs=100 , validation_data=(X_test, y_test))

#Keras documentation referred to (see end of document)