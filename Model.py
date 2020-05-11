# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:36:58 2020

@author: Babu Patil
"""
#Importing the Libraries
import pandas as pd
import numpy as np
import pickle

#Reading the source file
data = pd.read_csv('hiring.csv')
data

#Deviding the data into dependant and indipendent variables
X = data.iloc[:,:-1]
y = data.iloc[:,-1]

#As we have only smaller dataset no need to splitt the data into train and test
# We are training model on the complete dataset

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y)

#Saving the model as a Pickel file, 'Serialising the model'
pickle.dump(reg,open('model.pkl','wb'))

#Checking our model with sample data
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2,9,6]]))