# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 21:53:52 2018

@author: DeLL
"""
#imprt library files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

#no need to apply feature scaling
#simple linear regression model for compamrig
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#simple polynomial model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

#making prediction of liner regression model
plt.scatter(X, y , color ='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('truth or bluff(linear regression model)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()

#prediction through polynomial regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color ='blue')
plt.title('truth or bluff(polynomial regression model)')
plt.xlabel('position level')
plt.ylabel('Salary')
plt.show()



#predicting a new result using linear model
lin_reg.predict(6.5)

#precting the new result using polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))

