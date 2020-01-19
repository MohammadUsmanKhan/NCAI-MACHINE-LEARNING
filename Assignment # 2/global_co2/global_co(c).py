import pandas as pd
import numpy as np

dataset=pd.read_csv("global_co2.csv")


mean_value=dataset['Per Capita'].mean()
dataset['Per Capita']=dataset['Per Capita'].fillna(mean_value)

x=dataset.drop(["Per Capita"],axis="columns")
y=dataset["Per Capita"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(X_poly,y,random_state=0,test_size=0.3)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_train, y_train)

lin_reg_2.score(x_test,y_test)
print("CO2 in 2011=",lin_reg_2.predict(poly_reg.fit_transform([[2011,0.9,0,23,2,5,1]])))

print("CO2 in 2012=",lin_reg_2.predict(poly_reg.fit_transform([[2012,568,90,73,0,89,0]])))

print("CO2 in 2013=",lin_reg_2.predict(poly_reg.fit_transform([[2013,78,0,0.3,9,0,12]])))

print("The accuracy of Polynomial Regression=",lin_reg_2.score(x_test,y_test))
y_pred=lin_reg_2.predict(x_test)