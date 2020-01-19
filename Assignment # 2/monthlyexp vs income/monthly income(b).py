import pandas as pd
import numpy as np

dataset=pd.read_csv("monthlyexp vs incom.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split


from sklearn.linear_model import LinearRegression


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)


x_train,x_test,y_train,y_test=train_test_split(X_poly,y,random_state=0,test_size=0.2)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_train, y_train)

print("The Accuracy of Ploynomial Regression=",lin_reg_2.score(x_test,y_test))

print("The Predited Value1 from Polynomial Regression=",lin_reg_2.predict(poly_reg.fit_transform([[25]])))


print("The Predited Value2 from Polynomial Regression=",lin_reg_2.predict(poly_reg.fit_transform([[23]])))