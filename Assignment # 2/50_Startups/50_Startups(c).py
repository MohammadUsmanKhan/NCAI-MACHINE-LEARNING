import pandas as pd
import numpy as np

dataset=pd.read_csv("50_Startups.csv")

dumies=pd.get_dummies(dataset.State)


mered=pd.concat([dataset,dumies],axis='columns')

final_dataset=mered.drop(["New York","State"],axis="columns")

x=final_dataset.drop(["Profit"],axis="columns")
y=final_dataset.Profit

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(X_poly,y,random_state=0,test_size=0.2)
poly_reg.fit(x_train, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_train, y_train)

y_pred=lin_reg_2.predict(x_test)

print("The Accuracy of Polynomial Regression=",lin_reg_2.score(x_test,y_test))
print("The Profit of California=",lin_reg_2.predict(poly_reg.fit_transform([[66051.5,182646,118148,0,1]])))
print("The Profit of Florida=",lin_reg_2.predict(poly_reg.fit_transform([[100672,91790.6,249745,1,0]])))


