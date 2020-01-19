import pandas as pd
import numpy as np

dataset=pd.read_csv("monthlyexp vs incom.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

print("The Accuracy of Linear Regression=",regressor.score(x_test,y_test))


y_pred=regressor.predict(x_test)

print("The Predicted Value of Linear Regression=",regressor.predict([[50]]))