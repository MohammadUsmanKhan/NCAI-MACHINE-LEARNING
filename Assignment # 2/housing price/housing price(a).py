import numpy as np
import pandas as pd

dataset=pd.read_csv("housing price.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
regressor.score(x_test,y_test)
y_pred=regressor.predict(x_test)
print("The Accuracuy of Linear Regression =",regressor.score(x_test,y_test))
print("Enter Housing ID=",regressor.predict([[1293]]))


"""
Improving the accuracy of model using Regularization
"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=10)),
    ('model', LinearRegression())
]

pipeline = Pipeline(steps)

pipeline.fit(x_train, y_train)

print('Training score: {}'.format(pipeline.score(x_train, y_train)))
print('Test score: {}'.format(pipeline.score(x_test, y_test)))