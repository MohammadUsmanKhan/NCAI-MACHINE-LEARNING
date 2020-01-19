import pandas as pd 
import numpy as np

dataset=pd.read_csv("housing price.csv")

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 3)

from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=15)
regressor.fit(x_train,y_train)
print("The Accuracuy of Decision Tree =",regressor.score(x_test,y_test))
print("Enter Housing ID=",regressor.predict([[1293]]))
y_pred=regressor.predict(x_test)

