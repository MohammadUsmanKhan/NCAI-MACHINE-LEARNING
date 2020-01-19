import pandas as pd
import numpy as np

dataset=pd.read_csv("global_co2.csv")


mean_value=dataset['Per Capita'].mean()
dataset['Per Capita']=dataset['Per Capita'].fillna(mean_value)

x=dataset.drop(["Per Capita"],axis="columns")
y=dataset["Per Capita"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(x_train,y_train)

regressor.score(x_test,y_test)

print("CO2 in 2011=",regressor.predict([[2011,5280,808,2199,2094,128,51]]))

print("CO2 in 2012=",regressor.predict([[2012,5965,937,2412,241,152,50]]))

print("CO2 in 2013=",regressor.predict([[2013,78,0,0.3,9,0,12]]))

print("The Accuracy of Decision Tree=",regressor.score(x_test,y_test))


y_pred=regressor.predict(x_test)