import pandas as pd
import numpy as np

dataset=pd.read_csv("global_co2.csv")


mean_value=dataset['Per Capita'].mean()
dataset['Per Capita']=dataset['Per Capita'].fillna(mean_value)

x=dataset.drop(["Per Capita"],axis="columns")
y=dataset["Per Capita"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

regressor.score(x_test,y_test)


print("CO2 in 2011=",regressor.predict([[2011,0.9,0,23,2,5,1]]))

print("CO2 in 2012=",regressor.predict([[2012,568,90,73,0,89,0]]))

print("CO2 in 2013=",regressor.predict([[2013,78,0,0.3,9,0,12]]))

print("The Accuracy of Linear Regression=",regressor.score(x_test,y_test))

y_pred=regressor.predict(x_test)