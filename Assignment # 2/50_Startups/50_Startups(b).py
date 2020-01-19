import pandas as pd
import numpy as np

dataset=pd.read_csv("50_Startups.csv")

dumies=pd.get_dummies(dataset.State)


mered=pd.concat([dataset,dumies],axis='columns')

final_dataset=mered.drop(["New York","State"],axis="columns")

x=final_dataset.drop(["Profit"],axis="columns")
y=final_dataset.Profit

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

from sklearn.tree import DecisionTreeRegressor

regressor=DecisionTreeRegressor(random_state=0)

regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

print("The Accuracy of Decision Tree =",regressor.score(x_test,y_test))
print("The Profit of California=",regressor.predict([[66051.5,182646,118148,0,1]]))
print("The Profit of Florida=",regressor.predict([[100672,91790.6,249745,1,0]]))
