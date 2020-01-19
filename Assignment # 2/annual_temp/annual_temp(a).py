import pandas as pd
import numpy as np 

dataset=pd.read_csv("annual_temp.csv")

dumies=pd.get_dummies(dataset.Source)

merge=pd.concat([dataset,dumies],axis="columns")

final_dataset=merge.drop(["Source"],axis="columns")

x=final_dataset.drop(["Mean"],axis="columns")
y=final_dataset.Mean

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

 
regressor=LinearRegression()
regressor.fit(x_train,y_train)

print("The Annual Temperature of GCAG in 2016 is =",regressor.predict([[2016,1,0]]))
print("The Annual Temperature of GISTEMP in 2016 is =",regressor.predict([[2016,0,1]]))
print("The Annual Temperature of GCAG in 2017 is =",regressor.predict([[2017,1,0]]))
print("The Annual Temperature of GISTEMP in 2017 is =",regressor.predict([[2017,0,1]]))

print("The Accuracy of Annual Temperatur of Two Companies using Regression=",regressor.score(x_test,y_test))

y_pred=regressor.predict(x_test)
