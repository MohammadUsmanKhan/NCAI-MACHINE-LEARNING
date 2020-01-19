import pandas as pd
import numpy as np

dataset=pd.read_csv("monthlyexp vs incom.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split


from sklearn.tree import DecisionTreeRegressor


x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

lin_reg_2 = DecisionTreeRegressor()
lin_reg_2.fit(x_train, y_train)
print("The Accuracy of Decision Tree=",lin_reg_2.score(x_test,y_test))

y_pred=lin_reg_2.predict(x_test)


print("The predicted Value of Decision Tree",lin_reg_2.predict([[9]]))
