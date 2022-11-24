# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.

2. Calculate the null values present in the dataset and apply label encoder
. .
3. Determine test and training data set and apply decison tree regression in dataset.
4. 
5. calculate Mean square error,data prediction and r2.
## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: 
RegisterNumber:  
*/
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Anusha R
RegisterNumber:  212221230006
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

## Output:
## dataset

![c1](https://user-images.githubusercontent.com/94222288/203821084-55bfc5d4-810d-43f8-93db-a6021f060edd.png)
## null values
![c2](https://user-images.githubusercontent.com/94222288/203821146-f749ff01-773a-4b7e-be2a-12222ec63eea.png)


![c3](https://user-images.githubusercontent.com/94222288/203821411-6f6d491d-08a1-49ef-91b7-f9a23d5aaf88.png)
## applying label encoder

![c4](https://user-images.githubusercontent.com/94222288/203821438-108c31c7-d468-45ed-af26-43282e4b6b88.png)




![c5](https://user-images.githubusercontent.com/94222288/203821457-a32f9019-8599-466e-8c15-c48da6b45e77.png)

## mean
![c6](https://user-images.githubusercontent.com/94222288/203821475-7a4450a4-443c-424b-b38a-fa0239b035fd.png)

## R2
![c7](https://user-images.githubusercontent.com/94222288/203821499-e4eb4295-0eda-497c-86a1-6605387dd9f0.png)

## Data prediction
![c8](https://user-images.githubusercontent.com/94222288/203821544-a1a99eee-cb50-4b89-b873-d910fa08af59.png)





## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
