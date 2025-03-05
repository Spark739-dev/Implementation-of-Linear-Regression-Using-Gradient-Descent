# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the necessary python library to implement and to use. 
2. use the user defined function and give the input for X1 and y and theta values.
3. use for loop to iterate the inside ststement within the range.
4. Use pd.read_csv command to read the csv file and X1 and y input value from the dataset using iloc.
5. Use the standard scaler for X1 and y to normalize the range.
6. give 'new_data' as array input.
7. Use the 'pre' variable to store the inverse scaler transformed value.
8. print the variable 'pre' to get the predicted value.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: VESHWANTH.
RegisterNumber: 212224230300

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _  in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions  - y).reshape(-1,1)
        theta-=learning_rate * (1/len(X1)) * X.T.dot(errors)
    return theta
data=pd.read_csv("C:\\Users\\admin\\Desktop\\DS LAB FILES\\machine learning\\DATASET-20250226\\50_Startups.csv")

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1=scaler.fit_transform(X1)
Y1=scaler.fit_transform(y)

theta=linear_regression(X1,Y1)

new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print("Predicted value:",pre) 
*/
```

## Output:
![Screenshot 2025-03-05 092700](https://github.com/user-attachments/assets/f7a2db07-bee6-4a65-a332-f58e83cebeb8)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
