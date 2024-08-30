# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.import needed libriaries like numpy , pandas and StandardScaler from sklern preprocessing and deifne linear regression like X1,y learning_rate=0.01 and num_iters=1000
2.Apply the linear regression function to the standardlized features X1_scaled and Y1_scaled to obtain parameters.
3.prepare the new data and make predication using the trained models.
4.print the predited value found out from the regression analysis.

## Program:
```
 /*
Program to implement the linear regression using gradient descent.
Developed by: MOHAMED SULTHAN A
RegisterNumber:  212223230125
*/


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate = 0.1, num_iters = 1000):
    X = np.c_[np.ones(len(X1)),X1]
    
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        
        #calculate predictions
        predictions = (X).dot(theta).reshape(-1,1)
        
        #calculate errors
        errors=(predictions - y ).reshape(-1,1)
        
        #update theta using gradiant descent
        theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
                                        
data=pd.read_csv("C:/classes/ML/50_Startups.csv")
data.head()

#assuming the lost column is your target variable 'y' 

X = (data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#learn modwl paramerers

theta=linear_regression(X1_Scaled,Y1_Scaled)

#predict target value for a new data
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## Output:
 ## DATA.HEAD():
 ![image](https://github.com/user-attachments/assets/07f2b57f-9603-4495-9e86-a662b1b95cdf)
##  X VALUE:
 ![image](https://github.com/user-attachments/assets/72ca6355-0aba-43c1-a034-ce00899c1d42)
##  X1_SCALED VALUE:
 ![image](https://github.com/user-attachments/assets/c9ac09d9-c4bb-443d-8668-4823d85ac10f)
##  PREDICTED VALUES:
 ![image](https://github.com/user-attachments/assets/d4a5a3df-ee88-47d0-bf46-dd698550903a)






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
