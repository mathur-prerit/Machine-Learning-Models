#Smiple linear regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Salary_Data.csv')
#print(pd.DataFrame(dataset))
x=dataset.iloc[:,:1].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3, random_state=0)

#fitting simple linear regressor
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

#predicting regressed data
y_pred=regressor.predict(x_test)

#visualizing and plotting results
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary v/s Experience')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Salary v/s Experience(Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()