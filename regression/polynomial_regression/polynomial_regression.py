import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2]
y=dataset.iloc[:,2]

print(pd.DataFrame(x))
print(pd.DataFrame(y))

#fitting it in linear regression
from sklearn.linear_model import LinearRegression
lreg=LinearRegression()
y_pred=lreg.fit(x,y)

#fitting dataset in polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)

#fitting the polynomail regression dataset with linear regression model
lreg2=LinearRegression()
y_pred2=lreg2.fit(x_poly,y)  

#visualling linear regression result
plt.scatter(x,y,color='red')
plt.plot(x,lreg.predict(x),color='blue')
plt.title('Truth or bluff (linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt. show()

#visualling polynomial regression model
plt.scatter(x,y,color='red')
plt.plot(x,lreg2.predict(poly_reg.fit_transform(x)),color='blue')
plt.title('Truth or bluff (polynomial regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt. show()

#predicting new result with linear regression
lreg.predict(6.5)

#predicting new result with polynomial regression
lreg2.predict(poly_reg.fit_transform(6.5))