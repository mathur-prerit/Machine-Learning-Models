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

y_pred2=lreg.predict(6.5)


#visualling linear regression result
plt.scatter(x,y,color='red')
plt.plot(x,lreg.predict(x),color='blue')
plt.title('Truth or bluff (linear regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt. show()