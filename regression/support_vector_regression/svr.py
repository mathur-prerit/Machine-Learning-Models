import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x=x.reshape(-1,1)
x=sc_x.fit_transform(x)
y=y.reshape(-1,1)
sc_y=StandardScaler()
y=sc_y.fit_transform(y)

#Fitting dataset with SVR
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(x,y)

#predciting regression model
y_pred=sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

#visualising our result
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel=('Salary')
plt.show()