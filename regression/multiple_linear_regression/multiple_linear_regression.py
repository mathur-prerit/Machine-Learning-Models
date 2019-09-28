import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lencoder_x=LabelEncoder()
x[:,3]=lencoder_x.fit_transform(x[:,3])
oht=OneHotEncoder(categorical_features = [3])
x= oht.fit_transform(x).toarray()

#avoiding dummy vairable trap
x=x[:,1:]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#fitting training set for multiple linear regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

#predicting test result
y_pred=reg.predict(x_test)

#building optimal model using backward elimination
import statsmodels as sm
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)

x_opt=x[:,[0,1,2,3,4,5]]
reg_OLS=sm.regression.linear_model.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()

#removing the higest x values which is more than SL after reviewing summary
x_opt=x[:,[0,1,3,4,5]]
reg_OLS=sm.regression.linear_model.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()

x_opt=x[:,[0,3,4,5]]
reg_OLS=sm.regression.linear_model.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()

x_opt=x[:,[0,3,5]]
reg_OLS=sm.regression.linear_model.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()

x_opt=x[:,[0,3]]
reg_OLS=sm.regression.linear_model.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()
