import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')
x=(dataset.iloc[:, :-1].values)
y=(dataset.iloc[:, 3].values)

#removing mssign data
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)
imputer=imputer.fit(x[:, 1:3])
x[:, 1:3]=imputer.transform(x[:, 1:3])

#handling the encoding for text
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encode_y=LabelEncoder()
y=encode_y.fit_transform(y)

ex=LabelEncoder()
x[:,0]=ex.fit_transform(x[:,0])
encode_x=OneHotEncoder(categorical_features=[0])
x=encode_x.fit_transform(x).toarray()

#spliting data in train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=0 )
a=pd.DataFrame(x)
b=pd.DataFrame(y)

#feature scaling
from sklearn.preprocessing import StandardScaler
scale_x=StandardScaler
x_train=scale_x.fit_transform(x_train)
x_test=scale_x.transform(x_test)
