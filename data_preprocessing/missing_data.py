import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Data.csv')
x=dataset.iloc[:, :-1].values
y=dataset.iloc[:,:3].values

#to view full array
#np.set_printoptions(threshold=np.nan)

#missing data handling
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0)
imputer=imputer.fit(x[:, 1:3])
x[:, 1:3]=imputer.transform(x[:, 1:3])
print(x[:, 1:3])