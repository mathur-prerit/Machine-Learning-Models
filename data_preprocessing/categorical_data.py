import numpy as np
import matplotlib.pyplot as mlt
import pandas as pd

dataset = pd.read_csv('Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# using labelEncoder to have categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_x = LabelEncoder()
x[:, 0] = label_x.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()

label_y = LabelEncoder()
y = label_y.fit_transform(y)
