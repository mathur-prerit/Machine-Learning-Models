#XGBoost

#Data preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

#distinguishing categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x1=LabelEncoder()
x[:,1]=labelencoder_x1.fit_transform(x[:,1])
labelencoder_x2=LabelEncoder()
x[:,2]=labelencoder_x2.fit_transform(x[:,2])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
x=x[:,1:]

#spliting dataset into train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Implementing Xgboost to training set
from xgboost import XGBClassifier
classifier=XGBClassifier()
classifier.fit(x_train,y_train)

#Predcting testset results
y_pred=classifier.predict(x_test)

#applying confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Applying K fold cross validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10)
accuracies.mean()
accuracies.std()

