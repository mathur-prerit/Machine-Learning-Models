#Artifical Neural Network

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

#Applying Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

#Creating an ANN

#import keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

#INtitalising ANN
classifier=Sequential()

#Creating input layerand first hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim=11))

#Addina second hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))

#Adding output layer
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

#Compiling the ANN i.e. apllying gradient descent
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN in training set
classifier.fit(x_train,y_train,batch_size=10,nb_epoch=100)

#Predicting and evaluating the train model results

#Predcting testset results
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

#applying confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)