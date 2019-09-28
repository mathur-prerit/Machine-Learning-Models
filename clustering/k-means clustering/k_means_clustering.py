#K means clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values

#Using elbow method to fin optimal number of clusters
from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Applying kmeans to our dataset
kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_kmeans=kmeans.fit_predict(x)

#visualising the cluster
plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,color='red',label='Cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,color='blue',label='CLuster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,color='green',label='CLuster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,color='magenta',label='CLuster 4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,color='cyan',label='CLuster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,label='centroids',color='yellow')
plt.xlabel('Annual Score')
plt.ylabel('Spending Score')
plt.title('CLusters of customers')
plt.legend()
plt.show()