#HEiracrchical CLustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('Mall_Customers.csv')
x=dataset.iloc[:,[3,4]].values

#Using dendrogram to find optimal number of cluster
import scipy.cluster.hierarchy as sch
dendrogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()

#Fitting Heirarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(x)

#visualising the clusters
plt.scatter(x[y_hc==0,0],x[y_hc==0,1],s=100,color='red',label='Cluster 1')
plt.scatter(x[y_hc==1,0],x[y_hc==1,1],s=100,color='blue',label='CLuster 2')
plt.scatter(x[y_hc==2,0],x[y_hc==2,1],s=100,color='green',label='CLuster 3')
plt.scatter(x[y_hc==3,0],x[y_hc==3,1],s=100,color='magenta',label='CLuster 4')
plt.scatter(x[y_hc==4,0],x[y_hc==4,1],s=100,color='cyan',label='CLuster 5')
plt.xlabel('Annual Score')
plt.ylabel('Spending Score')
plt.title('CLusters of customers')
plt.legend()
plt.show()