# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 07:58:25 2019

@author: Ramazan Güven
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

data=pd.read_csv("Iris.csv")

data.drop("Id",axis=1,inplace=True)
data.drop("Species",axis=1,inplace=True)



#%% For Dendogram


from scipy.cluster.hierarchy import linkage,dendrogram

merg=linkage(data,method="ward") #other methods  """weighted, centroid ve median """
dendrogram(merg,leaf_rotation=90)
plt.xlabel("data")
plt.ylabel("Öklit Uzaklığı")
plt.show()

#%%

from sklearn.cluster import AgglomerativeClustering

HC_cluster=AgglomerativeClustering(n_clusters=4,affinity="euclidean",linkage="ward")
clusters=HC_cluster.fit_predict(data)
data["labels"]=clusters


plt.scatter(data.SepalLengthCm[data.labels==0],data.SepalWidthCm[data.labels==0],color="red")
plt.scatter(data.SepalLengthCm[data.labels==1],data.SepalWidthCm[data.labels==1],color="green")
plt.scatter(data.SepalLengthCm[data.labels==2],data.SepalWidthCm[data.labels==2],color="black")
plt.scatter(data.SepalLengthCm[data.labels==3],data.SepalWidthCm[data.labels==3],color="yellow")
plt.scatter(data.SepalLengthCm[data.labels==4],data.SepalWidthCm[data.labels==4],color="blue")
#plt.scatter(data.SepalLengthCm[data.labels==1],data.SepalWidthCm[data.labels==1],data.PetalLengthCm[data.labels==1],data.PetalWidthCm[data.labels==1])
#plt.scatter(data.SepalLengthCm[data.labels==2],data.SepalWidthCm[data.labels==2],data.PetalLengthCm[data.labels==2],data.PetalWidthCm[data.labels==2])
#plt.scatter(data.SepalLengthCm[data.labels==3],data.SepalWidthCm[data.labels==3],data.PetalLengthCm[data.labels==3],data.PetalWidthCm[data.labels==3])
#plt.scatter(data.SepalLengthCm[data.labels==4],data.SepalWidthCm[data.labels==4],data.PetalLengthCm[data.labels==4],data.PetalWidthCm[data.labels==4])
plt.show()