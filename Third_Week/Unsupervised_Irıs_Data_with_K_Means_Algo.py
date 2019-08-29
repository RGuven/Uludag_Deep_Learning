# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 07:04:48 2019

@author: Ramazan Güven
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

data=pd.read_csv("Iris.csv")

data.drop("Id",axis=1,inplace=True)
data.drop("Species",axis=1,inplace=True)
data=np.array(data)
#%%
"""FOR ELBOW """

inertia_list = np.empty(8)
for i in range(1,8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertia_list[i] = kmeans.inertia_
plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.show()
#%%

K_Means=KMeans(n_clusters=3,random_state=42)
K_Means.fit(data)
predict=K_Means.predict(data)

print(K_Means.cluster_centers_)

#%%
import matplotlib.pyplot as plt
plt.scatter(data[predict==0,0],data[predict==0,1],s=50,color='red')
plt.scatter(data[predict==1,0],data[predict==1,1],s=50,color='blue')
plt.scatter(data[predict==2,0],data[predict==2,1],s=50,color='green')
plt.title('UNSUPERVİSED_IRIS_DATA_WİTH_K_MEANS_ALGORİTHM')
plt.show()
#%% For Centroit
import matplotlib.pyplot as plt
plt.scatter(data[predict==0,0],data[predict==0,1],color='red')
plt.scatter(data[predict==1,0],data[predict==1,1],color='blue')
plt.scatter(data[predict==2,0],data[predict==2,1],color='green')
plt.scatter(K_Means.cluster_centers_[:,0],K_Means.cluster_centers_[:,1],color='yellow')
plt.title('UNSUPERVİSED_IRIS_DATA_WİTH_K_MEANS_ALGORİTHM')
plt.show()
