# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 06:22:20 2019

@author: Ramazan Güven
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#np.random.seed(0)
data = pd.read_csv("data.csv",header=0)


data.drop("id",axis=1,inplace=True)
data.drop("Unnamed: 32",axis=1,inplace=True)

data.diagnosis.unique()

data["diagnosis"]=data["diagnosis"].map({"M":1,"B":0})

print(data.iloc[1:5,1:5].describe())

plt.hist(data["diagnosis"])
plt.title("Diagnosis Malignant=kötü huylu (1) Belign=İyi huylu (0)")

x=np.array(data.drop(["diagnosis"],axis=1))
y=np.array(data.diagnosis)

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

for i in range(10):
    i=i+1
    KNN=KNeighborsClassifier(n_neighbors=i)
    KNN.fit(X_train,y_train)
    
    accuracy=KNN.score(X_test,y_test)
    print("episode {} için accuracy {} 'dir".format(i,accuracy))
    
    print("\n\n\nAccuracymiz : ",accuracy*100)
    
    tahmin=np.array([17,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]).reshape(1,-1)
    print(KNN.predict(tahmin))
    print("0 ise iyi huylu 1 ise Kötü huylu")
    
    
    
    