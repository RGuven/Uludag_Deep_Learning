# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 07:24:48 2019

@author: Ramazan Güven 
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


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

NaiveBayes = GaussianNB().fit(X_train, y_train)
predicted = NaiveBayes.predict(X_test)
print('cancer dataset')
print('Accuracy of GaussianNB classifier on training set: {:.2f}'.format(NaiveBayes.score(X_train, y_train)))
print('Accuracy of GaussianNB classifier on test set: {:.2f}'.format(NaiveBayes.score(X_test, y_test)))

#%%
from sklearn import metrics

print("Classification report for classifier %s:\n%s\n"
      % (NaiveBayes, metrics.classification_report(y_test, predicted)))
#########CONFUSİON MATRİX############
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))