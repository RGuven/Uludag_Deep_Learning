from sklearn.linear_model import LinearRegression as lr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data=np.loadtxt("ex1data1.txt",delimiter=",")
print(data[1:3])

x=data[:,0]*10 #metrekare
y=data[:,1]*10 #fiyat

assert x.shape==y.shape

#plt.scatter(x,y)
#plt.show()

m,b=np.polyfit(x,y,1) #1.dereceden polinom ile bul
y=np.arange(97)
plt.scatter(x,y)
plt.plot(m*y+b) # x i yukaıda kullandığım için y dedim
plt.show()

z=int(input("Kaç metrekare ev Tahmin etmek istersin: "))
tahmin=m*z+b
print(tahmin)
plt.scatter(z,tahmin)
plt.show()

###################################################################
#SCKİT LEARN İLE YAZMA 
x=x.reshape(-1,1)
y=y.reshape(-1,1)
lineerreg=lr()
lineerreg.fit(x,y)
lineerreg.predict(x)

print("Eğim: \n",lineerreg.coef_)
print("Y eksenini kestiği yer\n",lineerreg.intercept_)

tahmin=lineerreg.coef_*x+lineerreg.intercept_
a=np.arange(97)
plt.plot(a,tahmin)
plt.show()










