import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

data=pd.read_csv("linear.csv")

x=data["metrekare"]
y=data["fiyat"]
x=np.array(x)
y=np.array(y)
#plt.scatter(x,y)
#plt.show()

m,b=np.polyfit(x,y,1)
print("en uygun eğim" ,m)
print("en uygın b değeri",b)

uzunluk=np.arange(200)

plt.scatter(x,y)
plt.plot(m*uzunluk+b)
plt.show()

z=int(input("Kaç metrekare Tahmin etmek istersiniz?"))
print("Tahmininiz:{}".format(z))
tahmin=m*z+b
plt.scatter(x,y)
plt.plot(m*uzunluk+b)
plt.scatter(z,tahmin,c="red",marker="v")
plt.show()
