import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression as lr
import matplotlib.pyplot as plt 

data=pd.read_csv("linear.csv")

x=data["metrekare"]
y=data["fiyat"]
x=pd.DataFrame.as_matrix(x)
y=pd.DataFrame.as_matrix(y)
x = x.reshape(99,1)
y = y.reshape(99,1)
lineerregresyon = lr() 
lineerregresyon.fit(x,y) 
lineerregresyon.predict(x)

m = lineerregresyon.coef_
b= lineerregresyon.intercept_
a = np.arange(200)


plt.scatter(x,y) 
plt.scatter(a,m*a+b, c="red")
plt.show()


print('Eğim: ', lineerregresyon.coef_)
print("intercept:" ,lineerregresyon.intercept_)





