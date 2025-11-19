import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv('AmesHousing.csv')

print(df.head())

df.info()
df.describe()
df.isnull().sum()
df.columns()

print(df.shape)
print(df.dtypes)
print(df.describe)
print(df.info)
print(df.isnull().sum())
print(df.tail())
print(df.columns)

X = df['Gr Liv Area']
y = df['SalePrice']

X = X.values.reshape(-1, 1)
y = y.values.reshape(-1, 1)

print(X.shape)
print(y.shape)

plt.scatter(X, y)
plt.xlabel('Gr Liv Area')
plt.ylabel('SalePrice')
plt.show()

regr = linear_model.LinearRegression()
regr.fit(X, y)

y_pred = regr.predict(X)

print('Coefficients: \n', regr.coef_)
print("Mean squared error: %.2f" % mean_squared_error(y, y_pred))
print('Variance score: %.2f' % r2_score(y, y_pred))
plt.scatter(X, y, color='black')
plt.plot(X, y_pred, color='blue', linewidth=3)  
plt.show()