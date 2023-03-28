#IMPORTS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#Загружаем данные
df = pd.read_csv('ETHRUB_27_03.csv')

#Извлечение цены ETH
X = df.index.values.reshape(-1, 1)
y = df['price'].values.reshape(-1, 1)
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
eth_03_plt = plt.plot(y_pred)
#print(y_pred)
#print(X)
#print(y)