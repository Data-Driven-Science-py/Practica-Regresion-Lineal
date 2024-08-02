import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df = pd.read_csv('score.csv')

df1 = df.loc[:, ['Hours', 'Scores']].groupby('Hours').mean()

model1 = LinearRegression()
x1 = df1.index.values.reshape(-1, 1)
y1 = df1['Scores'].values.reshape(-1, 1)
model1.fit(x1, y1)
range_ = np.arange(0, 11, 1).reshape(-1, 1)
fig, ax = plt.subplots()
ax.set(ylim = (0, 100), xlim = (0, 10))
ax.scatter(x1, y1)
ax.plot(range_, model1.predict(range_), color = 'red')

df2 = pd.read_csv('Salary Data.csv')

df3 = df2.loc[:, ['YearsExperience', 'Salary']].groupby('YearsExperience').mean()

model2 = LinearRegression()
x2 = df3.index.values.reshape(-1, 1)
y2 = df3['Salary'].values.reshape(-1, 1)
model2.fit(x2, y2)
range2_ = np.arange(0, 16, 1).reshape(-1, 1)
fig2, ax2 = plt.subplots()
ax2.set(ylim = (0, 180000), xlim = (0, 15))
ax2.scatter(x2, y2)
ax2.plot(range2_, model2.predict(range2_), color = 'red')
