
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from datetime import datetime, timedelta


### LOADING DATA ###
data = pd.read_csv(r"C:\Users\ashut\PycharmProjects\NumPython\DataScience\Dataset\Covid\OurWorldInData\total_cases.csv")
# print(data.tail())

df1 = data[['date', 'World']]
df1["id"] = df1.index + 1
# print(df1.head())

df2 = df1[['id', 'World']]
# print(df2.head())


### PREPARING DATA ###
x = np.array(df2['id']).reshape(-1, 1)
y = np.array(df2['World']).reshape(-1, 1)
plt.plot(y, '-m')
# plt.show()

polyFeat = PolynomialFeatures(degree=3)
x = polyFeat.fit_transform(x)
# print(x)
# print(df2.shape)

### TRAINING DATA ###
model = linear_model.LinearRegression()
model.fit(x, y)
accuracy = model.score(x, y)
print(f'Accuracy:{round(accuracy*100, 3)} %')

y0 = model.predict(x)
plt.plot(y0, '--b')
# plt.show()


### PREDICTION ###
print('\nDataset is upto 25th Oct 2020')
days = 4

print(f'Cases after {days} days will be: ', end='')
print(round(int(model.predict(polyFeat.fit_transform([[300+days]]))) / 1000000, 2), 'Million')

