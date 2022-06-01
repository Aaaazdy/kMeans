from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import operator
from functools import reduce
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from torch import is_anomaly_enabled


model = KMeans(n_clusters=2)
df1 = pd.read_csv('train-normalized.csv', usecols=[0, 1, 2, 3])
xy1 = df1.iloc[:, [1, 2]]
model.fit(xy1)

# print( model.cluster_centers_, '\n')
print( model.labels_, '\n' )
# print( model.inertia_, '\n')
# print( df['is_anomaly'] )
label = df1['is_anomaly']
df1 = df1.values
xy1 = xy1.values

df2 = pd.read_csv('test-normalized.csv', usecols=[0, 1, 2, 3])
xy2 = df2.iloc[:, [1, 2]]
predict = model.predict(xy2)

label = df2['is_anomaly']
df2 = df2.values
xy2 = xy2.values

plt.figure(figsize=(21, 21))
plt.subplot(1, 2, 1)
plt.scatter(xy2[:, 0], xy2[:, 1],c=predict)
plt.xlabel('CPC')
plt.ylabel('CPM')
plt.title('The predictions of the test set based on KMeans')

plt.subplot(1, 2, 2)
plt.scatter(xy2[:, 0], xy2[:, 1], c=label)
plt.xlabel('CPC')
plt.ylabel('CPM')
plt.title('The real labels of the test set')


plt.show()

print(model.score(xy2))
