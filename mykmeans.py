import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt

class Kmeans:
    def __init__(self, k):
        self.k = k

    def calc_distance(self, x1, x2):
        diff = x1 - x2
        distances = np.sqrt(np.square(diff).sum(axis=1))
        return distances

    def fit(self, x):
        self.x = x
        m, n = self.x.shape
        # 随机选定k个数据作为初始质心，不重复选取
        self.original_ = np.random.choice(m, self.k, replace=False)
        # 默认类别是从0到k-1
        self.original_center = x[self.original_]
        while True:
            # 初始化一个字典，以类别作为key，赋值一个空数组
            dict_y = {}
            for j in range(self.k):
                dict_y[j] = np.empty((0, n))
            for i in range(m):
                distances = self.calc_distance(x[i], self.original_center)
                # 把第i个数据分配到距离最近的质心，存放在字典中
                label = np.argsort(distances)[0]
                dict_y[label] = np.r_[dict_y[label], x[i].reshape(1, -1)]
            centers = np.empty((0, n))
            # 对每个类别的样本重新求质心
            for i in range(self.k):
                center = np.mean(dict_y[i], axis=0).reshape(
                    1, -1)  # reshape转化成1行
                centers = np.r_[centers, center]
            # 与上一次迭代的质心比较，如果没有发生变化，则停止迭代（也可考虑收敛时停止）
            result = np.all(centers == self.original_center)
            if result == True:
                return centers # 返回质心,用于绘制聚合的簇心点
            else:
                # 继续更新质心
                self.original_center = centers

    def predict(self, x):
        y_preds = []
        m, n = x.shape
        for i in range(m):
            distances = self.calc_distance(x[i], self.original_center)
            y_pred = np.argsort(distances)[0]
            y_preds.append(y_pred)
        return y_preds


model = Kmeans(k=2)
df1 = pd.read_csv('train-normalized.csv', usecols=[0, 1, 2, 3])
xy1 = df1.iloc[:, [1, 2]]
xy1 = xy1.values
centers = model.fit(xy1)

# print( model.cluster_centers_, '\n')
# print( model.inertia_, '\n')

label = df1['is_anomaly']
df1 = df1.values

df2 = pd.read_csv('test-normalized.csv', usecols=[0, 1, 2, 3])
xy2 = df2.iloc[:, [1, 2]]
xy2 = xy2.values
predict = model.predict(xy2)

label = df2['is_anomaly']
df2 = df2.values

plt.figure(figsize=(21, 21))
plt.subplot(1, 2, 1)
plt.scatter(xy2[:, 0], xy2[:, 1], c=predict)
plt.scatter(centers[:, 0], centers[:, 1], c=['r','b'])
plt.xlabel('CPC')
plt.ylabel('CPM')
plt.title('The predictions of the test set based on KMeans')

plt.subplot(1, 2, 2)
plt.scatter(xy2[:, 0], xy2[:, 1], c=label)
plt.scatter(centers[:, 0], centers[:, 1], c=['r','b'])
plt.xlabel('CPC')
plt.ylabel('CPM')
plt.title('The real labels of the test set')


plt.show()

 
