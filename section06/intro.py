import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Irisデータの読み込み
iris = datasets.load_iris()

# 各花のサイズ
iris_data = iris.data

# 散布図で表示
st_data = iris_data[:50]  # Setosa
vc_data = iris_data[50:100]  # Versicolor
vn_data = iris_data[100:150]  # Versinica

plt.scatter(st_data[:, 0], st_data[:, 1], label="Setosa")  # Sepal lengthとSepal width
plt.scatter(vc_data[:, 0], vc_data[:, 1], label="Versicolor")  # Sepal lengthとSepal width
plt.scatter(vn_data[:, 0], vn_data[:, 1], label="Versinica")  # Sepal lengthとSepal width
plt.legend()
plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.show()

plt.scatter(st_data[:, 2], st_data[:, 3], label="Setosa")  # Petal lengthとPetal width
plt.scatter(vc_data[:, 2], vc_data[:, 3], label="Versicolor")  # Petal lengthとPetal width
plt.scatter(vn_data[:, 2], vn_data[:, 3], label="Versinica")  # Petal lengthとPetal width
plt.legend()
plt.xlabel("Petal length (cm)")
plt.ylabel("Petal width (cm)")
plt.show()
