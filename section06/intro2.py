from sklearn import datasets
from sklearn import svm

iris = datasets.load_iris()
#print(iris.target)  # 品種を表すラベル 0:Setosa、1:Versicolor、2:Versinica

# サポートベクターマシーン
clf = svm.SVC()
# 訓練
clf.fit(iris.data, iris.target)

# 品種の判定 (Sepal length, Sepal width, Petal length, Petal width)
#print(clf.predict([[5.1, 3.5, 1.4, 0.1], [6.5, 2.5, 4.4, 1.4], [5.9, 3.0, 6.2, 2.0]]))
# 判定
#print(clf.predict([[5.1, 3.5, 1.4, 0.1]]))

count = {}
for i in iris.data:
    result = clf.predict([i])
    count[result[0]] = count.get(result[0], 0) + 1

print(count)
