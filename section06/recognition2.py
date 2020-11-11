from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# 訓練データとテストデータに分割
digits = datasets.load_digits()
# 75％がテストデータ
x_train, x_test, t_train, t_test = train_test_split(digits.data, digits.target)

# サポートベクターマシーン
clf = svm.SVC()
# 訓練
clf.fit(x_train, t_train)

# テストデータで予測
#y_test = clf.predict(x_test)
# 正解率など
#print(metrics.classification_report(t_test, y_test))
# 行:正解、列:予測
#print(metrics.confusion_matrix(t_test, y_test))

# 予測結果と画像の対応
start = 110
end = start + 10
# チェック対象
images = digits.images[start:end]
# 正解値
correct = digits.target[start:end]
# 結果取得
result = clf.predict(digits.data[start:end])
for i in range(10):
    # 2行5列、i+1の位置
    plt.subplot(2, 5, i + 1)
    # 画像の表示
    plt.imshow(images[i], cmap="Greys")
    plt.axis("off")
    # タイトルとして認識した値を表示
    plt.title(str(result[i]) + ":" + str(correct[i]))
plt.show()
