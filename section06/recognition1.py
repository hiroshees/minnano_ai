from sklearn import datasets
from sklearn import svm
import matplotlib.pyplot as plt

# 数字画像データの読み込み
digits = datasets.load_digits()

print("--- 画像データ ---")
print(digits.images[0])
print(digits.images.shape)
print("--- 1次元画像データ ---")
print(digits.data[0])
print(digits.data.shape)
print("--- ラベル ---")
print(digits.target)
print(digits.target.shape)

# 画像と正解値の表示
images = digits.images
labels = digits.target
for i in range(10):
    plt.subplot(2, 5, i + 1)  # 2行5列、i+1の位置
    plt.imshow(images[i], cmap="Greys")
    plt.axis("off")
    plt.title("Label: " +  str(labels[i]))
plt.show()
