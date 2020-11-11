import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import random

iris = datasets.load_iris()
iris_data = iris.data
# 花びらの長さ
sepal_length = iris_data[:100, 0]
# 花びらの幅
sepal_width = iris_data[:100, 1]

# 平均値を0に
sepal_length_average = np.average(sepal_length)
sepal_length -= sepal_length_average
sepal_width_average = np.average(sepal_width)
sepal_width -= sepal_width_average

# 入力をリストに格納
train_data = []
for i in range(100):  # iには0から99までが入る
    correct = iris.target[i]
    train_data.append([sepal_length[i], sepal_width[i], correct])

# シグモイド関数
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ニューロン
class Neuron:
    def __init__(self):
        self.total = 0.0
        self.output = 0.0

    def add(self, input):
        self.total += input

    def get_output(self):
        self.output = sigmoid(self.total)
        return self.output

    def reset(self):
        self.total = 0.0
        self.output = 0.0


# ニューラルネットワーク
class NeuralNetwork:
    def __init__(self):  # 初期設定
        # 重み
        self.middle_layer_weigh = [
            [4.0, 4.0],
            [4.0, 4.0],
            [4.0, 4.0],
        ]
        self.output_layer_weigh = [
            [1.0, -1.0, 1.0]
        ]

        # バイアス
        self.middle_layer_bias = [3.0, 0.0, -3.0]
        self.output_layer_bias = [-0.5]

        # 各層の宣言
        self.input_layers = [0.0, 0.0]
        self.middle_layers = [Neuron(), Neuron(), Neuron()]
        self.output_layers = [Neuron()]

    def commit(self, input_data):  # 実行
        # 入力層の値代入
        self.input_layers[0] = input_data[0]
        self.input_layers[1] = input_data[1]
        # 中間層のリセット
        for middle_layer in self.middle_layers:
            middle_layer.reset()
        # 出力層のリセット
        for output_layer in self.output_layers:
            output_layer.reset()

        # 入力層→中間層
        for i, middle_layer in enumerate(self.middle_layers):
            for j, input_layer in enumerate(self.input_layers):
                middle_layer.add(input_layer * self.middle_layer_weigh[i][j])
            middle_layer.add(self.middle_layer_bias[i])

        # 中間層→出力層
        for index, middle_layer in enumerate(self.middle_layers):
            self.output_layers[0].add(middle_layer.get_output() * self.output_layer_weigh[0][index])
        self.output_layers[0].add(self.output_layer_bias[0])

        return self.output_layers[0].get_output()

    def train(self, correct):
        # 学習係数
        k = 0.3

        #  出力
        output_o = self.output_layers[0].output
        output_m0 = self.middle_layers[0].output
        output_m1 = self.middle_layers[1].output
        output_m2 = self.middle_layers[2].output

        # δ
        delta_o = (output_o - correct) * output_o * (1.0 - output_o)
        delta_m0 = delta_o * self.middle_layer_weigh[0][0] * output_m0 * (1.0 - output_m0)
        delta_m1 = delta_o * self.middle_layer_weigh[0][1] * output_m1 * (1.0 - output_m1)

        # パラメータの更新
        self.output_layer_weigh[0][0] -= k * delta_o * output_m0
        self.output_layer_weigh[0][1] -= k * delta_o * output_m1
        self.output_layer_weigh[0][2] -= k * delta_o * output_m2
        self.output_layer_bias[0] -= k * delta_o

        self.middle_layer_weigh[0][0] -= k * delta_m0 * self.input_layers[0]
        self.middle_layer_weigh[0][1] -= k * delta_m0 * self.input_layers[1]
        self.middle_layer_weigh[1][0] -= k * delta_m1 * self.input_layers[0]
        self.middle_layer_weigh[1][1] -= k * delta_m1 * self.input_layers[1]
        self.middle_layer_weigh[2][0] -= k * delta_m1 * self.input_layers[0]
        self.middle_layer_weigh[2][1] -= k * delta_m1 * self.input_layers[1]
        self.middle_layer_bias[0] -= k * delta_m0
        self.middle_layer_bias[1] -= k * delta_m1
        self.middle_layer_bias[2] -= k * delta_m1


# ニューラルネットワークのインスタンス
neural_network = NeuralNetwork()


# グラフ表示用の関数
def show_graph(epoch):
    print("Epoch:", epoch)
    # 実行
    st_predicted = [[], []]  # Setosa
    vc_predicted = [[], []]  # Versicolor
    for data in train_data:
        if neural_network.commit(data) < 0.5:
            st_predicted[0].append(data[0]+sepal_length_average)
            st_predicted[1].append(data[1]+sepal_width_average)
        else:
            vc_predicted[0].append(data[0]+sepal_length_average)
            vc_predicted[1].append(data[1]+sepal_width_average)

    # 分類結果をグラフ表示
    plt.scatter(st_predicted[0], st_predicted[1], label="Setosa")
    plt.scatter(vc_predicted[0], vc_predicted[1], label="Versicolor")
    plt.legend()

    plt.title("Epoch:" + str(epoch))

    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Sepal width (cm)")
    plt.show()


show_graph(0)


# 学習と結果の表示
for t in range(0, 64):
    random.shuffle(train_data)
    for data in train_data:
        neural_network.commit(data[:2])  # 順伝播
        neural_network.train(data[2])  # 逆伝播
    if t+1 in [1, 2, 4, 8, 16, 32, 64]:
        show_graph(t+1)

# 比較用に元の分類を散布図で表示
st_data = iris_data[:50]  # Setosa
vc_data = iris_data[50:100]  # Versicolor
plt.scatter(st_data[:, 0], st_data[:, 1], label="Setosa")
plt.scatter(vc_data[:, 0], vc_data[:, 1], label="Versicolor")
plt.legend()

plt.xlabel("Sepal length (cm)")
plt.ylabel("Sepal width (cm)")
plt.title("Original")
plt.show()
