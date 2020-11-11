from sklearn import svm
from sklearn.model_selection import train_test_split

# ファイルの読み込み
with open("stock_price.txt", "r") as f:
    stock_file_data = f.read()

# 改行で分割しリストに格納
stock_file_data = stock_file_data.split()
stock_data = []
for stock_string in stock_file_data:
    # 小数に変換した上でリストに格納
    stock_data.append(float(stock_string))

# データの確認
#print("株価", stock_data)
n_price = len(stock_data)
#print("株価データの数", n_price)

# 株価の変化率
ratio_data = []
for i in range(1, n_price):
    # 翌日 - 当日 / 当日
    ratio_data.append(float(stock_data[i] - stock_data[i-1]) / float(stock_data[i-1]))

#print("株価の変化率", ratio_data)
n_ratio = len(ratio_data)
#print("株価の変化率データの数", n_ratio)

# 前日までの4連続の変化率のデータ
successive_data = []
# 正解値 価格上昇: 1 価格低下: 0
answers = []
for i in range(4, n_ratio):
    successive_data.append([ratio_data[i-4], ratio_data[i-3], ratio_data[i-2], ratio_data[i-1]])
    if ratio_data[i] > 0:
        answers.append(1)
    else:
        answers.append(0)
#print("4日連続の変化率", successive_data)
#print("正解", answers)

# データを訓練用の70%とテスト用の25%に分割する。シャッフルはしない
x_train, x_test, t_train, t_test = train_test_split(successive_data, answers, shuffle=False)

# サポートベクターマシーン
clf = svm.SVC()
# 訓練
clf.fit(x_train, t_train)
# テスト用データで予測
y_test = clf.predict(x_test)

# 末尾の10個を比較
#print ("正解:", t_test[-10:])
#print ("予測:", list(y_test[-10:]))

# 正解率の計算
correct = 0.0
wrong = 0.0
for i in range(len(t_test)):
    if y_test[i] == t_test[i]:
        correct += 1.0
    else:
        wrong += 1.0
print ("正解率:", str(correct / (correct+wrong) * 100), "%")

