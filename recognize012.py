# 0,1,2の画像を学習し、数字認識を行う

# TensorFlowとtf.kerasのインポート
import tensorflow as tf
from tensorflow import keras

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt
import cv2
from random import randint

# 訓練用データとテスト用データの画像配列を初期化
train_images = np.zeros((80,28,28), np.uint8)
test_images = np.zeros((40,28,28), np.uint8)
# 訓練用データとテスト用データのラベル配列を初期化
train_labels = np.zeros(80, np.uint8)
test_labels = np.zeros(40, np.uint8)

# 訓練用データの読み込み
for i in range(80):
    fname = 'train/train'+str(i)+'.png'
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    train_images[i] = img
    if i <= 26:
        train_labels[i] = 0
    elif 27 <= i <= 53:
        train_labels[i] = 1
    elif 54 <= i:
        train_labels[i] = 2

# テスト用データの読み込み
for i in range(40):
    fname = 'test/test'+str(i)+'.png'
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    test_images[i] = img
    if i <= 12:
        test_labels[i] = 0
    elif 13 <= i <= 25:
        test_labels[i] = 1
    elif 26 <= i:
        test_labels[i] = 2

# クラス名
class_names = ['0', '1', '2']

# 正規化
train_images = train_images / 255.0
test_images = test_images / 255.0

# ニュートラルネットワークの層の設定
model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(128, activation='relu'),
                          keras.layers.Dense(3, activation='softmax')])

# モデルのコンパイル
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# モデルの訓練
result = model.fit(train_images, train_labels, epochs=30)

# 正答率の評価
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)
predictions = model.predict(test_images)
# print(np.argmax(predictions[0]))
# print(test_labels[0])

# 予測値の表示
# print('------------------------------------------------')
# print('predictions : ')
# for i in range(40):
#     print('{} : {}' .format(class_names[test_labels[i]], predictions[i]))

# 評価指標
evaluation = np.zeros((3,3), np.uint8) # 混同行列
for i in range(40):
    predicted_label = np.argmax(predictions[i])
    if class_names[test_labels[i]] == '0':
        if class_names[predicted_label] == '0':
            evaluation[0,0] += 1
        elif class_names[predicted_label] == '1':
            evaluation[0,1] += 1
        elif class_names[predicted_label] == '2':
            evaluation[0,2] += 1
    elif class_names[test_labels[i]] == '1':
        if class_names[predicted_label] == '0':
            evaluation[1,0] += 1
        elif class_names[predicted_label] == '1':
            evaluation[1,1] += 1
        elif class_names[predicted_label] == '2':
            evaluation[1,2] += 1
    elif class_names[test_labels[i]] == '2':
        if class_names[predicted_label] == '0':
            evaluation[2,0] += 1
        elif class_names[predicted_label] == '1':
            evaluation[2,1] += 1
        elif class_names[predicted_label] == '2':
            evaluation[2,2] += 1
print('------------------------------------------------')
print('混同行列 : ')
print(evaluation)

print('------------------------------------------------')
accuracy = evaluation[0,0] + evaluation[1,1] + evaluation[2,2] # 正答率
accuracy = accuracy/np.sum(evaluation) * 100
print('正答率 : {} %' .format(accuracy))

print('------------------------------------------------')
row_sum = np.sum(evaluation, axis=0) # 列ごとの和
precision0 = evaluation[0,0] / row_sum[0] * 100 # 0の適合率
print('0の適合率 : {} %' .format(round(precision0, 2)))
precision1 = evaluation[1,1] / row_sum[1] * 100 # 1の適合率
print('1の適合率 : {} %' .format(round(precision1, 2)))
precision2 = evaluation[2,2] / row_sum[2] * 100 # 2の適合率
print('2の適合率 : {} %' .format(round(precision2, 2)))
precision = (precision0 + precision1 + precision2)/3 # 平均適合率
print('平均適合率 : {} %' .format(round(precision, 2)))

print('------------------------------------------------')
col_sum = np.sum(evaluation, axis=1) # 行ごとの和
recall0 = evaluation[0,0] / col_sum[0] * 100 # 0の再現率
print('0の再現率 : {} %' .format(round(recall0, 2)))
recall1 = evaluation[1,1] / col_sum[1] * 100 # 1の再現率
print('1の再現率 : {} %' .format(round(recall1, 2)))
recall2 = evaluation[2,2] / col_sum[2] * 100 # 2の再現率
print('2の再現率 : {} %' .format(round(recall2, 2)))
recall = (recall0 + recall1 + recall2)/3 # 平均再現率
print('平均再現率 : {} %' .format(round(recall, 2)))

# テスト用データの画像10枚のうち9枚の比較
# plt.figure(figsize=(7,7))
# for i in range(40):
#     plt.subplot(5,8,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(test_images[i], cmap=plt.cm.binary)
#     predicted_label = np.argmax(predictions[i])
#     plt.xlabel("{} ({})" .format(class_names[test_labels[i]], class_names[predicted_label]))
# plt.show()

# 損失率と正答率を図示
plt.plot(range(1,31), result.history['loss'], label="training")
plt.plot(range(1,31), result.history['accuracy'], label="training")
plt.show()
# plt.plot(range(1,21), result.history['accuracy'], label="training", color='orange')
# plt.show()
# plt.plot(range(1,21), result.history['loss'], label="training")
# plt.show()