"""前準備"""
# import pdb
import time
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from src.visualize_data import show_images, show_history

SEED = 22
tf.random.set_seed(SEED)

# train_normal_paths = glob.glob("outsource/Hazelnut/train/good/*.jpg")     # フォルダ内のパス取得
# train_anomaly_paths = glob.glob("outsource/Hazelnut/train/crack/*.jpg")
# test_normal_paths = glob.glob("outsource/Hazelnut/test/good/*.jpg")
# test_anomaly_paths = glob.glob("outsource/Hazelnut/test/crack/*.jpg")
# train_paths = train_normal_paths + train_anomaly_paths
# test_paths = test_normal_paths + test_anomaly_paths
# n_train = len(train_paths)
# n_test = len(test_paths)
# print(f"訓練用データ数: {n_train}")
# print(f"テスト用データ数: {n_test}")

normal_paths = glob.glob("outsource/Hazelnut/train/good/*.jpg")
anomaly_paths = glob.glob("outsource/Hazelnut/train/crack/*.jpg")
image_paths = normal_paths + anomaly_paths
n_data = len(image_paths)

# train_labels = np.zeros(n_train, dtype=np.int32)
# test_labels = np.zeros(n_test, dtype=np.int32)
# train_labels[len(train_normal_paths):] = 1
# test_labels[len(test_normal_paths):] = 1

labels = np.zeros(n_data, dtype=np.int32)
labels[len(normal_paths):] = 1

# 画像リサイズの画素数設定
h_resize = 64                                   # リサイズ後の高さ(ピクセル数)
im = cv2.imread(image_paths[0])                 # データ読み込み
height = im.shape[0]                            # 元画像の高さ
width = im.shape[1]                             # 元画像の幅
w_resize = round(width * h_resize / height)     # リサイズ後の幅(ピクセル数)


"""正常データの読み込み&前処理"""
# def preprocess(im):
#     im_cvt = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)        # RGBの順に変更
#     im_res = cv2.resize(im_cvt, (w_resize, h_resize))   # 画素数の変更
#     return im_res / 255                                 # 正規化

# train_ims = np.zeros((n_train, h_resize, w_resize, 3), dtype=np.float32)
# for i in range(n_train):
#     im = cv2.imread(train_paths[i])                     # BGRの順で格納
#     train_ims[i, :, :, :] = preprocess(im)

# test_ims = np.zeros((n_test, h_resize, w_resize, 3), dtype=np.float32)
# for i in range(n_test):
#     im = cv2.imread(test_paths[i])
#     test_ims[i, :, :, :] = preprocess(im)

images = np.zeros((n_data, h_resize, w_resize, 3), dtype=np.float32)
for i in range(n_data):
    image = cv2.imread(image_paths[i])
    image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)         # RGBの順に変更
    image = cv2.resize(image, (w_resize, h_resize))     # 画素数の変更
    images[i, :, :, :] = image / 255                    # 0-1にスケーリング

# show_images(train_ims[::4])
show_images(images[::4])

"""深層学習モデルの構築"""
inputs = keras.Input(shape=(h_resize, w_resize, 3))
x = layers.Conv2D(32, 3)(inputs)
x = layers.Activation("relu")(x)
x = layers.Conv2D(32, 3)(x)
x = layers.Activation("relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(32)(x)
x = layers.Activation("relu")(x)
x = layers.Dense(1)(x)
outputs = layers.Activation("sigmoid")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
print(model.summary())

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=["accuracy"],
)

x_train, x_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=SEED, stratify=labels
)

# history = model.fit(
#     train_ims, train_labels, batch_size=1, epochs=10, validation_data=(test_ims, test_labels)
# )

history = model.fit(
    x_train, y_train, batch_size=1, epochs=2, validation_data=(x_val, y_val)
)

_, val_acc = model.evaluate(x_val, y_val)
print(f"バリデーション時の正解数: {round(len(x_val) * val_acc)} / {len(x_val)}")

show_history(history)

model.save("deep_learning_model.h5")
