"""前準備"""
# import pdb
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from src.visualize_data import show_images, show_results

SEED = 22
tf.random.set_seed(SEED)

# train_normal_paths = glob.glob("outsource/Hazelnut/train/good/*.jpg")     # フォルダ内のパス取得
# train_anomaly_paths = glob.glob("outsource/Hazelnut/train/crack/*.jpg")
# test_normal_paths = glob.glob("outsource/Hazelnut/test/good/*.jpg")
# test_anomaly_paths = glob.glob("outsource/Hazelnut/test/crack/*.jpg")
# train_paths = train_normal_paths + train_anomaly_paths
# test_paths = test_normal_paths + test_anomaly_paths

normal_paths = glob.glob("outsource/Hazelnut/train/good/*.jpg")
normal_paths.extend(glob.glob("outsource/Hazelnut/test/good/*.jpg"))
anomaly_paths = glob.glob("outsource/Hazelnut/train/crack/*.jpg")
anomaly_paths.extend(glob.glob("outsource/Hazelnut/test/crack/*.jpg"))
image_paths = normal_paths + anomaly_paths
n_images = len(image_paths)

labels = np.zeros(n_images, dtype=np.int64)
labels[len(normal_paths):] = 1

# n_train = len(train_paths)
# n_test = len(test_paths)
# train_labels = np.zeros(n_train, dtype=np.int64)
# test_labels = np.zeros(n_test, dtype=np.int64)
# train_labels[len(train_normal_paths):] = 1
# test_labels[len(test_normal_paths):] = 1

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

images = np.zeros((n_images, h_resize, w_resize, 3), dtype=np.float32)
for i in range(n_images):
    im = cv2.imread(image_paths[i])                     # BGRの順で格納
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)            # RGBの順に変換
    im_resize = cv2.resize(im, (w_resize, h_resize))    # 画素数の変換
    images[i, :, :, :] = im_resize / 255                # 0-1にスケーリング
show_images(images[::4])


# train_ims = np.zeros((n_train, h_resize, w_resize, 3))     # 異常検知のための参照画像
# for i in range(n_train):
#     im = cv2.imread(train_paths[i])         # BGRの順で格納
#     train_ims[i, :, :, :] = preprocess(im)  # 処理後データの格納

# test_ims = np.zeros((n_test, h_resize, w_resize, 3))     # 異常検知のための参照画像
# for i in range(n_test):
#     im = cv2.imread(test_paths[i])         # BGRの順で格納
#     test_ims[i, :, :, :] = preprocess(im)  # 処理後データの格納

# show_images(train_ims[::4])   # 前処理後の画像を確認

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

x_train, x_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=SEED, stratify=labels
)

history = model.fit(
    x_train, y_train, batch_size=1, epochs=10, validation_data=(x_test, y_test)
)

test_scores = model.evaluate(x_test, y_test)

preds = model.predict(x_test, y_test)

show_results

# """異常検知のための閾値決定"""
# sims = np.zeros(n_data)     # 正常データ間の距離を格納する配列
# for i in range(n_data):
#     sim_max = -1
#     for j in range(n_data):
#         if i != j:
#             # cos類似度
#             sim = np.sum(ims_ref[i] * ims_ref[j])
#             sim /= np.linalg.norm(ims_ref[i])
#             sim /= np.linalg.norm(ims_ref[j])
#             sim_max = max(sim_max, sim)     # 全データに対する最大類似度を算出
#     sims[i] = sim_max

# idx_ref = int(n_data * 0.2)         # 異常の過検出をどの程度許容するか
# th_sim = np.sort(sims)[idx_ref]     # 閾値
# print(f"閾値: {th_sim}")

# """パターンマッチングによる異常検知"""
# test_paths = glob.glob("outsource/Hazelnut/test/good/*.jpg")
# test_paths.extend(glob.glob("outsource/Hazelnut/test/crack/*.jpg"))
# n_test = len(test_paths)

# ims = np.zeros((n_test, h_resize, w_resize, 3))
# results = []
# sims = np.zeros(n_test)
# for i in range(n_test):
#     im = cv2.imread(test_paths[i])
#     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#     im_res = cv2.resize(im, (w_resize, h_resize))
#     sim_max = -1
#     for im_ref in ims_ref:
#         # cos類似度
#         sim = np.sum(im_res * im_ref)
#         sim /= np.linalg.norm(im_res)
#         sim /= np.linalg.norm(im_ref)
#         sim_max = max(sim_max, sim)

#     sims[i] = sim_max
#     ims[i, :, :, :] = im_res

#     if sim_max > th_sim:
#         results.append("normal")
#     else:
#         results.append("abnormal")

# show_similarities(sims, th_sim)
# show_results(ims[::2], results[::2])
