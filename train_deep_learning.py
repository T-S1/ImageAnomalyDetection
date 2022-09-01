"""------------------------------使用するライブラリ-----------------------------"""
import os                   # ファイル存在の有無の確認のため
import glob                 # フォルダ内ファイルのパスを取得するため
import tqdm                 # for文の進捗状況表示のため
import cv2                  # 画像読み込み画像処理ライブラリ
import numpy as np          # 数値計算ライブラリ
import tensorflow as tf     # 機械学習用ライブラリ
from tensorflow import keras                            # モデル構築に使用
from tensorflow.keras import layers                     # モデル構築に使用
from tensorflow.keras.utils import plot_model           # モデル構造の可視化に使用
from sklearn.model_selection import train_test_split    # 訓練・テストデータの分割に使用

from src.visualize_data import show_images, show_history, show_results  # 自作のグラフ表示モジュール

SEED = 22
tf.random.set_seed(SEED)    # 乱数のシード値を固定

normal_paths = glob.glob("data/normal/*.jpg")       # 正常データのパス
anomaly_paths = glob.glob("data/anomaly/*.jpg")     # 異常データのパス
image_paths = normal_paths + anomaly_paths          # 全データのパス
n_data = len(image_paths)                           # データ数

labels = np.zeros(n_data, dtype=np.int64)   
labels[len(normal_paths):] = 1

# 画像リサイズの画素数設定
h_resize = 64                                   # リサイズ後の高さ(ピクセル数)
image = cv2.imread(image_paths[0])              # データ読み込み
height = image.shape[0]                         # 元画像の高さ
width = image.shape[1]                          # 元画像の幅
w_resize = round(width * h_resize / height)     # リサイズ後の幅(ピクセル数)

"""正常データの読み込み&前処理"""
if not os.path.isfile("dataset.npy"):   # 処理後データが存在しない場合
    images = np.zeros((n_data, h_resize, w_resize, 3), dtype=np.float32)    # 画像を格納
    for i in tqdm.tqdm(range(n_data)):                      # tqdmで進捗状況確認
        image = cv2.imread(image_paths[i])                  # BGRの順で読み込み
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # RGBの順に変更
        image = cv2.resize(image, (w_resize, h_resize))     # 画素数の変更
        images[i, :, :, :] = image / 255                    # 0-1にスケーリング
    np.save("dataset.npy", images)      # 読み込み時間短縮のため，処理後データを保存
else:
    images = np.load("dataset.npy")     # 処理後データの読み込み

# show_images(train_ims[::4])
show_images(images[::4])

y = tf.one_hot(labels, 2).numpy()
x_train, x_val, y_train, y_val = train_test_split(
    images, y, test_size=0.2, random_state=SEED, stratify=labels
)

"""深層学習モデルの構築"""
inputs = keras.Input(shape=(h_resize, w_resize, 3))
x = layers.Conv2D(32, 3)(inputs)
x = layers.Activation("relu")(x)
x = layers.Conv2D(32, 3)(x)
x = layers.Activation("relu")(x)
x = layers.Flatten()(x)
x = layers.Dense(32)(x)
x = layers.Activation("relu")(x)
x = layers.Dense(2)(x)
outputs = layers.Activation("softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
# print(model.summary())
plot_model(model, show_shapes=True)
model.compile(
    loss=tf.keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.SGD(learning_rate=1e-3),
    metrics=["accuracy"],
)
history = model.fit(
    x_train, y_train, batch_size=64, epochs=100, validation_data=(x_val, y_val)
)

show_history(history)

_, val_acc = model.evaluate(x_val, y_val)
print(f"バリデーション時の正解数: {round(len(x_val) * val_acc)} / {len(x_val)}")

preds = model.predict(x_val)
pred_labels = np.argmax(preds, axis=1).astype(np.int64)

show_results(x_val, pred_labels)

model.save("model.h5")
