"""--------------------------------使用するライブラリ-------------------------------"""
import glob                 # フォルダ内ファイルのパスを取得するため
import tqdm                 # for文の進捗状況表示のため
import cv2                  # 画像読み込み画像処理ライブラリ
import numpy as np          # 数値計算ライブラリ
import tensorflow as tf     # 機械学習用ライブラリ
from tensorflow import keras                            # モデル構築に使用
from tensorflow.keras import layers                     # モデル構築に使用
from tensorflow.keras.utils import plot_model           # モデル構造の可視化に使用
from sklearn.model_selection import train_test_split    # 訓練・テストデータの分割に使用

from src.visualize_data import show_images, show_history, show_results   # 自作のグラフ表示モジュール

"""------------------------------データの読み込み&前処理-----------------------------"""
normal_paths = glob.glob("./data/forDeepLearning/normal/*.jpg")     # 正常データのパス
anomaly_paths = glob.glob("./data/forDeepLearning/anomaly/*.jpg")   # 異常データのパス
image_paths = normal_paths + anomaly_paths          # 全データのパス
n_data = len(image_paths)                           # データ数

h_resize = 128                                  # リサイズ後の高さ[ピクセル]
image = cv2.imread(image_paths[0])              # データ読み込み
height = image.shape[0]                         # 元画像の高さ
width = image.shape[1]                          # 元画像の幅
w_resize = round(width * h_resize / height)     # リサイズ後の幅[ピクセル]

images = np.zeros(
    (n_data, h_resize, w_resize, 3), dtype=np.float32
)   # 画像を格納
print("画像読み込み中")
for i in tqdm.tqdm(range(n_data)):                      # tqdmで進捗状況確認
    image = cv2.imread(image_paths[i])                  # BGRの順で読み込み
    im_proc = cv2.resize(image, (w_resize, h_resize))   # 画素数の変更
    im_proc = cv2.cvtColor(im_proc, cv2.COLOR_BGR2RGB)  # RGBの順に変更
    images[i, :, :, :] = im_proc / 255                  # 正規化

show_images(images[::100])    # 処理後データの確認

"""-----------------------------出力（教師）データの生成----------------------------"""
labels = np.zeros(n_data, dtype=np.int64)   # 教師ラベルを格納
labels[len(normal_paths):] = 1              # 正常なら0,異常なら1
y = tf.one_hot(labels, 2).numpy()           # one-hot表現に変換

"""-------------------モデルの学習と性能評価のためのデータセット分割------------------"""
SEED = 22   # 乱数のシード値

x_trainval, x_test, y_trainval, y_test = train_test_split(
    images, y, test_size=200, random_state=SEED, stratify=y
)   # データセットを訓練・検証用とテスト用に分割
x_train, x_val, y_train, y_val = train_test_split(
    x_trainval, y_trainval, test_size=200, random_state=SEED, stratify=y_trainval
)   # データセットを訓練用と検証用に分割

print("訓練用データ数:", len(x_train))
print("検証用データ数:", len(x_val))
print("テスト用データ数:", len(x_test))

tf.random.set_seed(SEED)    # tensorflowに係るシード値固定

"""----------↓モデルの定義(例)↓ 演習時はコメントアウト,範囲選択してCtrl + / (スラッシュ)---------"""
inputs = keras.Input(shape=(h_resize, w_resize, 3))                                 # 入力層
x = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same")(inputs)      # 畳み込み層
x = layers.Activation("relu")(x)                                                    # 活性化関数(ReLU)
x = layers.MaxPool2D(pool_size=2, strides=2, padding="valid")(x)                    # マックスプーリング層
x = layers.Flatten()(x)                                                             # 平坦化層
x = layers.Dense(units=16)(x)                                                        # 全結合層
x = layers.Activation("sigmoid")(x)                                                 # 活性化関数(シグモイド)
x = layers.Dense(units=2)(x)                                                        # 全結合層(出力)
outputs = layers.Activation("softmax")(x)                                           # 活性化関数(ソフトマックス)
model = keras.Model(inputs=inputs, outputs=outputs)     # モデル入出力の定義

plot_model(model, to_file="model.png", show_shapes=True)    # モデル構造の図を保存

model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.SGD(learning_rate=1e-3),
    metrics=["Accuracy"],
)   # 最適化に関する設定

history = model.fit(
    x_train, y_train, batch_size=64, epochs=50,
    validation_data=(x_val, y_val)
)   # 学習

show_history(history, save_path="history.png")                   # 学習中の正解率と損失の表示
"""----------↑モデルの定義(例)↑ 演習時はコメントアウト,範囲選択してCtrl + / (スラッシュ)---------"""

"""-----------------↓演習用↓ 使う時は範囲選択してCtrl + / (スラッシュ)--------------------------"""
# inputs = keras.Input(shape=(h_resize, w_resize, 3))   # 入力層

# # 中間層を記述

# x = layers.Dense(units=2)(x)                          # 全結合層（出力）
# outputs = layers.Activation("softmax")(x)             # 活性化関数(ソフトマックス)
# model = keras.Model(inputs=inputs, outputs=outputs)   # モデル入出力の定義

# plot_model(model, to_file="model.png", show_shapes=True)    # モデル構造の図を保存

# model.compile(
#     loss=keras.losses.CategoricalCrossentropy(),
#     optimizer=keras.optimizers.SGD(learning_rate=1e-3),
#     metrics=["Accuracy"],
# )   # 最適化に関する設定

# history = model.fit(
#     x_train, y_train, batch_size=64, epochs=50,
#     validation_data=(x_val, y_val)
# )   # 学習

# show_history(history, save_path="history.png")                   # 学習中の正解率と損失の表示
"""-----------------↑演習用↑ 使う時は範囲選択してCtrl + / (スラッシュ)--------------------------"""

"""---------------------------------モデル性能のテスト-----------------------------------------"""
_, accuracy = model.evaluate(x_test, y_test)
print("テスト用データに対する正解率：", accuracy)

"""-----------------------------------モデルの保存と読み込み----------------------------------"""
model.save("model.h5")  # モデルの保存

loaded_model = keras.models.load_model("model.h5")  # モデルの読み込み

preds = loaded_model.predict(x_test)                        # 予測
pred_labels = np.argmax(preds, axis=1).astype(np.int64)     # one-hot表現を直す
confidences = [preds[i, idx] for i, idx in enumerate(pred_labels)]  # モデルの出力値

show_results(x_test, pred_labels, values=confidences, value_name="出力値")        # 予測結果の出力
