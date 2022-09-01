import glob
import cv2
import numpy as np
from tensorflow import keras

from src.visualize_data import RT_Drawer, show_results

image_paths = glob.glob("./data/forRealtimeDetection/*.JPG")
n_data = len(image_paths)

h_resize = 64                                   # リサイズ後の高さ(ピクセル数)
image = cv2.imread(image_paths[0])              # データ読み込み
height = image.shape[0]                         # 元画像の高さ
width = image.shape[1]                          # 元画像の幅
w_resize = round(width * h_resize / height)     # リサイズ後の幅(ピクセル数)

model = keras.models.load_model("model.h5")

drawer = RT_Drawer()
images = np.zeros((n_data, h_resize, w_resize, 3))
labels = np.zeros(n_data, dtype=np.int32) - 1

for i in range(n_data):
    image = cv2.imread(image_paths[i])      # リアルタイムに取得した画像を想定
    im_proc = cv2.resize(image, (w_resize, h_resize))     # 画素数の変更
    im_proc = cv2.cvtColor(im_proc, cv2.COLOR_BGR2RGB)        # RGBの順に変更
    im_proc = im_proc / 255                                 # 正規化
    x = np.expand_dims(im_proc, axis=0)     # モデルの入力形式に合わせる
    y = model.predict(x)                    # 予測
    pred_label = np.argmax(y, axis=1)[0]    # one-hot表現から直す
    drawer.update(im_proc, pred_label)      # リアルタイム描画に反映

    images[i, :, :, :] = im_proc
    labels[i] = pred_label

drawer.cleanup()    # 終了処理
show_results(images, labels)    # 全体の予測結果の確認
