
"""前準備"""
# import pdb
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.visualize_data import RT_Drawer, show_results

normal_paths = glob.glob("outsource/Hazelnut/test/good/*.jpg")
anomaly_paths = glob.glob("outsource/Hazelnut/test/crack/*.jpg")
image_paths = normal_paths + anomaly_paths
# n_data = len(image_paths)
image_paths = image_paths[::2]
n_data = 20

labels = np.zeros(n_data, dtype=np.int32)
labels[len(normal_paths):] = 1

h_resize = 64                                   # リサイズ後の高さ(ピクセル数)
image = cv2.imread(image_paths[0])               # データ読み込み
height = image.shape[0]                         # 元画像の高さ
width = image.shape[1]                          # 元画像の幅
w_resize = round(width * h_resize / height)     # リサイズ後の幅(ピクセル数)

def preprocess(image):
    im_proc = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        # RGBの順に変更
    im_proc = cv2.resize(im_proc, (w_resize, h_resize))     # 画素数の変更
    im_proc = im_proc / 255                                 # 0-1にスケーリング
    # im_proc = np.expand_dims(im_proc, 0)                    # モデルの入力形式に合わせる
    return im_proc

model = keras.models.load_model("deep_learning_model.h5")

drawer = RT_Drawer()
images = np.zeros((n_data, h_resize, w_resize, 3))
labels = np.zeros(n_data, dtype=np.int32) - 1

for i in range(n_data):
    image = cv2.imread(image_paths[i])  # リアルタイムに取得した画像を想定
    im_proc = preprocess(image)
    x = np.expand_dims(im_proc, axis=0)
    y = model.predict(x)
    pred_label = np.argmax(y, axis=1)[0]
    drawer.update(im_proc, pred_label)

    images[i, :, :, :] = im_proc
    labels[i] = pred_label

drawer.cleanup()
show_results(images, labels)
