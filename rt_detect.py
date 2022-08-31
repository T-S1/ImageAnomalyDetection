
"""前準備"""
# import pdb
import glob
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from src.visualize_data import show_results

normal_paths = glob.glob("outsource/Hazelnut/test/good/*.jpg")
anomaly_paths = glob.glob("outsource/Hazelnut/test/crack/*.jpg")
image_paths = normal_paths + anomaly_paths
n_data = len(image_paths)

labels = np.zeros(n_data, dtype=np.int32)
labels[len(normal_paths):] = 1

h_resize = 64                                   # リサイズ後の高さ(ピクセル数)
image = cv2.imread(image_paths[0])               # データ読み込み
height = image.shape[0]                         # 元画像の高さ
width = image.shape[1]                          # 元画像の幅
w_resize = round(width * h_resize / height)     # リサイズ後の幅(ピクセル数)

def preprocess(image):
    im_proc = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # RGBの順に変更
    im_proc = cv2.resize(im_proc, (w_resize, h_resize))   # 画素数の変更
    return im_proc / 255                                  # 0-1にスケーリング

model = keras.models.load_model("deep_learning_model.h5")

for i in range(n_data):
    image = cv2.imread(image_paths[i])  # リアルタイムに取得した画像を想定
