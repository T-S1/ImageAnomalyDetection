import glob             # フォルダ内ファイルのパスを取得するため
import cv2              # 画像読み込み画像処理ライブラリ
import numpy as np      # 数値計算ライブラリ
from src.visualize_data import show_image, show_images, show_results
# 自作のグラフ表示モジュール


def read_rgb_image(image_path):
    """画像をRGBの順で読み込む関数"""
    image = cv2.imread(image_path)                      # BGRの順でJPGファイルの読み込み
    im_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)     # RGBの順に変換
    return im_cvt.astype(np.float64)                    # 計算のため浮動小数にする


template_image = read_rgb_image("./data/forPatternMatching/template.jpg")
height = template_image.shape[0]    # 画像の高さ[ピクセル]
width = template_image.shape[1]     # 画像の幅[ピクセル]
show_image(template_image)          # テンプレート画像の表示

threshold = 15   # 異常検知の閾値

resized_pixels = 128

image_paths = glob.glob("./data/forPatternMatching/test/*.jpg")     # テスト用画像ファイルのパス取得
n_data = len(image_paths)                               # データ数
# images = np.zeros((n_data, height, width, 3))         # 画像を格納
# dif_images = np.zeros((n_data, height, width, 3))     # 差分画像を格納
images = np.zeros((n_data, resized_pixels, resized_pixels, 3))
dif_images = np.zeros((n_data, resized_pixels, resized_pixels, 3))
labels = np.zeros(n_data, np.int64) - 1     # 異常検知の結果を格納
mad_arr = np.zeros(n_data)                  # MADを格納

template_image = cv2.resize(template_image, (resized_pixels, resized_pixels))

for i in range(n_data):
    image = read_rgb_image(image_paths[i])
    image = cv2.resize(image, (resized_pixels, resized_pixels))
    dif_image = np.abs(image - template_image)  # 画像差分算出
    sad = np.sum(dif_image)                     # SADの算出
    mad = sad / (height * width)                # 画素数で割りMADを算出

    labels[i] = 0 if mad < threshold else 1     # 正常なら0, 異常なら1

    images[i, :, :, :] = image
    # dif_images[i, :, :, :] = dif_image
    # dif_images[i, :, :, :] = 255 - (dif_image * 255 / np.amax(dif_image))
    dif_images[i, :, :, :] = 255 - dif_image
    mad_arr[i] = mad

show_results(images, labels, nrows=2, ncols=3, values=mad_arr, value_name="MAD")    # 判定結果の表示
show_images(dif_images, nrows=2, ncols=3)                                           # 差分画像の表示
