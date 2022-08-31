"""前準備"""
# import pdb
import glob
import cv2
import numpy as np

from src.visualize_data import show_images, show_similarities, show_results

image_paths = glob.glob("outsource/Hazelnut/train/good/*.jpg")  # フォルダ内のパス取得
n_data = len(image_paths)   # データ数

# 画像リサイズの画素数設定
h_resize = 64                                   # リサイズ後の高さ(ピクセル数)
im = cv2.imread(image_paths[0])                 # データ読み込み
height = im.shape[0]                            # 元画像の高さ
width = im.shape[1]                             # 元画像の幅
w_resize = round(width * h_resize / height)     # リサイズ後の幅(ピクセル数)

"""正常データの読み込み&前処理"""
ims_ref = np.zeros((n_data, h_resize, w_resize, 3))     # 異常検知のための参照画像
for i in range(len(image_paths)):
    im = cv2.imread(image_paths[i])                 # BGRの順で格納
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)        # RGBの順に変更
    im_res = cv2.resize(im, (w_resize, h_resize))   # 画素数の変更
    ims_ref[i, :, :, :] = im_res                    # 処理後データの格納

show_images(ims_ref[::4])   # 前処理後の画像を確認

"""異常検知のための閾値決定"""
sims = np.zeros(n_data)     # 正常データ間の距離を格納する配列
for i in range(n_data):
    sim_max = -1
    for j in range(n_data):
        if i != j:
            # cos類似度
            sim = np.sum(ims_ref[i] * ims_ref[j])
            sim /= np.linalg.norm(ims_ref[i])
            sim /= np.linalg.norm(ims_ref[j])
            sim_max = max(sim_max, sim)     # 全データに対する最大類似度を算出
    sims[i] = sim_max

idx_ref = int(n_data * 0.2)         # 異常の過検出をどの程度許容するか
th_sim = np.sort(sims)[idx_ref]     # 閾値
print(f"閾値: {th_sim}")

"""パターンマッチングによる異常検知"""
test_paths = glob.glob("outsource/Hazelnut/test/good/*.jpg")
test_paths.extend(glob.glob("outsource/Hazelnut/test/crack/*.jpg"))
n_test = len(test_paths)

ims = np.zeros((n_test, h_resize, w_resize, 3))
preds = np.zeros(n_test, dtype=np.int64) - 1
sims = np.zeros(n_test)
for i in range(n_test):
    im = cv2.imread(test_paths[i])
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_res = cv2.resize(im, (w_resize, h_resize))
    sim_max = -1
    for im_ref in ims_ref:
        # cos類似度
        sim = np.sum(im_res * im_ref)
        sim /= np.linalg.norm(im_res)
        sim /= np.linalg.norm(im_ref)
        sim_max = max(sim_max, sim)

    sims[i] = sim_max
    ims[i, :, :, :] = im_res

    if sim_max > th_sim:
        preds[i] = 0
    else:
        preds[i] = 1

    # print(f"{im_path}: {result}, {sim_max}")

show_similarities(sims, th_sim)
show_results(ims[::2], preds[::2])
