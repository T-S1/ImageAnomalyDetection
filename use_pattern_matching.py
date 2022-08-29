"""前準備"""
# import pdb
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_paths = glob.glob("outsource/Hazelnut/train/good/*.jpg")
n_data = len(image_paths)

short_side_size = 32
im = cv2.imread(image_paths[0])
height = im.shape[0]
width = im.shape[1]
if height < width:
    h_resize = short_side_size
    w_resize = round(width * short_side_size / height)
else:
    h_resize = round(height * short_side_size / width)
    w_resize = short_side_size

"""正常データの読み込み&前処理"""
ims_ref = np.zeros((n_data, h_resize, w_resize, 3))
for i in range(len(image_paths)):
    im = cv2.imread(image_paths[i])     # BGR
    im_res = cv2.resize(im, (w_resize, h_resize))
    ims_ref[i, :, :, :] = im_res

    if i == 0:
        im_rgb = cv2.cvtColor(im_res, cv2.COLOR_BGR2RGB)
        plt.axis("off")
        plt.imshow(im_rgb)
        plt.show()

sims = np.zeros(n_data)
for i in range(n_data):
    sim_max = -1
    for j in range(n_data):
        if i != j:
            # cos類似度
            sim = np.sum(ims_ref[i] * ims_ref[j])
            sim /= np.linalg.norm(ims_ref[i])
            sim /= np.linalg.norm(ims_ref[j])
            sim_max = max(sim_max, sim)
    sims[i] = sim_max

plt.plot(sims, "o")
plt.show()

idx_ref = int(n_data * 0.2)     # 異常の過検出をどの程度許容するか
th_sim = np.sort(sims)[idx_ref]
print(f"閾値: {th_sim}")

"""パターンマッチングによる異常検知"""
test_paths = glob.glob("outsource/Hazelnut/test/good/*.jpg")
test_paths.extend(glob.glob("outsource/Hazelnut/test/crack/*.jpg"))
# th_sim = 0.96
n_test = len(test_paths)

for im_path in test_paths:
    im = cv2.imread(im_path)
    im_res = cv2.resize(im, (w_resize, h_resize))
    sim_max = -1
    for im_ref in ims_ref:
        # cos類似度
        sim = np.sum(im_res * im_ref)
        sim /= np.linalg.norm(im_res)
        sim /= np.linalg.norm(im_ref)
        sim_max = max(sim_max, sim)

    if sim_max > th_sim:
        result = "good"
    else:
        result = "bad "

    print(f"{im_path}: {result}, {sim_max}")
