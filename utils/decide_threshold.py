"""前準備"""
# import pdb
import glob
import cv2
import numpy as np

from src.visualize_data import show_image, show_results


def read_rgb_image(image_path):
    image = cv2.imread(image_path)
    im_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return im_cvt


# 画像リサイズの画素数設定
h_resize = 64                                   # リサイズ後の高さ(ピクセル数)
im = cv2.imread(image_paths[0])                 # データ読み込み
height = im.shape[0]                            # 元画像の高さ
width = im.shape[1]                             # 元画像の幅
w_resize = round(width * h_resize / height)     # リサイズ後の幅(ピクセル数)

"""正常データの読み込み&前処理"""
if not os.path.isfile("dataset.npy"):
    images = np.zeros((n_data, h_resize, w_resize, 3), dtype=np.float32)
    for i in tqdm.tqdm(range(n_data)):
        image = cv2.imread(image_paths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)         # RGBの順に変更
        image = cv2.resize(image, (w_resize, h_resize))     # 画素数の変更
        images[i, :, :, :] = image / 255                    # 0-1にスケーリング
    np.save("dataset.npy", images)

def main():
    template_image = read_rgb_image("./data/template.JPG")
    show_image(template_image)

    normal_paths = glob.glob("data/normal/*.jpg")
    anomaly_paths = glob.glob("data/anomaly/*.jpg")
    image_paths = normal_paths + anomaly_paths
    n_data = len(image_paths)

    labels = np.zeros(n_data, dtype=np.int64)
    labels[len(normal_paths):] = 1

    height = template_image.shape[0]
    width = template_image.shape[1]
    threshold = 1

    image_paths = glob.glob("data/for_pattern_matching/*.jpg")
    image_paths.extend(glob.glob("data/for_pattern_matching/*.jpg"))
    n_data = len(image_paths)

    images = np.zeros((n_data, height, width, 3))
    labels = np.zeros(n_data, np.int32) - 1
    for i in range(n_data):
        image = read_rgb_image(image_paths[i])
        sad = np.sum(np.abs(image - template_image))

        if sad < threshold:
            labels[i] = 0
        else:
            labels[i] = 1

        images[i, :, :, :] = image

    print(images[0])
    show_results(images, labels)

if __name__ == "__main__":
    main()