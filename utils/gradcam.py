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

test_normal_paths = glob.glob("outsource/Hazelnut/test/good/*.jpg")
test_anomaly_paths = glob.glob("outsource/Hazelnut/test/crack/*.jpg")
test_paths = test_normal_paths + test_anomaly_paths
n_test = len(test_paths)
print(f"テスト用データ数: {n_test}")

test_labels = np.zeros(n_test, dtype=np.int32)
test_labels[len(test_normal_paths):] = 1

h_resize = 64                                   # リサイズ後の高さ(ピクセル数)
im = cv2.imread(test_paths[0])                  # データ読み込み
height = im.shape[0]                            # 元画像の高さ
width = im.shape[1]                             # 元画像の幅
w_resize = round(width * h_resize / height)     # リサイズ後の幅(ピクセル数)

def preprocess(im):
    im_cvt = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)        # RGBの順に変更
    im_res = cv2.resize(im_cvt, (w_resize, h_resize))   # 画素数の変更
    return im_res / 255                                 # 正規化

test_ims = np.zeros((n_test, h_resize, w_resize, 3), dtype=np.float32)
for i in range(n_test):
    im = cv2.imread(test_paths[i])
    test_ims[i, :, :, :] = preprocess(im)

model = keras.models.load_model("deep_learning_model.h5")
preds = model.predict(test_ims)
preds = np.squeeze(preds)
preds = preds.round().astype(np.int32)
# print(preds)

show_results(test_ims[::2], preds[::2])

"""予測根拠の可視化"""
def make_gradcam_heatmap(image, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    model_input = np.expand_dims(image, axis=0)

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(model_input)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]    # 行列積
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def show_gradcam(image, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")

    jec_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jec_colors[heatmap]

    jet_heatmap = cv2.resize(jet_heatmap, (w_resize, h_resize))

    superimposed_image = jet_heatmap * alpha + image * (1 - alpha)
    plt.matshow(superimposed_image)
    plt.show()


last_conv_layer_name = "conv2d_1"
heatmap = make_gradcam_heatmap(test_ims[-1:], model, last_conv_layer_name)

plt.matshow(heatmap)
plt.show()

show_gradcam(test_ims[-1], heatmap)

"""
Refrences:
    [1] Keras, Grad-CAM class activation visualization
        https://keras.io/examples/vision/grad_cam/ (Retrieved 2022.8.31)
"""
