"""前準備"""
# import pdb
import glob
import cv2
import numpy as np

from src.visualize_data import show_image, show_images, show_similarities, show_results


def read_rgb_image(image_path):
    image = cv2.imread(image_path)
    im_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return im_cvt


template_image = read_rgb_image("outsource/Hazelnut/train/good/o001.jpg")
show_image(template_image)

height = template_image.shape[0]
width = template_image.shape[1]
threshold = 1

image_paths = glob.glob("outsource/Hazelnut/test/good/*.jpg")
image_paths.extend(glob.glob("outsource/Hazelnut/test/crack/*.jpg"))
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
