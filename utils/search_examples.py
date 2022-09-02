import argparse
import glob
import tqdm
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "ref_dir",
        help="directory path of reference images",
        type=str
    )
    parser.add_argument(
        "dif_dir",
        help="directory path of different images to references",
        type=str
    )
    args = parser.parse_args()

    ref_paths = glob.glob(f"{args.ref_dir}/*.jpg")
    dif_paths = glob.glob(f"{args.dif_dir}/*.jpg")[18:25]
    n_ref = len(ref_paths)
    n_dif = len(dif_paths)

    ref_image_list = []
    for image_path in tqdm.tqdm(ref_paths):
        image = cv2.imread(image_path)
        ref_image_list.append(image)

    ref_images = np.stack(ref_image_list)
    mean_image = np.mean(ref_images, axis=0)
    # mean_image = ref_images[13]
    expanded_mean = np.expand_dims(mean_image, axis=0)

    # plt.imshow(mean_image.astype(np.int64))
    # plt.show()

    height = mean_image.shape[0]
    width = mean_image.shape[1]

    sads = np.sum(np.abs(ref_images - expanded_mean), axis=(1, 2, 3)) / (height * width)
    idxs = np.arange(n_ref, dtype=np.int64)
    idxs = sorted(idxs, key=lambda x: sads[x])

    sorted_ref_paths = [ref_paths[idx] for idx in idxs]

    print(sorted_ref_paths)
    print(sads[idxs])
    print(idxs)

    dif_image_list = []
    for image_path in tqdm.tqdm(dif_paths):
        image = cv2.imread(image_path)
        dif_image_list.append(image)

    dif_images = np.stack(dif_image_list)
    sads = np.sum(np.abs(dif_images - expanded_mean), axis=(1, 2, 3)) / (height * width)
    idxs = np.arange(n_dif, dtype=np.int64)
    idxs = sorted(idxs, key=lambda x: sads[x])
    # idxs = idxs[-20:]

    sorted_dif_paths = [dif_paths[idx] for idx in idxs]

    print(sorted_dif_paths)
    print(sads[idxs])
    print(idxs)

    # cv2.imwrite("./data/mean_image.jpg", mean_image)


if __name__ == "__main__":
    main()
