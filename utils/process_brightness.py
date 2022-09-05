import argparse
import os
import glob
import tqdm
import numpy as np
import cv2

SEED = 22
BRIGHTNESS_DEV = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        help="input directory path",
        type=str
    )
    parser.add_argument(
        "output_dir",
        help="output directory path",
        type=str
    )
    args = parser.parse_args()

    np.random.seed(SEED)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    input_paths = glob.glob(f"{args.input_dir}/*.jpg")
    n_data = len(input_paths)
    for i in tqdm.tqdm(range(n_data)):
        image = cv2.imread(input_paths[i])
        height = image.shape[0]
        width = image.shape[1]
        im_proc = image + np.random.normal(0, BRIGHTNESS_DEV)
        im_proc[im_proc < 0] = 0
        im_proc[im_proc > 255] = 255
        basename = os.path.basename(input_paths[i])
        cv2.imwrite(f"{args.output_dir}/{basename}", im_proc)


if __name__ == "__main__":
    main()

