import glob
import os
import tqdm
import numpy as np
import cv2

RESIZED_PIXELS = 1024

def main():
    sep = os.path.sep
    for image_path in tqdm.tqdm(glob.glob(os.path.join(".", "init_data", "**", "*.jpg"), recursive=True)):
        splitted = image_path.split(sep)
        # print(splitted)
        if splitted[0] == ".":
            splitted[1] = "data"
        else:
            splitted[0] = "data"
        save_path = sep.join(splitted)

        if os.path.isfile(save_path):
            continue

        image = cv2.imread(image_path)
        height = image.shape[0]
        width = image.shape[1]
        if height < width:
            left = (width - height) // 2
            right = (width + height) // 2
            trimmed = image[:, left: right]
        else:
            up = (height - width) // 2
            bottom = (height + width) // 2
            trimmed = image[up: bottom, :]
        resized = cv2.resize(trimmed, (RESIZED_PIXELS, RESIZED_PIXELS))

        save_dir = os.path.dirname(save_path)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cv2.imwrite(f"{save_path}", resized)


if __name__ == "__main__":
    main()
