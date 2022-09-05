import glob
import os
import numpy as np
import cv2

CLIPPED_PIXELS = 1024
MAX_SLIDES = 128
template_path = "./init_data/forPatternMatching/template.jpg"
test_data_dir = "./init_data/forPatternMatching/test"
save_dir = "./data/forPatternMatching/clipped"


def main():
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    image = cv2.imread(template_path)
    height = image.shape[0]
    width = image.shape[1]
    w_resize = CLIPPED_PIXELS + MAX_SLIDES * 2
    h_resize = height * w_resize // width
    resized = cv2.resize(image, (w_resize, h_resize))
    w_template = CLIPPED_PIXELS
    h_template = CLIPPED_PIXELS
    left = (w_resize - w_template) // 2
    right = (w_resize + w_template) // 2
    top = (h_resize - h_template) // 2
    bottom = (h_resize + h_template) // 2
    template = resized[top: bottom, left: right]
    cv2.imwrite(f"{save_dir}/template.jpg", template)

    test_data_paths = glob.glob(f"{test_data_dir}/*.jpg")

    for i, data_path in enumerate(test_data_paths):
        image = cv2.imread(data_path)
        target = cv2.resize(image, (w_resize, h_resize))
        res = cv2.matchTemplate(target, template, cv2.TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        left = min_loc[0]
        right = left + CLIPPED_PIXELS
        top = min_loc[1]
        bottom = top + CLIPPED_PIXELS
        saved = target[top: bottom, left: right]
        basename = os.path.basename(data_path)
        cv2.imwrite(f"{save_dir}/{basename}", saved)
        print(basename, min_val)


if __name__ == "__main__":
    main()
