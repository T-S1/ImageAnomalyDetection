import argparse
import glob
import os
import shutil
import numpy as np


N_NORMAL = 700
N_ANOMALY = 500
dataset_dir = "./init_data/forDeepLearning"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "normal_dir",
        help="original normal dataset directory path",
        type=str
    )
    parser.add_argument(
        "anomaly_dir",
        help="original anomaly dataset directory path",
        type=str
    )
    args = parser.parse_args()

    if not os.path.isdir(f"{dataset_dir}/normal"):
        os.makedirs(f"{dataset_dir}/normal")
    if not os.path.isdir(f"{dataset_dir}/anomaly"):
        os.makedirs(f"{dataset_dir}/anomaly")

    normal_paths = glob.glob(args.normal_dir + "/**/*.JPG", recursive=True)
    anomaly_paths = glob.glob(args.anomaly_dir + "/**/*.JPG", recursive=True)

    print("正常データ数:", len(normal_paths))
    print("異常データ数:", len(anomaly_paths))

    normal_idxs = np.int64(np.linspace(0, len(normal_paths), N_NORMAL, endpoint=False))
    anomaly_idxs = np.int64(np.linspace(0, len(anomaly_paths), N_ANOMALY, endpoint=False))

    picked_norm_paths = [normal_paths[idx] for idx in normal_idxs]
    picked_anom_paths = [anomaly_paths[idx] for idx in anomaly_idxs]

    for i in range(len(picked_norm_paths)):
        shutil.copy(picked_norm_paths[i], f"{dataset_dir}/normal/{i:05}.jpg")

    for i in range(len(picked_anom_paths)):
        shutil.copy(picked_anom_paths[i], f"{dataset_dir}/anomaly/{i:05}.jpg")


if __name__ == "__main__":
    main()
