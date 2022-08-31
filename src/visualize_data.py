import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = "Meiryo"  # フォントの設定


def show_images(images, nrows=3, ncols=4):
    """show images in a figure
    Args:
        images (numpy.ndarray [int]): shape is (num_data, height, width, channels)
    """
    if np.amax(images) > 1:
        ims = images.astype(np.int32)
    else:
        ims = images.astype(np.float64)
    n_plots = min(len(ims), nrows * ncols)
    fig, axs = plt.subplots(nrows, ncols)
    for i in range(n_plots):
        row = i // ncols
        col = i % ncols
        axs[row, col].axis("off")
        axs[row, col].imshow(ims[i])
    plt.tight_layout()
    plt.show()


def show_similarities(sims, threshold=None):
    """show similarities for each data
    Args:
        sims (numpy.ndarray): shape is (num_data)
        threshold (float): threshold for anomaly detection
    """
    plt.xlabel("データ番号")
    plt.ylabel("類似度")
    plt.plot(sims, "o")
    if threshold is not None:
        plt.axhline(threshold)
    plt.tight_layout()
    plt.show()


def show_results(images, preds, gts=None, nrows=4, ncols=5, names=["正常", "異常"]):
    """show images in a figure
    Args:
        images (numpy.ndarray [int]): shape is (num_data, height, width, channels)
        labels (list [str]): list of normal or anomaly
    """
    if np.amax(images) > 1:
        ims = images.astype(np.int32)
    n_plots = min(len(images), nrows * ncols)
    fig, axs = plt.subplots(nrows, ncols)
    for i in range(n_plots):
        row = i // ncols
        col = i % ncols
        if gts is None:
            title = f"{names[preds[i]]}"
        if gts is not None:
            title += f"予測: {names[preds[i]]}/ 正解: {names[gts[i]]}"
        axs[row, col].set_title(title)
        axs[row, col].axis("off")
        axs[row, col].imshow(images[i])
    plt.tight_layout()
    plt.show()


def show_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(history.history["accuracy"])
    ax1.plot(history.history["val_accuracy"])
    ax1.set_title("Model accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(["Train", "Validation"], loc="upper left")

    ax2.plot(history.history["loss"])
    ax2.plot(history.history["val_loss"])
    ax2.set_title("Model accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend(["Train", "Validation"], loc="upper left")

    plt.tight_layout()
    plt.show()

