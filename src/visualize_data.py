import matplotlib.pyplot as plt


plt.rcParams['font.family'] = "Meiryo"  # フォントの設定


def show_images(images, nrows=3, ncols=4):
    """show images in a figure
    Args:
        images (numpy.ndarray [int]): shape is (num_data, height, width, channels)
    """
    n_plots = min(len(images), nrows * ncols)
    fig, axs = plt.subplots(nrows, ncols)
    for i in range(n_plots):
        row = i // ncols
        col = i % ncols
        axs[row, col].axis("off")
        axs[row, col].imshow(images[i].astype(int))
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


def show_results(images, results, nrows=4, ncols=5):
    """show images in a figure
    Args:
        images (numpy.ndarray [int]): shape is (num_data, height, width, channels)
        results (list [str]): list of normality or anomaly
    """
    n_plots = min(len(images), nrows * ncols)
    fig, axs = plt.subplots(nrows, ncols)
    for i in range(n_plots):
        row = i // ncols
        col = i % ncols
        axs[row, col].set_title(results[i])
        axs[row, col].axis("off")
        axs[row, col].imshow(images[i].astype(int))
    plt.tight_layout()
    plt.show()
