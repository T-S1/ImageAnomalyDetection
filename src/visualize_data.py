import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.family'] = "Meiryo"  # フォントの設定


def show_image(image):
    if np.amax(image) > 1:
        im = image.astype(np.int32)
    else:
        im = image
    fig = plt.figure()
    plt.imshow(im)
    plt.axis("off")
    plt.show()


def show_images(images, nrows=3, ncols=4):
    if np.amax(images) > 1:
        ims = images.astype(np.int32)
    else:
        ims = images.astype(np.float64)
    n_plots = min(len(ims), nrows * ncols)
    fig = plt.figure()
    for i in range(n_plots):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.axis("off")
        ax.imshow(ims[i])
    plt.tight_layout()
    plt.show()


def show_similarities(sims, threshold=None):
    plt.xlabel("データ番号")
    plt.ylabel("類似度")
    plt.plot(sims, "o")
    if threshold is not None:
        plt.axhline(threshold)
    plt.tight_layout()
    plt.show()


def show_results(images, labels, values=None, nrows=3, ncols=4, names=["正常", "異常"], value_name="評価値"):
    if np.amax(images) > 1:
        ims = images.astype(np.int64)
    else:
        ims = images
    n_plots = min(len(ims), nrows * ncols)
    fig = plt.figure()
    for i in range(n_plots):
        pred = int(labels[i])
        title = f"予測: {names[pred]}"
        if values is not None:
            title += f", {value_name}: {values[i]:.4f}"
        ax = fig.add_subplot(nrows, ncols, i + 1)
        ax.set_title(title)
        ax.axis("off")
        ax.imshow(ims[i])
    plt.tight_layout()
    plt.show()


def show_history(history, save_path=None):
    keys = list(history.history.keys())
    metrics_key = keys[1]
    val_idx = len(keys) // 2 + 1
    val_metrics_key = keys[val_idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.8))

    ax1.plot(history.history[metrics_key])
    ax1.plot(history.history[val_metrics_key])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(metrics_key)
    ax1.legend(["Train", "Validation"], loc="lower right")

    ax2.plot(history.history["loss"])
    ax2.plot(history.history["val_loss"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(["Train", "Validation"], loc="upper right")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


class RT_Drawer():
    def __init__(self, names=["正常", "異常"], colors=["g", "r"]):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(9.6, 4.8))
        self.names = names
        self.colors = colors

    def update(self, image, label):
        self.ax1.imshow(image)
        self.ax1.set_title("取得画像")
        self.ax1.axis("off")

        self.ax2.text(
            0.5, 0.5, self.names[label], fontsize=30, color=self.colors[label],
            horizontalalignment="center", verticalalignment="center"
        )
        self.ax2.set_title("予測結果")
        self.ax2.axis("off")
        self.ax2.set_facecolor(self.colors[label])

        plt.pause(3)
        self.ax1.cla()
        self.ax2.cla()

    def cleanup(self):
        plt.close()
