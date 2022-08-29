import matplotlib.pyplot as plt


def show_images(images, nrows=3, ncols=4):
    """show images in a figure
    Args:
        images (numpy.ndarray): shape is (num_data, height, width, channels)
    """
    n_plots = min(len(images), nrows * ncols)
    fig, axs = plt.subplots(nrows, ncols)
    for i in range(n_plots):
        row = i // ncols
        col = i % ncols
        axs[row, col].axis("off")
        axs[row, col].imshow(images[i])
    plt.tight_layout()
    plt.show()
