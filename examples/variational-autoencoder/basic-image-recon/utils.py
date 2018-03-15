import os, struct
import numpy as np
import math


# adapted from https://github.com/clab/dynet/blob/master/examples/mnist/mnist-autobatch.py
def load_mnist(dataset, path):
    """
    wget -O - http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz | gunzip > train-images-idx3-ubyte
    wget -O - http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz | gunzip > train-labels-idx1-ubyte
    wget -O - http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz | gunzip > t10k-images-idx3-ubyte
    wget -O - http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz | gunzip > t10k-labels-idx1-ubyte
    """
    if dataset is "training":
        fname_img = os.path.join(path, "train-images-idx3-ubyte")
        fname_lbl = os.path.join(path, "train-labels-idx1-ubyte")
    elif dataset is "testing":
        fname_img = os.path.join(path, "t10k-images-idx3-ubyte")
        fname_lbl = os.path.join(path, "t10k-labels-idx1-ubyte")
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in numpy arrays
    with open(fname_lbl, "rb") as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        labels = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, "rb") as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        images = np.multiply(
            np.fromfile(fimg, dtype=np.uint8).reshape(len(labels), rows*cols),
            1.0 / 255.0)

    get_instance = lambda idx: (labels[idx], images[idx].reshape(1, 28, 28))

    # Create an iterator which returns each image in turn
    # for i in range(len(labels)):
    #     yield get_instance(i)

    size_reset = lambda x: x.reshape(1, 28, 28)
    return list(map(size_reset, images))


def make_grid(tensor, nrow=8, padding=2, pad_value=0):
    """Make a grid of images, via numpy.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        pad_value (float, optional): Value for the padded pixels.

    """
    if not (isinstance(tensor, np.ndarray) or
            (isinstance(tensor, list) and all(isinstance(t, np.ndarray) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = np.stack(tensor, 0)

    if tensor.ndim == 2:  # single image H x W
        tensor = tensor.reshape((1, tensor.shape[0], tensor.shape[1]))

    if tensor.ndim == 3:
        if tensor.shape[0] == 1:  # if single-channel, single image, convert to 3-channel
            tensor = np.concatenate((tensor, tensor, tensor), 0)
        tensor = tensor.reshape((1, tensor.shape[0], tensor.shape[1], tensor.shape[2]))

    if tensor.ndim == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = np.concatenate((tensor, tensor, tensor), 1)

    if tensor.shape[0] == 1:
        return np.squeeze(tensor)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    grid = np.ones((3, height * ymaps + padding, width * xmaps + padding)) * pad_value
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, y * height + padding:(y+1) * height,\
                 x * width + padding:(x+1) * width] = tensor[k]
            k = k + 1
    return grid


def pre_pillow_float_img_process(float_img):
    img = float_img * 255
    img = img.clip(0, 255).astype('uint8').transpose(1, 2, 0)
    return img


def save_image(tensor, filename, nrow=8, padding=2, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value)
    im = Image.fromarray(pre_pillow_float_img_process(grid))
    im.save(filename)
