import numpy as np
import matplotlib.pyplot as plt


def most_square_shape(num_blocks, blockshape=(1,1)):
    x, y = blockshape
    num_x = np.ceil(np.sqrt(num_blocks * y / float(x)))
    num_y = np.ceil(num_blocks / num_x)
    return (num_x, num_y)


def visualize_grid(chunk, range=None):
    if chunk.ndim == 4 and chunk.shape[1] == 1:
        # this is a chunk with one input channel, select it
        chunk = chunk[:, 0]

    if chunk.ndim != 3:
        raise RuntimeError("Only 3D tensors or 4D tensors with one input channel are supported as input, input dimensionality is %d" % chunk.ndim)

    if range is None:
        range = chunk.min(), chunk.max()
    vmin, vmax = range

    patch_size = chunk.shape[1:]
    num_x, num_y = most_square_shape(chunk.shape[0], patch_size)
    
    #pad with zeros so that the number of filters equals num_x * num_y
    chunk_padded = np.zeros((num_x * num_y,) + patch_size)
    chunk_padded[:chunk.shape[0]] = chunk
    
    chunk_split = chunk_padded.reshape(num_x, num_y, patch_size[0], patch_size[1])
    chunk_with_border = np.ones((num_x, num_y, patch_size[0] + 1, patch_size[1] + 1)) * vmax
    chunk_with_border[:, :, :patch_size[0], :patch_size[1]] = chunk_split

    grid = chunk_with_border.transpose(0, 2, 1, 3).reshape(num_x * (patch_size[0] + 1), num_y * (patch_size[1] + 1))
    grid_with_left_border = np.ones((num_x * (patch_size[0] + 1) + 1, num_y * (patch_size[1] + 1) + 1)) * vmax
    grid_with_left_border[1:, 1:] = grid

    plt.imshow(grid_with_left_border, interpolation='nearest', cmap=plt.cm.binary, vmin=vmin, vmax=vmax)


