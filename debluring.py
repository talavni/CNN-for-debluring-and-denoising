import numpy as np
from scipy.ndimage.filters import convolve

from skimage.draw import line
import utils

def add_motion_blur(image, kernel_size, angle):
    """
    add motion blur to an image
    """
    kernal = motion_blur_kernel(kernel_size, angle)
    return convolve(image, kernal, mode='constant')


def motion_blur_kernel(kernel_size, angle):
    """Returns a 2D image kernel for motion blur effect.

    Arguments:
    kernel_size -- the height and width of the kernel. Controls strength of blur.
    angle -- angle in the range [0, np.pi) for the direction of the motion.
    """
    if kernel_size % 2 == 0:
        raise ValueError('kernel_size must be an odd number!')
    if angle < 0 or angle > np.pi:
        raise ValueError('angle must be between 0 (including) and pi (not including)')
    norm_angle = 2.0 * angle / np.pi
    if norm_angle > 1:
        norm_angle = 1 - norm_angle
    half_size = kernel_size // 2
    if abs(norm_angle) == 1:
        p1 = (half_size, 0)
        p2 = (half_size, kernel_size-1)
    else:
        alpha = np.tan(np.pi * 0.5 * norm_angle)
        if abs(norm_angle) <= 0.5:
            p1 = (2*half_size, half_size - int(round(alpha * half_size)))
            p2 = (kernel_size-1 - p1[0], kernel_size-1 - p1[1])
        else:
            alpha = np.tan(np.pi * 0.5 * (1-norm_angle))
            p1 = (half_size - int(round(alpha * half_size)), 2*half_size)
            p2 = (kernel_size - 1 - p1[0], kernel_size-1 - p1[1])
    rr, cc = line(p1[0], p1[1], p2[0], p2[1])
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    kernel[rr, cc] = 1.0
    kernel /= kernel.sum()
    return kernel


def random_motion_blur(image, list_of_kernel_sizes):
    """
    adds motion blur in random angle
    """
    angle = np.random.uniform(0, np.pi)
    kernel_num = np.random.randint(0, len(list_of_kernel_sizes))
    new_im = add_motion_blur(image, list_of_kernel_sizes[kernel_num], angle)
    new_im = new_im * 255
    new_im = np.round(new_im)
    new_im = new_im / 255
    return new_im


def learn_deblurring_model(images, num_res_blocks=5, quick_mode=False):
    """
    learn a cnn for deblurring an image
    :param images: list of images
    :param num_res_blocks: number of residua blocks
    :param quick_mode: for testing
    :return: the cnn
    """
    model = utils.build_nn_model(16, 16, 32, num_res_blocks)
    if quick_mode:
        utils.train_model(model, images, lambda x: random_motion_blur(x, [7]), 10, 3, 2, 30)
    else:
        utils.train_model(model, images, lambda x: random_motion_blur(x, [7]), 100, 100, 10, 1000)
    return model
