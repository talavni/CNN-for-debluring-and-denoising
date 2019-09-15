import numpy as np
import utils

def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    adds gaussian noise to an image
    """
    sigma = np.random.uniform(min_sigma, max_sigma)
    noise = np.random.normal(0, sigma, image.shape)
    new_im = image + noise
    new_im = new_im * 255
    new_im = np.round(new_im)
    new_im = new_im / 255
    return np.clip(new_im, 0 ,1)


def learn_denoising_model(images, num_res_blocks=5, quick_mode=False):
    """
    learn a cnn for denoising an image
    :param images: list of images
    :param num_res_blocks: number of residua blocks
    :param quick_mode: for testing
    :return: the cnn
    """
    model = utils.build_nn_model(24, 24, 48, num_res_blocks)
    if quick_mode:
        utils.train_model(model, images, lambda x: add_gaussian_noise(x, 0, 0.2), 10, 3, 2, 30)
    else:
        utils.train_model(model, images, lambda x: add_gaussian_noise(x, 0, 0.2), 100, 100, 5, 1000)
    return model
