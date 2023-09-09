# https://github.com/owkin/FLamby/blob/main/flamby/datasets/fed_isic2019/dataset_creation_scripts/resize_images.py

from __future__ import division

import os

import numpy
import numpy as np
from PIL import Image
from joblib import Parallel, delayed
from tqdm import tqdm


def color_constancy(img, power=6, gamma=None):
    import cv2

    """
    Preprocessing step to make sure that the images appear with similar brightness
    and contrast.
    See this [link}(https://en.wikipedia.org/wiki/Color_constancy) for an explanation.
    Thank you to [Aman Arora](https://github.com/amaarora) for this
    [implementation](https://github.com/amaarora/melonama)
    Parameters
    ----------
    img: 3D numpy array, the original image
    power: int, degree of norm
    gamma: float, value of gamma correction
    """
    img_dtype = img.dtype

    if gamma is not None:
        img = img.astype("uint8")
        look_up_table = numpy.ones((256, 1), dtype="uint8") * 0
        for i in range(256):
            look_up_table[i][0] = 255 * pow(i / 255, 1 / gamma)
        img = cv2.LUT(img, look_up_table)

    img = img.astype("float32")
    img_power = numpy.power(img, power)
    rgb_vec = numpy.power(numpy.mean(img_power, (0, 1)), 1 / power)
    rgb_norm = numpy.sqrt(numpy.sum(numpy.power(rgb_vec, 2.0)))
    rgb_vec = rgb_vec / rgb_norm
    rgb_vec = 1 / (rgb_vec * numpy.sqrt(3))
    img = numpy.multiply(img, rgb_vec)

    return img.astype(img_dtype)


def resize_and_maintain(path, in_path, output_path, sz: tuple, cc):
    """Preprocessing of images
    Mantains aspect ratio fo input image. Possibility to add color constancy.
    Thank you to [Aman Arora](https://github.com/amaarora) for this
    [implementation](https://github.com/amaarora/melonama)
    Parameters
    ----------
    path : path to input image
    output_path : path to output image
    sz : tuple, shorter edge of resized image is sz[0]
    cc : color constancy is added if True
    """
    try:
        fn = os.path.basename(path)
        img = Image.open(path)
        size = sz[0]
        old_size = img.size
        ratio = float(size) / min(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        img = img.resize(new_size, resample=Image.BILINEAR)
        if cc:
            img = color_constancy(np.array(img))
            img = Image.fromarray(img)
        save_path = path.replace(in_path, output_path)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
    except:
        print('Error:', path)


def process(input_folder, output_folder, sz, cc):
    images = []
    for root, dirs, files in os.walk(input_folder):
        for name in files:
            if name.endswith('.jpg'):
                images.append(os.path.join(root, name))

    os.makedirs(output_folder, exist_ok=True)
    Parallel(n_jobs=48)(
        delayed(resize_and_maintain)(i, input_folder, output_folder, (sz, sz), cc)
        for i in tqdm(images)
    )


if __name__ == "__main__":
    input_folder = './dataset/fed_isic2019/ISIC_2019_Training_Input'
    output_folder = './dataset/fed_isic2019/ISIC_2019_Training_Input_preprocessed'
    cc = True  # only for isic2019
    sz = 224
    process(input_folder, output_folder, sz, cc)

    input_folder = './dataset/Resized 2015 & 2019 Blindness Detection Images/resized train 15'
    output_folder = './dataset/Resized 2015 & 2019 Blindness Detection Images/resized train 15_224'
    cc = False
    sz = 224
    process(input_folder, output_folder, sz, cc)

    input_folder = './dataset/Resized 2015 & 2019 Blindness Detection Images/resized test 15'
    output_folder = './dataset/Resized 2015 & 2019 Blindness Detection Images/resized test 15_224'
    cc = False
    sz = 224
    process(input_folder, output_folder, sz, cc)
