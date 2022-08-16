import os
"""
Tries to remove unnecessary black borders around the images, and
"trim" the images to they take up the entirety of the image.
"""

import cv2
import numpy as np
from PIL import Image
import warnings
from multiprocessing import Pool
from tqdm import tqdm

def trim(im):
    """
    Converts image to grayscale using cv2, then computes binary matrix
    of the pixels that are above a certain threshold, then takes out
    the first row where a certain percetage of the pixels are above the
    threshold will be the first clip point. Same idea for col, max row, max col.
    """

    percentage = 0.02 # porcentaje de valores mayores a cierto umbral

    img = np.array(im)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convertir a grises
    im = img_gray > 0.1 * np.mean(img_gray[img_gray != 0]) # Obtener una matriz binaria en escala de grises 
    # Que está un 0.1 por encima de la media, obteniendo una media sobre los valores de los pixeles que no 
    # son solo valores negros
    row_sums = np.sum(im, axis=1) # Sumar las filas
    col_sums = np.sum(im, axis=0) # Sumar las columnas
    rows = np.where(row_sums > img.shape[1] * percentage)[0]
    cols = np.where(col_sums > img.shape[0] * percentage)[0]
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    im_crop = img[min_row : max_row + 1, min_col : max_col + 1]
    return Image.fromarray(im_crop)
def resize_mantain_aspect(image, desired_size):
    """
    In this function we want to resize but we want to maintain the aspect ratio
    therefore it's going to pad the image with black
    """
    old_size = image.size # old_size[0] is in (width, height) format
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    im = image.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size - new_size[0]) // 2, (desired_size - new_size[1]) // 2))
    return new_im


def save_single(args):
    img_file, input_path_folder, output_path_folder, output_size = args
    image_original = Image.open(os.path.join(input_path_folder, img_file))
    image = trim(image_original)
    image = resize_mantain_aspect(image, desired_size=output_size[0])
    #image = image.resize(output_size)
    image.save(os.path.join(output_path_folder + img_file))

def fast_image_resize(input_path_folder, output_path_folder, output_size=None):
    """
    Uses multiprocessing to make it fast
    """
    if not output_size:
        warnings.warn("Need to specify output_size! For example: output_size=100")
        exit()
    if not os.path.exists(output_path_folder):
        os.makedirs(output_path_folder)
    
    jobs = [
        (file, input_path_folder, output_path_folder, output_size)
        for file in os.listdir(input_path_folder)
    ]

    with Pool() as p:
        list(tqdm(p.imap_unordered(save_single, jobs), total=len(jobs)))


if __name__ == "__main__":
    # fast_image_resize("./test/images/", "./images/test/resized_150/", output_size=(150,150)) 
    # fast_image_resize("./train/images/", "./images/train/resized_150/", output_size=(150,150))
    # fast_image_resize("./test/images/", "./images/test/resized_250/", output_size=(250,250)) 
    # fast_image_resize("./train/images/", "./images/train/resized_250/", output_size=(250,250))
    # fast_image_resize("./test/images/", "./images/test/resized_350/", output_size=(350,350)) 
    # fast_image_resize("./train/images/", "./images/train/resized_350/", output_size=(350,350))
    # fast_image_resize("./test/images/", "./images/test/resized_450/", output_size=(450,450)) 
    # fast_image_resize("./train/images/", "./images/train/resized_450/", output_size=(450,450))
    # fast_image_resize("./test/images/", "./images/test/resized_550/", output_size=(550,550)) 
    # fast_image_resize("./train/images/", "./images/train/resized_550/", output_size=(550,550))
    # fast_image_resize("./test/images/", "./images/test/resized_650/", output_size=(650,650)) 
    # fast_image_resize("./train/images/", "./images/train/resized_650/", output_size=(650,650))
    # fast_image_resize("./test/images/", "./images/test/resized_1000/", output_size=(1000,1000)) 
    fast_image_resize("./train/images/", "./images/train/resized_1000/", output_size=(1000,1000))