""" Launcher of most common functions
"""

import cv2
import numpy as np
import dlutils
import sys


source_path_A = '/media/antonio/Data/DataSets/Raw/Selfie-dataset/images'
source_path_B = '/media/antonio/Data/DataSets/Raw/LLD-logo-files'
destination_path = '/media/antonio/Data/DataSets/Projects/Stickerizer/FaceToSticker'


def gen_data_set_GAN():
    """ Generate a GAN structure of folders
    """
    dlutils.generate_dataset_GAN(
        source_path_A, source_path_B, destination_path, 0.2, 0.1)


def unsharp_one_channel(image, sigma, k):

    # Median filtering
    image_blurred = cv2.medianBlur(image, sigma)

    # Calculate the Laplacian
    lap = cv2.Laplacian(image_blurred, cv2.CV_64F)

    # Calculate the sharpened image
    sharp = image-k*lap

    # Saturate the pixels in either direction
    sharp[sharp > 255] = 255
    sharp[sharp < 0] = 0

    return sharp


def unsharp_RGB(image, sigma, k):
    sharp = np.zeros_like(image)
    for i in range(3):
        sharp[:, :, i] = unsharp_one_channel(image[:, :, i], 5, 0.8)

    return sharp


def higher_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final


def main():
    filename = sys.argv[1]

    name, _ = filename.split('.')

    image = cv2.imread(filename)
    size = min(10, image.shape[0]//150)
    #image_smooth = cv2.ximgproc.edgePreservingFilter(image, size, 14)
    #image_smooth = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    #image_smooth = image
    image_smooth = cv2.medianBlur(image, 9)
    cv2.imwrite(name + '_result_smooth.jpg', image_smooth)

    # kernel_sharpening = np.array([[-1, -1, -1],
    #                               [-1, 9, -1],
    #                               [-1, -1, -1]])

    # sharpened = cv2.filter2D(image_smooth, -1, kernel_sharpening)
    sharpened = unsharp_RGB(image_smooth, 5.0, 1.0)
    # sharpened = image_smooth
    cv2.imwrite(name + '_result_sharp.jpg', sharpened)
    kernel = np.ones((5, 5), np.uint8)
    ori_opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    opening = cv2.morphologyEx(sharpened, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(name + '_result_ori_open.jpg', ori_opening)
    cv2.imwrite(name + '_result_open.jpg', opening)
    ori_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    close = cv2.morphologyEx(sharpened, cv2.MORPH_CLOSE, kernel)
    open_close = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
    close_open = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    cv2.imwrite(name + '_result_ori_close.jpg', ori_close)
    cv2.imwrite(name + '_result_close.jpg', close)
    cv2.imwrite(name + '_result_open_close.jpg', open_close)
    cv2.imwrite(name + '_result_close_open.jpg', close_open)


if __name__ == '__main__':
    dlutils.downscale_upscale_folders(path=source_path_B,
                                      down_path='../LLD-logo-files-downscaled',
                                      resize_path='../LLD-logo-files-resized')
