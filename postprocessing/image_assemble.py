import os
import argparse

from glob import glob
from tqdm import tqdm

import slidingwindow as sw

import cv2
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("source_path", help="file path to store segmented images", type=str)
parser.add_argument("store_path", help="file path to save images", type=str)
parser.add_argument("--source_ext", default='jpg', type=str, help="file extension of the original images")
parser.add_argument("--store_ext", default='jpg', type=str, help="file extension of the segmented images")
parser.add_argument("--crop_size", default=1000, type=int, help="image size used during segmentation")
parser.add_argument("--date_stamp", default=True, type=bool, help="Include the date and time of script operation in the file name")

args = parser.parse_args()


def imread(path):
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    nparray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(nparray, cv2.IMREAD_UNCHANGED)
    return bgrImage


def imwrite(path, image):
    _, ext = os.path.splitext(path)
    cv2.imencode(ext, image)[1].tofile(path)


def main():
    source_path = args.source_path
    store_path = args.store_path
    source_ext = args.source_ext
    store_ext = args.store_ext
    crop_size = args.crop_size
    date_stamp = args.date_stamp

    segmented_images = glob(os.path.join(source_path, f'*.{source_ext}'))

    image_ex = segmented_images[0]
    image_ex_name = os.path.basename(image_ex)
    img_shape_0, img_shape_1, x_start, y_start = map(int, image_ex_name.split('_')[-5:-1])  # Extract x_start, y_start, and crop_size from the file name
    
    reassembled_image = np.zeros((img_shape_0, img_shape_1, 3), dtype=np.uint8)  # Assuming 3 channels, adjust accordingly
    
    reassembled_image_path = os.path.join(store_path, f'reassembled.png')

    for segmented_image_path in tqdm(segmented_images, desc='Reassembling'):
        file_name = os.path.basename(segmented_image_path)

        img_shape_0, img_shape_1, x_start, y_start = map(int, file_name.split('_')[-5:-1])  # Extract x_start, y_start, and crop_size from the file name

        
        subset = imread(segmented_image_path)
        reassembled_image[x_start:(x_start + crop_size), y_start:(y_start + crop_size), :] = subset

        imwrite(reassembled_image_path, reassembled_image)


if __name__ == '__main__':
    main()