
import os
import argparse

from glob import glob 
from tqdm import tqdm

from datetime import datetime

import slidingwindow as sw

import cv2
import numpy as np


parser = argparse.ArgumentParser()

parser.add_argument("source_img_folder", help="folder path to images to be cropped", type=str)
parser.add_argument("store_path", help="file path to store cropped images", type=str)
parser.add_argument("--source_ext", default='png', type=str,
                    help="file extension to be cropped")
parser.add_argument("--store_ext", default='png', type=str,
                    help="file extension to be cropped")
parser.add_argument("--crop_size", default=3000, type=int,
                    help="image size to be cropped")
parser.add_argument("--date_stamp", default=True, type=bool,
                    help="Include the date and time of script operation in the file name")


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

def make_cityscapes_format_imagetype (image, save_dir, image_type) :
    temp_img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    gt = np.zeros((temp_img.shape[0], temp_img.shape[1]), dtype=np.uint8)
    
    save_dir_img = os.path.join(save_dir, 'leftImg8bit')
    save_dir_gt = os.path.join(save_dir, 'gtFine')

    os.makedirs(save_dir_img, exist_ok=True)
    os.makedirs(save_dir_gt, exist_ok=True)

    img_filename = os.path.basename(image)
    img_filename = img_filename.replace(f'.{image_type}', '_leftImg8bit.png')
    
    gt_filename = img_filename.replace('_leftImg8bit.png', '_gtFine_labelIds.png')

    is_success, org_img = cv2.imencode(".png", temp_img)
    org_img.tofile(os.path.join(save_dir_img, img_filename))

    is_success, gt_img = cv2.imencode(".png", gt)
    gt_img.tofile(os.path.join(save_dir_gt, gt_filename))
    gt_path = os.path.join(save_dir_gt, gt_filename) 
    
    return gt_path

def main():

    source_path = args.source_img_folder
    store_path = args.store_path
    source_ext = args.source_ext
    store_ext = args.store_ext
    crop_size = args.crop_size
    date_stamp = args.date_stamp
    
    file_list = glob(os.path.join(source_path, f"*.{source_ext}"))

    for file in tqdm(file_list) :
        
        img = imread(file)
        print(os.path.basename(file))
        file_name = os.path.basename(file).split('\\')[-1]
        print(file_name)

        windows = sw.generate(img, sw.DimOrder.HeightWidthChannel, crop_size, 0)

        for window in windows:
            subset = img[ window.indices()]

            x_start = window.indices()[0].start
            y_start = window.indices()[1].start
            
            
            file_store_path = os.path.join(store_path, file_name)
            if date_stamp : 
                datetime_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                img_shape_0 = img.shape[0]
                img_shape_1 = img.shape[1]
                file_store_path = file_store_path.replace(f'.{source_ext}', f'_{datetime_stamp}_{img_shape_0}_{img_shape_1}_{x_start}_{y_start}_{crop_size}.{store_ext}')
            else: 
                file_store_path = file_store_path.replace(f'.{source_ext}', f'_{img_shape_0}_{img_shape_1}_{x_start}_{y_start}_{crop_size}.{store_ext}')
            
            imwrite(file_store_path, subset)



if __name__ == '__main__':
    

    main()