import os 
import re
import argparse

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2,40).__str__()

import cv2

from glob import glob 

import numpy as np
import slidingwindow as sw

from modules.utils import imread, imwrite, generateForNumberOfWindows

"""
This script concatenates the small images into large images.
"""

parser = argparse.ArgumentParser()

parser.add_argument("--source_path", type=str, help="path of source dataset")
parser.add_argument("--target_path", type=str, help="path of target dataset")

def main():
    # decode parser
    args = parser.parse_args()
    source_path = args.source_path
    target_path = args.target_path

    # print the path to the dataset
    print("Path to the dataset: {}".format(source_path))
    
    # get all the training and validation images
    train_images = glob(os.path.join(source_path, "leftImg8bit/train/*_leftImg8bit.png"))
    val_images = glob(os.path.join(source_path, "leftImg8bit/val/*_leftImg8bit.png"))
    test_images = glob(os.path.join(source_path, "leftImg8bit/test/*_leftImg8bit.png"))


    # print the number of training and validation images 
    print("Number of training images: {}".format(len(train_images)))
    print("Number of validation images: {}".format(len(val_images)))
    print("Number of test images: {}".format(len(test_images)))

    unique_filenames = get_unique_filenames(train_images)
    concat_images(train_images, unique_filenames, target_path, 'train')

    unique_filenames = get_unique_filenames(val_images)
    concat_images(val_images, unique_filenames, target_path, 'val')

    unique_filenames = get_unique_filenames(test_images)
    concat_images(test_images, unique_filenames, target_path, 'test')

    
def get_unique_filenames(image_list):
    """
    Args: 
        image_list: list of images to loop through
    Returns: 
        unique_filenames: list of unique filenames
    """
    # get unique filenames
    unique_filenames = []
    for image in image_list:
        # file suffix format is _*_*_leftImg8bit.png

        filename = os.path.basename(image)
        filename, _ = os.path.splitext(filename)
        match = re.search(r"^(.+)_[\d_]_[\d_]_+", filename)
        if match:
            filename = match.group(1)
        
        if filename not in unique_filenames:
            unique_filenames.append(filename)

    return unique_filenames
     
def get_num_of_col_and_row(image_list):
    """
    Args:
        image_list: list of images to loop through
    Returns:
        num_of_col: number of columns
        num_of_row: number of rows
    """
    # get the column and row of images
    columns = []
    rows = []

    for img in image_list: 
        # get the column and row of images
        filename = os.path.basename(img)
        filename, _ = os.path.splitext(filename)
        match = re.search(r"^.+_(\d+)_(\d+)_+", filename)
        if match:
            rows.append(int(match.group(1)))
            columns.append(int(match.group(2)))
        else:
            raise ValueError("column and row cannot be found")
        
    return max(columns)+1, max(rows)+1

def concat_images(image_list, unique_filenames, target_path, dataset='train'):
    """
    Args:
        image_list: list of images to loop through
        unique_filenames: list of unique filenames
        target_path: path to save the concatenated images
        dataset: train / val / test
    """

    # check image_list is empty and pass 
    if len(image_list) == 0:
        return None 

    for u_f in unique_filenames:
        # find all the images starting with u_f
        images = []
        for image in image_list:
            filename = os.path.basename(image)
            
            if filename.startswith(u_f):
                images.append(image)

        images = sorted(images)
        
        # get the column and row of images 
        num_column, num_row = get_num_of_col_and_row(images)

        print("Number of columns: {}".format(num_column))
        print("Number of rows: {}".format(num_row))

        # shape of concatenated image
        # assume all the images have the same shape
        img = imread(images[0])
        height, width = img.shape[0] * num_row, img.shape[1] * num_column

        # concatenate images
        img_concat = np.zeros((height, width, 3), dtype=np.uint8)
        label_concat = np.zeros((height, width), dtype=np.uint8)

        for r in range(num_row):
            for c in range(num_column):
                image_path = images[r*num_column+c]
                label_path = image_path.replace("_leftImg8bit", "_gtFine_labelIds").replace("leftImg8bit", "gtFine")

                img = imread(image_path)
                label = imread(label_path)

                img_concat[r*img.shape[0]:(r+1)*img.shape[0], c*img.shape[1]:(c+1)*img.shape[1], :] = img
                label_concat[r*img.shape[0]:(r+1)*img.shape[0], c*img.shape[1]:(c+1)*img.shape[1]] = label

        image_dest_path = os.path.join(target_path, "leftImg8bit", dataset, u_f + "_leftImg8bit.png")
        label_dest_path = os.path.join(target_path, "gtFine", dataset, u_f + "_gtFine_labelIds.png")

        # create the directory if it does not exist
        os.makedirs(os.path.dirname(image_dest_path), exist_ok=True)
        os.makedirs(os.path.dirname(label_dest_path), exist_ok=True)

        imwrite(image_dest_path, img_concat)
        imwrite(label_dest_path, label_concat)


if __name__ == "__main__":
    main()