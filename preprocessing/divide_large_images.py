import os 
import argparse

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2,40).__str__()

import cv2

from glob import glob 

import numpy as np
import slidingwindow as sw

from modules.utils import imread, imwrite, generateForNumberOfWindows

"""
This script divides the large images into smaller images.
"""

parser = argparse.ArgumentParser()

parser.add_argument("--source_path", type=str, help="path of source dataset")
parser.add_argument("--target_path", type=str, help="path of target dataset")
parser.add_argument("--window_size", type=int, default=8000, help="window size")

def main():
    # decode parser
    args = parser.parse_args()
    source_path = args.source_path
    target_path = args.target_path
    window_size = args.window_size

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

    # loop through all the images
    loop_through_images(train_images, target_path, window_size, dataset='train')
    loop_through_images(val_images, target_path, window_size, dataset='val')
    loop_through_images(test_images, target_path, window_size, dataset='test')


def loop_through_images(image_list, target_path, window_size, dataset='train'):
    """
    Args: 
        image_list: list of images to loop through
        target_path: path to the target dataset,
        window_size: size of the window
        dataset: name of the dataset
    """
    # check image_list is empty and pass 
    if len(image_list) == 0:
        return None 
    
    
    for image_path in image_list:
        label_path = image_path.replace("_leftImg8bit", "_gtFine_labelIds").replace("leftImg8bit", "gtFine")

        # crop images and labels into smaller images
        image = imread(image_path)
        label = imread(label_path)

        count_x = image.shape[1] // window_size + 1
        count_y = image.shape[0] // window_size + 1
        window_count = (count_y,  count_x)
        windows = generateForNumberOfWindows(image, sw.DimOrder.HeightWidthChannel, window_count, 0)

        for _x in range(count_x):
            for _y in range(count_y):

                window_idx = _x * count_y + _y 
                        
                window = windows[window_idx]

                image_subset = image[ window.indices()]
                label_subset = label[ window.indices()]

                # filename of image_path 
                image_filename = os.path.basename(image_path)
                label_filename = os.path.basename(label_path)
                
                # save the images
                image_filename = image_filename.replace("leftImg8bit.png", f"{_y}_{_x}_leftImg8bit.png")
                label_filename = label_filename.replace("gtFine_labelIds.png", f"{_y}_{_x}_gtFine_labelIds.png")

                # destination path of the image
                image_dest_path = os.path.join(target_path, "leftImg8bit", dataset, image_filename)
                label_dest_path = os.path.join(target_path, "gtFine", dataset, label_filename)

                # create the directory if it does not exist
                os.makedirs(os.path.dirname(image_dest_path), exist_ok=True)
                os.makedirs(os.path.dirname(label_dest_path), exist_ok=True)

                # save the images
                imwrite(image_dest_path, image_subset)
                imwrite(label_dest_path, label_subset)
                


if __name__ == "__main__":
    main()