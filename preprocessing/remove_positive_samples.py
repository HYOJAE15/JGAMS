import os 
import argparse

import cv2

from glob import glob 

import numpy as np

"""
This script removes all the positive samples from the dataset.
    positive samples: ones that have a label that is not all 0.
    dataset style: Cityscapes 
"""

parser = argparse.ArgumentParser()

parser.add_argument("--path", type=str, help="path to the dataset")


def main():
    # decode parser
    args = parser.parse_args()
    path = args.path

    # print the path to the dataset
    print("Path to the dataset: {}".format(path))
    
    # get all the training and validation images
    train_images = glob(os.path.join(path, "leftImg8bit/train/*_leftImg8bit.png"))
    val_images = glob(os.path.join(path, "leftImg8bit/val/*_leftImg8bit.png"))


    # print the number of training and validation images 
    print("Number of training images: {}".format(len(train_images)))
    print("Number of validation images: {}".format(len(val_images)))

    # loop through all the images
    loop_through_images(train_images)
    loop_through_images(val_images)

def loop_through_images(image_list: list):
    """
    Args: 
        image_list: list of images to loop through
    """
    for image_path in image_list:
        # check if the label is all 0
        label_path = image_path.replace("_leftImg8bit", "_gtFine_labelIds").replace("leftImg8bit", "gtFine")
        if check_positive_sample(label_path):
            os.remove(image_path)
            os.remove(label_path)

def check_positive_sample(label_path: str):
    """
    Args:
        label_path: path to the label

    Returns:
        True if the label is not all 0
    """
    # opencv imread unchanged 
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    # check if label is all zero 
    if np.all(label == 0):
        return False
    else:
        return True


if __name__ == "__main__":
    main()

