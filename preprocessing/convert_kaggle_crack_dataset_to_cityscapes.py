import numpy as np
import os 

import cv2

from glob import glob

from copy import deepcopy


"""
Convert the Kaggle Crack Dataset to Cityscapes format

Open Crack Dataset:
    Mask format 
        - 0, 0, 0: background -> mask is saved as jpg 
        - everything else: crack
    Dataset structure
        - train
            - images
                - **.jpg
            - masks
                - **.jpg

        - test
            - images
                - **.jpg
            - masks
                - **.jpg

Cityscapes Dataset:
    Dataset structure
        - leftImg8bit
            - train
                - **_leftImg8bit.png
            - val
                - **_leftImg8bit.png
        - gtFine
            - train
                - **_gtFine_labelIds.png
            - val
                - **_gtFine_labelIds.png
"""



### Global variables

# path to the dataset
OPENCRACK_PATH = r"\\172.16.113.151\UOS-SSaS Dropbox\05. Data\00. Benchmarks\23. KaggleCrack\02. Subset Split & Add Prefix"

# path to the new dataset
CITYSCAPES_PATH = r"\\172.16.113.151\UOS-SSaS Dropbox\05. Data\00. Benchmarks\23. KaggleCrack\03. Cityscapes Dataset"

SUBSETS = ["train", "val"]

def convert_mask_to_cityscapes_format(mask: np.ndarray):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    mask_temp = deepcopy(mask)
    mask[mask_temp > 100] = 1
    mask[mask_temp <= 100] = 0

    return mask


def main():
    
    # loop through all the subsets
    for subset in SUBSETS:
        # read the open crack dataset 
        train_images = glob(os.path.join(OPENCRACK_PATH, f"{subset}/images/*.jpg"))

        # loop through all the images
        for image_path in train_images:
            # get the image name
            image_name = os.path.basename(image_path)

            # get the mask path
            label_path = os.path.join(OPENCRACK_PATH, f"{subset}/masks", image_name)

            # read the image and mask
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

            # convert the mask to cityscapes format
            mask = convert_mask_to_cityscapes_format(mask)

            # save the image and mask
            # add suffix to the image name
            image_name = image_name.replace(".jpg", "_leftImg8bit.png")
            label_name = image_name.replace("_leftImg8bit", "_gtFine_labelIds")

            # create the directory if it does not exist
            if not os.path.exists(os.path.join(CITYSCAPES_PATH, f"leftImg8bit/{subset}")):
                os.makedirs(os.path.join(CITYSCAPES_PATH, f"leftImg8bit/{subset}"))
            
            if not os.path.exists(os.path.join(CITYSCAPES_PATH, f"gtFine/{subset}")):
                os.makedirs(os.path.join(CITYSCAPES_PATH, f"gtFine/{subset}"))
            
            # save the image and mask
            cv2.imwrite(os.path.join(CITYSCAPES_PATH, f"leftImg8bit/{subset}", image_name), image)
            cv2.imwrite(os.path.join(CITYSCAPES_PATH, f"gtFine/{subset}", label_name), mask)

if __name__ == "__main__":
    main()

