import numpy as np

import cv2 

import os 

from glob import glob

from tqdm import tqdm



def rotate_imgs(imgs: np.ndarray, angle: int):
    """
    Rotate images by a given angle

    Parameters
    ----------
    imgs : np.ndarray
        Images to rotate
    angle : int
        Angle to rotate the images

    Returns
    -------
    np.ndarray
        Rotated images
    """
    rotated_imgs = []
    for img in imgs:
        rotated_imgs.append(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))

    return np.array(rotated_imgs)


def rotate_masks(masks: np.ndarray, angle: int):
    """
    Rotate masks by a given angle

    Parameters
    ----------
    masks : np.ndarray
        Masks to rotate
    angle : int
        Angle to rotate the masks

    Returns
    -------
    np.ndarray
        Rotated masks
    """
    rotated_masks = []
    for mask in masks:
        rotated_masks.append(cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE))

    return np.array(rotated_masks)


IMG_DIR = r"\\172.16.113.151\UOS-SSaS Dropbox\05. Data\00. Benchmarks\23. KaggleCrack\03. Cityscapes Dataset\gtFine\temp"

def main():
    # path to the dataset
    
    img_list = glob(os.path.join(IMG_DIR, "*.png"))

    for img_path in tqdm(img_list):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        # rotated_img = cv2.rotate(rotated_img, cv2.ROTATE_90_CLOCKWISE)
        
        if rotated_img.ndim == 3:
            rotated_img = rotated_img[:, :, 0]
        cv2.imwrite(img_path, rotated_img)


if __name__ == "__main__":
    main()