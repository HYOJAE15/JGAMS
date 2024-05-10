import numpy as np

import cv2

import os

from glob import glob

from tqdm import tqdm


"""
Resize Images in Cityscapes Dataset 

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

ROOT_DIR = r"\\172.16.113.151\UOS-SSaS Dropbox\05. Data\00. Benchmarks\23. KaggleCrack\04. Cityscapes Dataset Resize"

SUBSETS = ["train", "val"]

def main():

    # read image list 

    for subset in SUBSETS:

        img_list = glob(os.path.join(ROOT_DIR, f"leftImg8bit/{subset}/*.png"))


        for img_path in tqdm(img_list):

            # get image name 
            img_name = os.path.basename(img_path)

            # get label name 
            label_name = img_name.replace("leftImg8bit", "gtFine_labelIds")
            # get label path 
            label_path = os.path.join(ROOT_DIR, f"gtFine/{subset}", label_name)


            # read image

            img = cv2.imread(img_path)

            # read label
            label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

            # resize image and label
            # set shorter side to 1024

            h, w, _ = img.shape

            if h > w:
                new_h = 1024
                new_w = int(w * 1024 / h)
            else:
                new_w = 1024
                new_h = int(h * 1024 / w)

            img = cv2.resize(img, (new_w, new_h))
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            # save image and label
            cv2.imwrite(img_path, img)
            cv2.imwrite(label_path, label)



if __name__ == "__main__":
    main()