import cv2 

from glob import glob

import numpy as np

import os

import sys

SRX_DIR = r'\\172.16.113.151\UOS-SSaS Dropbox\05. Data\02. Training&Test\012. General_Crack\02. Negative Samples\v0.1.1'

CLEAN_DIR = r'\\172.16.113.151\UOS-SSaS Dropbox\05. Data\02. Training&Test\012. General_Crack\03. Clean Samples\v0.1.1'

RST_DIR = r'\\172.16.113.151\UOS-SSaS Dropbox\05. Data\02. Training&Test\012. General_Crack\02. Negative Samples\v0.1.2'

SPLIT = 'val'


def main():
    # concat images in two different cityscapes 

    # read image list CITYSCAPES    
    image_list = glob(os.path.join(SRX_DIR, 'leftImg8bit', SPLIT, '*_leftImg8bit.png'))
    clean_list = glob(os.path.join(CLEAN_DIR, 'leftImg8bit', 'train', '*_leftImg8bit.png'))

    for image_path in image_list:

        # get image name
        image_name = os.path.basename(image_path).replace('_leftImg8bit.png', '')
        # get label name
        label_name = image_name + '_gtFine_labelIds.png'
        # get label path
        label_path = os.path.join(SRX_DIR, 'gtFine', SPLIT, label_name)

        # get image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # get label
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # get random clean image
        clean_image_path = np.random.choice(clean_list)
        clean_image = cv2.imread(clean_image_path, cv2.IMREAD_UNCHANGED)

        # get clean label path
        clean_label_path = clean_image_path.replace('leftImg8bit.png', 'gtFine_labelIds.png')
        clean_label_path = clean_label_path.replace('leftImg8bit', 'gtFine')

        # get clean label
        clean_label = cv2.imread(clean_label_path, cv2.IMREAD_UNCHANGED)

        # overlay clean image and image 
        # get random position
        x = np.random.randint(0, image.shape[1] - clean_image.shape[1])
        y = np.random.randint(0, image.shape[0] - clean_image.shape[0])

        image[y:y+clean_image.shape[0], x:x+clean_image.shape[1], :] = clean_image

        # overlay clean label and label
        label[y:y+clean_label.shape[0], x:x+clean_label.shape[1]] = clean_label

        # save image
        new_image_path = os.path.join(RST_DIR, 'leftImg8bit', SPLIT, image_name + '_leftImg8bit.png')
        new_label_path = os.path.join(RST_DIR, 'gtFine', SPLIT, label_name)

        # create directory
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(new_label_path), exist_ok=True)

        cv2.imwrite(new_image_path, image)
        cv2.imwrite(new_label_path, label)

        # print progress
        sys.stdout.write(f'\r{image_name}')
        sys.stdout.flush()










if __name__ == '__main__':
    main()