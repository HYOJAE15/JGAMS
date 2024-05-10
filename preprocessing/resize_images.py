import cv2 

from glob import glob

import numpy as np

import os

import sys



SRX_DIR = r'\\172.16.113.151\UOS-SSaS Dropbox\05. Data\02. Training&Test\012. General_Crack\02. Negative Samples\RawImages\val'
SRX_EXT = '.jpg'

RST_DIR = r'\\172.16.113.151\UOS-SSaS Dropbox\05. Data\02. Training&Test\012. General_Crack\02. Negative Samples\ResizeImages\val'
RST_EXT = '.jpg'



def main():
    # read image list 
    image_list = glob(os.path.join(SRX_DIR, f'*{SRX_EXT}'))

    for image_path in image_list:
        # get image name
        image_name = os.path.basename(image_path).replace(SRX_EXT, '')

        # get image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

        # resize image
        # resize if shorter side is less than 512
        if image.shape[0] < 512 or image.shape[1] < 512:
            # make shorter size 1024
            if image.shape[0] < image.shape[1]:
                image = cv2.resize(image, (int(image.shape[1] * 1024 / image.shape[0]), 1024))
            else:
                image = cv2.resize(image, (1024, int(image.shape[0] * 1024 / image.shape[1])))

        # save image
        new_image_path = os.path.join(RST_DIR, image_name + RST_EXT)

        cv2.imwrite(new_image_path, image)

        # print progress
        sys.stdout.write(f'\r{image_name}')
        sys.stdout.flush()


if __name__ == '__main__':
    main()


