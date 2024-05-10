import numpy as np
import os 

# unlock cv2 imread limitation
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2,40).__str__()

import cv2

from glob import glob


LABEL_DIR = r'\\172.16.113.151\UOS-SSaS Dropbox\05. Data\02. Training&Test\025. Seoul_Facilities_Corporation\Tancheon-P46-48-Ortho\v0.1.7 - Copy (2)\gtFine\test'
LABEL_SUFFIX = '_gtFine_labelIds.png'

MASK_DIR = r'\\172.16.113.151\UOS-SSaS Dropbox\05. Data\01. Under_Process\025. Seoul Facility Corporation\2023.04.03 Concrete Crack Detection Cascade Mask R-CNN only Positive Samples'
MASK_SUFFIX = '_add_positive.png'

CM_DIR = r'\\172.16.113.151\UOS-SSaS Dropbox\05. Data\01. Under_Process\025. Seoul Facility Corporation\2023.04.03 Concrete Crack Detection Cascade Mask R-CNN only Positive Samples'
CM_SUFFIX = '_cm_map.png'

NEW_LABEL_DIR = r'\\172.16.113.151\UOS-SSaS Dropbox\05. Data\02. Training&Test\025. Seoul_Facilities_Corporation\Tancheon-P46-48-Ortho\v0.1.8\gtFine\test'



def main():

    # read labels 
    label_list = glob(os.path.join(LABEL_DIR, '*' + LABEL_SUFFIX))

    for label_path in label_list:
        # get label name
        label_name = os.path.basename(label_path).replace(LABEL_SUFFIX, '')

        # get mask path
        mask_path = os.path.join(MASK_DIR, label_name + MASK_SUFFIX)
        # get confusion matrix path
        cm_path = os.path.join(CM_DIR, label_name + CM_SUFFIX)

        # read label
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        # read mask
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        # read confusion matrix
        cm = cv2.imread(cm_path, cv2.IMREAD_UNCHANGED)

        # get indices of green area in cm (RGB) 
        green_area = np.all(cm == [0, 255, 0], axis=-1)

        mask_0 = mask[:, :, 0]
        mask_1 = mask[:, :, 1]
        mask_2 = mask[:, :, 2]
        mask_3 = mask[:, :, 3]

        # get indices of white area in mask (RGB)       
        _mask = (mask_0 > 200) 
        _mask = _mask.astype(np.uint8)
        _mask[(mask_1 > 200)] += 1
        _mask[(mask_2 > 200)] += 1
        _mask[(mask_3 > 200)] += 1

        mask = _mask == 4

        del mask_0, mask_1, mask_2, mask_3, _mask

        # intersection of mask and cm 
        intersection = np.logical_and(mask, green_area)

        # print number of intersection pixels
        print('label_name: ', label_name)
        print('number of intersection pixels: ', np.sum(intersection))
        
        # add intersection to label
        label[intersection] = 1

        del mask, green_area, intersection

        # save label
        new_label_path = os.path.join(NEW_LABEL_DIR, label_name + LABEL_SUFFIX)

        cv2.imwrite(new_label_path, label)

if __name__ == '__main__':
    main()