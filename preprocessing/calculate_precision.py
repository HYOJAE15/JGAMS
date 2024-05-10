import numpy as np 
import cv2 

import os 
from glob import glob


RESULT_DIR = '//172.16.113.151/UOS-SSaS Dropbox/05. Data/01. Under_Process/025. Seoul Facility Corporation/2023.03.30 Concrete Crack Detection Cascade Mask R-CNN only Positive Samples/'
RESULT_SUFFIX = '_mask_output.png'

LABEL_DIR = '//172.16.113.151/UOS-SSaS Dropbox/05. Data/02. Training&Test/025. Seoul_Facilities_Corporation/Tancheon-P46-48-Ortho/v0.1.4/gtFine/test/'
LABEL_SUFFIX = '_gtFine_labelIds.png'


def main():

    # RESULT list 
    result_list = glob(os.path.join(RESULT_DIR, '*' + RESULT_SUFFIX))

    for result_path in result_list:
        result_name = os.path.basename(result_path).replace(RESULT_SUFFIX, '')
        label_path = os.path.join(LABEL_DIR, result_name + LABEL_SUFFIX)

        result = cv2.imread(result_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        true_positive = np.sum(result & label)
        false_positive = np.sum(result & ~label)
        false_negative = np.sum(~result & label)

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        print('result_name: ', result_name)
        print('precision: ', precision)
        print('recall: ', recall)
        print('')

        #

    pass 



if __name__ == "__main__":
    main()
    


