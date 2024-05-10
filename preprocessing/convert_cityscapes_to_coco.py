import cv2
import numpy as np 
import os
import json 

from glob import glob 
from tqdm import tqdm

from copy import deepcopy

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries

from skimage.measure import label, regionprops

from pycococreatortools import create_image_info, create_annotation_info

import datetime
import shutil

# global arguments 

INFO = {
    "description": "Concrete Damage Dataset",
    "version": "1.0",
    "year": 2018,
    "contributor": "UOS-SSaS",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

CATEGORIES = []

SOURCE_DIR = 'C:/Users/tls15/Downloads/ex'

TARGET_DIR = 'C:/Users/tls15/Downloads/ex/bbox'

SUBSET = 'train' # choose one of 'train', 'val', 'test'

CLASSES = ['crack', 'spall'] #, ['crack', 'effl', 'rebar', 'spall']

CONVERT_STYLE = {
    'crack' : 'overlap',
    'effl' : 'normal',
    'rebar' : 'overlap',
    'spall' : 'normal'
}

CATEGORIES = []

WINDOW_SIZE = 256

OVERLAP = 0.5

DIALATE = False

for i, name in enumerate(CLASSES):
    cat = {'id': i, 'name': name, 'supercategory': 'concrete_damage'}
    CATEGORIES.append(cat)

def create_normal_annotation(coco_output, gt, img_id, width, height, class_idx, segmentation_id, num_classes, class_name):
    """
    Create normal annotation (without overalapping)
    Args:
        coco_output: coco output
        gt: ground truth
        img_id: image id
        width: width of the image
        height: height of the image
        class_idx: class index
        segmentation_id: segmentation id
        num_classes: 

    Returns:
        coco_output: coco output
        segmentation_id: segmentation id
    """


    gt_label = label(gt == class_idx)

    cat_id_idx = CLASSES.index(class_name)
    
    
    category_info = {'id': cat_id_idx, 'is_crowd': 0}

    
    for label_idx in range(1, np.max(gt_label)+1):
        binary_mask = gt_label == label_idx

        annotation_info = create_annotation_info(
            segmentation_id, img_id, category_info, binary_mask, (width, height), tolerance=2
        )

        if annotation_info is not None:
            coco_output["annotations"].append(annotation_info)

        segmentation_id += 1

    return coco_output, segmentation_id

def create_overlap_annotation(coco_output, gt, img_id, width, height, class_idx, segmentation_id, num_classes, class_name, window_size = 256, overlap = 0.5):
    """
    Create overlap annotation
    Args:
        coco_output: coco output
        gt: ground truth
        img_id: image id
        width: width of the image
        height: height of the image
        class_idx: class index
        segmentation_id: segmentation id
        window_size: window size
        overlap: overlap ratio

    Returns:
        coco_output: coco output
        segmentation_id: segmentation id
    """
    gt_label = label(gt == class_idx)
    # get regionprops
    props = regionprops(gt_label)
    
    cat_id_idx = CLASSES.index(class_name)
    
    category_info = {'id': cat_id_idx, 'is_crowd': 0}

    for label_idx in range(1, np.max(gt_label)+1):
        binary_mask = gt_label == label_idx

        # extract bounding box
        minr, minc, maxr, maxc = props[label_idx-1].bbox

        # check bounding box size
        # if bounding box is smaller than 10, skip
        if (maxr - minr) < 10 and (maxc - minc) < 10:
            pass

        # else if bounding box is smaller than window size, create normal annotation
        elif (maxr - minr) < window_size and (maxc - minc) < window_size:
            annotation_info = create_annotation_info(
            segmentation_id, img_id, category_info, binary_mask, (width, height), tolerance=2
            )

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)

            segmentation_id += 1

        # else if bounding box is bigger than window size, create overlap annotation
        else:
            # coordinate of grid 
            num_grid_x = int((maxc - minc) / (window_size * overlap)) + 1 
            num_grid_y = int((maxr - minr) / (window_size * overlap)) + 1

            if (maxc - minc) > (maxr - minr):
                num_grid_y = 1
            else:
                num_grid_x = 1

            # create grid
            grid_x = np.linspace(minc, maxc, num_grid_x, dtype=np.int, endpoint=False)
            grid_y = np.linspace(minr, maxr, num_grid_y, dtype=np.int, endpoint=False)

            # deepcopy binary_mask
            _binary_mask = deepcopy(binary_mask)

            # remove pixels on the grid line 
            for i in range(0, len(grid_x), 2):
                _binary_mask[:, grid_x[i]] = 0

            for i in range(0, len(grid_y), 2):
                _binary_mask[grid_y[i], :] = 0

            if len(grid_x) > 2: 
                if len(grid_x) % 2 == 0:
                    pass 
                else:
                    _binary_mask[:, grid_x[-1]:] = 0

            if len(grid_y) > 2:
                if len(grid_y) % 2 == 0:
                    pass 
                else:
                    _binary_mask[grid_y[-1]:, :] = 0

            _binary_label = label(_binary_mask)

            for _label_idx in range(1, np.max(_binary_label)+1):
                _binary_mask = _binary_label == _label_idx

                # extract bounding box
                _minr, _minc, _maxr, _maxc = regionprops(_binary_label)[_label_idx-1].bbox
                # check bounding box size

                # if bounding box is smaller than window size, create normal annotation
                if (_maxr - _minr) > window_size or (_maxc - _minc) > window_size:
                    coco_output, segmentation_id = create_overlap_annotation(coco_output, _binary_mask, img_id, width, height, class_idx, segmentation_id, num_classes, class_name, window_size, overlap)

                else: 
                    annotation_info = create_annotation_info(
                        segmentation_id, img_id, category_info, _binary_mask, (width, height), tolerance=2
                        )

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id += 1

            # deepcopy binary_mask
            _binary_mask = deepcopy(binary_mask)

            # remove pixels on the grid line 
            for i in range(1, len(grid_x), 2):
                _binary_mask[:, grid_x[i]] = 0

            for i in range(1, len(grid_y), 2):
                _binary_mask[grid_y[i], :] = 0

            if len(grid_x) > 2: 
                _binary_mask[:, :grid_x[1]] = 0
                if len(grid_x) % 2 == 0:
                    _binary_mask[:, grid_x[-1]:] = 0
                    

            if len(grid_y) > 2:
                _binary_mask[:grid_y[1], :] = 0
                if len(grid_y) % 2 == 0:
                    _binary_mask[grid_y[-1]:, :] = 0

            _binary_label = label(_binary_mask)

            for _label_idx in range(1, np.max(_binary_label)+1):
                _binary_mask = _binary_label == _label_idx

                _minr, _minc, _maxr, _maxc = regionprops(_binary_label)[_label_idx-1].bbox

                if (_maxr - _minr) > window_size or (_maxc - _minc) > window_size:
                    coco_output, segmentation_id = create_overlap_annotation(coco_output, _binary_mask, img_id, width, height, class_idx, segmentation_id, num_classes, class_name, window_size, overlap)

                else: 
                    annotation_info = create_annotation_info(
                        segmentation_id, img_id, category_info, _binary_mask, (width, height), tolerance=2
                        )

                    if annotation_info is not None:
                        coco_output["annotations"].append(annotation_info)

                    segmentation_id += 1

    return coco_output, segmentation_id


def main():
    
    coco_output = {
    "info": INFO,
    "licenses": LICENSES,
    "categories": CATEGORIES,
    "images": [],
    "annotations": []
    }

    segmentation_id = 1


    # source dataset style is Cityscapes
    img_dir = os.path.join(SOURCE_DIR, 'leftImg8bit', SUBSET)
    img_list = glob(os.path.join(img_dir, '*.png'))

    # target dataset style is COCO
    target_dir = TARGET_DIR
    target_img_dir = os.path.join(target_dir, f'{SUBSET}2018')
    target_json_path = os.path.join(target_dir, 'annotations', f'instances_{SUBSET}2018.json')

    os.makedirs(target_img_dir, exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'annotations'), exist_ok=True)

    new_img_id = 1

    # add tqdm and description
    for img_id, img_path in enumerate(tqdm(img_list, desc="Creating COCO dataset")):

        # read images 
        img = cv2.imread(img_path)
        
        # copy image to target dir
        target_img_path = os.path.join(target_img_dir, os.path.basename(img_path))

        # read gt
        gt_path = img_path.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
        gt_path = gt_path.replace("leftImg8bit", "gtFine")
        
        gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        # dilate gt
        if DIALATE:
            gt = cv2.dilate(gt, np.ones((3, 3), np.uint8), iterations=1)

        for class_idx, class_name in enumerate(CLASSES, start=1):

            
            if class_name == 'crack' :
                class_idx = 1
            elif class_name == 'effl' :
                class_idx = 2
            elif class_name == 'rebar' :
                class_idx = 3
            elif class_name == 'spall' :
                class_idx = 4

            convert_style = CONVERT_STYLE[class_name]

            num_classes = len(CLASSES)

            if convert_style == 'normal':
                coco_output, segmentation_id = create_normal_annotation(coco_output, 
                                                                        gt, 
                                                                        new_img_id, 
                                                                        img.shape[1], 
                                                                        img.shape[0], 
                                                                        class_idx, 
                                                                        segmentation_id, 
                                                                        num_classes, 
                                                                        class_name = class_name)                

            elif convert_style == 'overlap':
                coco_output, segmentation_id = create_overlap_annotation(coco_output, 
                                                                         gt, 
                                                                         new_img_id, 
                                                                         img.shape[1], 
                                                                         img.shape[0], 
                                                                         class_idx, 
                                                                         segmentation_id, 
                                                                         num_classes, 
                                                                         class_name = class_name, 
                                                                         window_size=WINDOW_SIZE, 
                                                                         overlap=OVERLAP)

            if np.sum(gt == class_idx) > 0:
                # print(f"img: {img_path}, target img:{target_img_path}")
                shutil.copy(img_path, target_img_path)
                # create image info
                
                image_info = create_image_info(
                    image_id=new_img_id, 
                    file_name=os.path.basename(img_path), 
                    image_size=(img.shape[1], img.shape[0])
                )
                coco_output["images"].append(image_info)
                new_img_id += 1
            


    with open(target_json_path, 'w') as output_json_file:
        json.dump(coco_output, output_json_file)
        



if __name__ == "__main__":
    main()