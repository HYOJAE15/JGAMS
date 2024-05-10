import os
import argparse

from glob import glob
from tqdm import tqdm

import slidingwindow as sw

import cv2
import numpy as np

from skimage.measure import label, regionprops_table
from skimage.morphology import medial_axis

import json

parser = argparse.ArgumentParser()

parser.add_argument("source_path", help="file path to store segmented images", type=str)
parser.add_argument("--quantify", default=True, type=bool)
parser.add_argument("--color", default=True, type=bool)

args = parser.parse_args()

def imread(path):
    stream = open(path, "rb")
    bytes = bytearray(stream.read())
    nparray = np.asarray(bytes, dtype=np.uint8)
    bgrImage = cv2.imdecode(nparray, cv2.IMREAD_UNCHANGED)
    return bgrImage

def imwrite(path, image):
    _, ext = os.path.splitext(path)
    cv2.imencode(ext, image)[1].tofile(path)

def blendImageWithColorMap(image, label, palette, alpha):
    color_map = np.zeros_like(image)
        
    for idx, color in enumerate(palette) : 
        
        if idx == 0 :
            color_map[label == idx, :] = image[label == idx, :] * 1
        else :
            color_map[label == idx, :] = image[label == idx, :] * alpha + color * (1-alpha)

    return color_map

def main():
    source_path = args.source_path
    quantify = args.quantify
    color = args.color
    
    segmented_images = glob(os.path.join(source_path, 'leftImg8bit', '*', f'*.png'))
    segmented_gt = glob(os.path.join(source_path, 'gtFine', '*', f'*.png'))

    image_ex = segmented_images[0]
    image_ex_name = os.path.basename(image_ex)
    img_shape_0, img_shape_1, x_start, y_start, crop_size = map(int, image_ex_name.split('_')[-6:-1])  # Extract x_start, y_start, and crop_size from the file name
    
    reassembled_image = np.zeros((img_shape_0, img_shape_1, 3), dtype=np.uint8)  # Assuming 3 channels, adjust accordingly
    reassembled_gt = np.zeros((img_shape_0, img_shape_1), dtype=np.uint8)

    reassembled_dir_path = os.path.join(source_path, 'Assembled')
    os.makedirs(reassembled_dir_path, exist_ok=True)
    assemble_file_name = image_ex_name.split('_')[:-8]
    if len(assemble_file_name)>=1:
        full_name = ''
        for idx, name in enumerate(assemble_file_name):
            if idx == 0:
                full_name = full_name + str(name)
            else :
                full_name = full_name + '_' + str(name)

    reassembled_image_path = os.path.join(reassembled_dir_path, f'{full_name}_assembled.png')
    reassembled_gt_path = os.path.join(reassembled_dir_path, f'{full_name}_gt_assembled.png')

    for segmented_image_path, segmented_gt_path in zip(tqdm(segmented_images, desc='Reassembling'), segmented_gt):
        file_name = os.path.basename(segmented_image_path)
        img_shape_0, img_shape_1, x_start, y_start, crop_size = map(int, file_name.split('_')[-6:-1])  # Extract x_start, y_start, and crop_size from the file name   
        
        subset_img = imread(segmented_image_path)
        reassembled_image[x_start:(x_start + crop_size), y_start:(y_start + crop_size), :] = subset_img
        imwrite(reassembled_image_path, reassembled_image)
        
        subset_gt = imread(segmented_gt_path)
        reassembled_gt[x_start:(x_start + crop_size), y_start:(y_start + crop_size)] = subset_gt
        imwrite(reassembled_gt_path, reassembled_gt)

    if quantify:
        print(quantify)

        det_result_dict = {}
        img_basename = os.path.basename(reassembled_image_path)
        det_result_dict[img_basename] = {}
        det_result_dict[img_basename]["anly_output"] = []



        for damage_idx in tqdm(np.unique(reassembled_gt)[1:], desc="Quantifying"):
            
            if damage_idx == 1:
                damage_type = 'nusu'
            elif damage_idx == 2:
                damage_type = 'baektae'
            elif damage_idx == 3:
                damage_type = 'bakri'
            elif damage_idx == 4 :
                damage_type = 'bakrak'
            elif damage_idx == 5 :
                damage_type = 'kyunyeol'
            elif damage_idx == 6 :
                damage_type = 'cheolgeunnochul'
            elif damage_idx == 7 :
                damage_type = 'chungbunli'
            elif damage_idx == 8 :
                damage_type = 'kyunyeolbosu'
        
            labels = label(reassembled_gt == damage_idx)
            damage_region_prop = regionprops_table(labels, properties=('label', "bbox", 'area'))

            for label_num in range(np.max(labels)) :
        
                if damage_region_prop['bbox-0'][label_num] > 50:

                    a_label = labels == label_num + 1

                    contours, _ = cv2.findContours(a_label.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                    poly_step = 50 if damage_type == 'kyunyeol' else 100

                    coords = []
                    for countour_num in range(0, len(contours[0]), poly_step):
                        coords.append([str(contours[0][countour_num][0][1]), str(contours[0][countour_num][0][0])])

                    dmg_info = {}
                    if damage_idx == 5:

                        min_row = int(damage_region_prop['bbox-0'][label_num])
                        max_row = int(damage_region_prop['bbox-2'][label_num])
                        min_col = int(damage_region_prop['bbox-1'][label_num])
                        max_col = int(damage_region_prop['bbox-3'][label_num])

                        a_label_skel = a_label[min_row: max_row, min_col: max_col].copy()

                        skel, distance = medial_axis(a_label_skel, return_distance=True)
                        dist_label = distance * skel

                        width = str(dist_label[np.nonzero(dist_label)].mean()) # * lenPerPixel)

                        dmg_info['damage_type'] = damage_type
                        dmg_info['id'] = str(label_num)
                        dmg_info['length'] = str(np.sum(skel)) # * lenPerPixel)
                        dmg_info['width'] = width
                        dmg_info['height'] = ""
                        dmg_info['area'] = ""
                        dmg_info['coords'] = coords

                    else:

                        min_row = int(damage_region_prop['bbox-0'][label_num])
                        max_row = int(damage_region_prop['bbox-2'][label_num])
                        min_col = int(damage_region_prop['bbox-1'][label_num])
                        max_col = int(damage_region_prop['bbox-3'][label_num])

                        dmg_info['damage_type'] = damage_type
                        dmg_info['id'] = str(label_num)
                        dmg_info['length'] = ""
                        dmg_info['width'] = str((max_col - min_col)) # * lenPerPixel)
                        dmg_info['height'] = str((max_row - min_row)) # * lenPerPixel)
                        dmg_info['area'] = str((max_col - min_col) * (max_row - min_row)) # * lenPerPixel * lenPerPixel)
                        dmg_info['coords'] = coords

                    det_result_dict[img_basename]["anly_output"].append(dmg_info)
        
        json_path = os.path.dirname(reassembled_gt_path)
        json_save_path = reassembled_gt_path.replace(".png", ".json")
        with open(json_save_path, 'w', encoding='utf8') as f:
            json.dump(det_result_dict, f, ensure_ascii=False)
    
    if color:
        print(color)
        palette = np.array([
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [255, 0, 255], 
            [127, 0, 255],
            [0, 0, 255],
            [0, 255, 255], 
            [0, 170, 0], 
            [0, 170, 255]
            ])
        
        color_path = reassembled_image_path.replace('.png', '_color.png')
        colormap = blendImageWithColorMap(image=reassembled_image, label=reassembled_gt, palette=palette, alpha=float(0.5))
        
        imwrite(color_path, colormap)

if __name__ == '__main__':
    main()