"""
Based on the work of Waleed Abdulla (Matterport)
Modified by github.com/GustavZ
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import random
import colorsys


from pathlib import Path
from math import trunc
from skimage.draw import polygon2mask, rectangle_perimeter
from matplotlib import patches,  lines
from matplotlib.patches import Polygon

def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    mask_px = np.where(mask)
    for c in range(3):
        image[mask_px[0], mask_px[1], c] = (1 - alpha)*image[mask_px[0], mask_px[1], c] + alpha * color[c] * 255
    return image

class CocoDataset():
    def __init__(self, annotation_path, image_dir):
        self.annotation_path = annotation_path
        self.image_dir = image_dir

        json_file = open(self.annotation_path)
        self.coco = json.load(json_file)
        json_file.close()

        self._process_info()
        self._process_licenses()
        self._process_categories()
        self._process_images()
        self._process_segmentations()

    def _process_info(self):
        self.info = self.coco['info']

    def _process_licenses(self):
        self.licenses = self.coco['licenses']

    def _process_categories(self):
        self.categories = dict()
        self.super_categories = dict()

        for category in self.coco['categories']:
            cat_id = category['id']
            super_category = category['supercategory']

            # Add category to categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
            else:
                print(f'ERROR: Skipping duplicate category id: {category}')

            # Add category id to the super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {cat_id}
            else:
                self.super_categories[super_category] |= {cat_id}  # e.g. {1, 2, 3} |= {4} => {1, 2, 3, 4}

    def _process_images(self):
        self.images = dict()
        for image in self.coco['images']:
            image_id = image['id']
            if image_id not in self.images:
                self.images[image_id] = image
            else:
                print(f'ERROR: Skipping duplicate image id: {image}')

    def _process_segmentations(self):
        self.segmentations = dict()
        for segmentation in self.coco['annotations']:
            image_id = segmentation['image_id']
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)

    def display_info(self):
        print('Dataset Info')
        print('==================')
        for key, item in self.info.items():
            print(f'  {key}: {item}')

    def display_licenses(self):
        print('Licenses')
        print('==================')
        for license in self.licenses:
            for key, item in license.items():
                print(f'  {key}: {item}')

    def display_categories(self):
        print('Categories')
        print('==================')
        for sc_name, set_of_cat_ids in self.super_categories.items():
            print(f'  super_category: {sc_name}')
            for cat_id in set_of_cat_ids:
                print(f'    id {cat_id}: {self.categories[cat_id]["name"]}'
                      )

            print('')

    def display_image(self, image_id, show_bbox=True, show_polys=True, show_crowds=True):
        print('Image')
        print('==================')

        # Print image info
        image = self.images[image_id]
        for key, val in image.items():
            print(f'  {key}: {val}')

        # Open the image
        image_path = Path(self.image_dir) / image['file_name']
        image = io.imread(image_path)

        image_width, image_height = image.shape[0], image.shape[1]

        # Create bounding boxes and polygons
        bboxes = dict()
        polygons = dict()
        rle_regions = dict()
        annot_categories = dict()

        for i, seg in enumerate(self.segmentations[image_id]):

            annot_categories[seg['id']] = seg["category_id"]

            bboxes[seg['id']] = np.array(seg['bbox']).astype(int)

            if seg['iscrowd'] == 0:
                polygons[seg['id']] = []
                for seg_points in seg['segmentation']:
                    seg_points = np.array(seg_points).astype(int)
                    polygons[seg['id']].append(seg_points)
            else:
                # Decode the RLE
                px = 0
                rle_list = []
                for j, counts in enumerate(seg['segmentation']['counts']):
                    if counts < 0:
                        print(f'ERROR: One of the counts was negative, treating as 0: {counts}')
                        counts = 0

                    if j % 2 == 0:
                        # Empty pixels
                        px += counts
                    else:
                        # Create one or more vertical rectangles
                        x1 = trunc(px / image_height)
                        y1 = px % image_height
                        px += counts
                        x2 = trunc(px / image_height)
                        y2 = px % image_height

                        if x2 == x1:  # One vertical column
                            line = [x1, y1, 1, (y2 - y1)]
                            rle_list.append(line)
                        else:  # Two or more columns
                            # Insert left-most line first
                            left_line = [x1, y1, 1, (image_height - y1)]
                            rle_list.append(left_line)

                            # Insert middle lines (if needed)
                            lines_spanned = x2 - x1 + 1
                            if lines_spanned > 2:  # Two columns won't have a middle
                                middle_lines = [(x1 + 1), 0, lines_spanned - 2, image_height]
                                rle_list.append(middle_lines)

                            # Insert right-most line
                            right_line = [x2, 0, 1, y2]
                            rle_list.append(right_line)

                if len(rle_list) > 0:
                    rle_regions[seg['id']] = rle_list

        _, ax = plt.subplots(1, figsize=(15, 20))

        height, width = image.shape[:2]
        ax.set_ylim(height + 10, -10)
        ax.set_xlim(-10, width + 10)
        ax.axis('off')

        brightness = 0.7
        print(f"self.categories: {self.categories}")
        n_class = len(self.categories)
        hsv = [(i / n_class, 1, brightness) for i in range(n_class)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

        masked_image = image.copy()
        
        print(f"bboxes.items(): {bboxes.items()}")
        print(f"colors: {colors}")
        print(f"annot_categories: {annot_categories}")
        # edge_color_check = colors[annot_categories[0]]
        # print(edge_color_check)

        # Draw shapes on image
        if show_bbox:
            for seg_id, bbox in bboxes.items():
                
                print(f"seg_id: {seg_id}")
                print(f"bbox: {bbox}")
                edge_color = colors[annot_categories[seg_id]]
                print(f"edge_color: {edge_color}")

                bbox = np.asarray(bbox)
                x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

                p = patches.Rectangle((x, y), w, h, linewidth=2,
                                      alpha=0.7, linestyle="dashed",
                                      edgecolor=edge_color, facecolor='none')
                ax.add_patch(p)

        if show_polys:
            for seg_id, points_lists in polygons.items():
                edge_color = colors[annot_categories[seg_id]]

                for points_list in points_lists : 
                    points_list = np.asarray(points_list)
                    points_list = np.reshape(points_list, (int(points_list.shape[0] / 2), 2))
                    
                    p = Polygon(points_list, facecolor="none", edgecolor=edge_color)
                    mask = polygon2mask((image.shape[1], image.shape[0]), points_list).T
                    masked_image = apply_mask(masked_image, mask, edge_color)
                    ax.add_patch(p)

        ax.imshow(masked_image.astype(np.uint8))
        plt.show()