import cv2
import numpy as np

from .utils import *

from collections import Counter
from skimage.measure import label, regionprops, regionprops_table
from skimage import data, measure, morphology

import copy

from modules.image_functions import ImageFunctions

def logits_np_to_prob(logits):
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    return probs

def softmax(logits):
    # 각 행에 대해 소프트맥스 계산
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return softmax_probs
    
def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def extract_values_above_threshold(scores, threshold):
    
    binary_image = scores > threshold

    return binary_image

def column_based_sampling(gimage, mask, num_samples=10, num_columns=10):
    rows, cols = mask.shape
    col_step = cols // num_columns

    sampled_coords = []

    for j in range(num_columns):
        col_start = j * col_step
        col_end = (j + 1) * col_step if (j + 1) * col_step < cols else cols

        region_mask = mask[:, col_start:col_end]
        region_gimage = gimage[:, col_start:col_end]

        region_coords = np.argwhere((region_mask) & (region_gimage != 0))
        if region_coords.size == 0:
            continue

        region_pixel_values = [region_gimage[tuple(coord)] for coord in region_coords]

        region_coords_values = list(zip(region_coords, region_pixel_values))

        best_coord = sorted(region_coords_values, key=lambda x: x[1])[0][0]

        best_coord_adjusted = (best_coord[0], best_coord[1] + col_start)
        sampled_coords.append(best_coord_adjusted)

        if len(sampled_coords) >= num_samples:
            return np.array(sampled_coords)

    return np.array(sampled_coords)

def width_based_sampling(gimage, mask, num_samples=10, num_rows=10):
    rows, cols = mask.shape
    row_step = rows // num_rows

    sampled_coords = []

    for i in range(num_rows):
        row_start = i * row_step
        row_end = (i + 1) * row_step if (i + 1) * row_step < rows else rows

        region_mask = mask[row_start:row_end, :]
        region_gimage = gimage[row_start:row_end, :]

        region_coords = np.argwhere((region_mask) & (region_gimage != 0))
        if region_coords.size == 0:
            continue

        region_pixel_values = [region_gimage[tuple(coord)] for coord in region_coords]

        region_coords_values = list(zip(region_coords, region_pixel_values))

        best_coord = sorted(region_coords_values, key=lambda x: x[1])[0][0]

        best_coord_adjusted = (best_coord[0] + row_start, best_coord[1])
        sampled_coords.append(best_coord_adjusted)

        if len(sampled_coords) >= num_samples:
            return np.array(sampled_coords)

    return np.array(sampled_coords)


def getTop6Centroid(label, onlycenter = False):
    top6 = []
    labels = measure.label(label)
    props = measure.regionprops(labels)
    for region in props:
        y, x = region.centroid
        min_x, min_y, max_x, max_y = region.bbox
        top6.append([region.area, (round(x), round(y)), (min_x, min_y, max_x, max_y)])
        
    top6.sort(reverse=True)
    top6 = top6[0:6]

    if onlycenter == True:
        centers = []
        for att in top6:
            centers.append(att[1])

    return top6 if onlycenter==False else centers

def getTop6Skeletonize(label, onlycenter = False):
    # 연결 영역 계산
    top6_info = []
    top6_idx = []
    top6 = []
    labels = measure.label(label)
    props = measure.regionprops(labels)

    for region in props:
        min_x, min_y, max_x, max_y = region.bbox
        top6_info.append([region.area, region.label, (min_x, min_y, max_x, max_y)])
        
    top6_info.sort(reverse=True)
    top6_info = top6_info[0:6]
    top6_idx = [info[1] for info in top6_info]

    for idx in top6_idx:
        msk = (labels == idx).astype(np.uint8)
        top6.append(msk)
    
    # 연결 영역이 발생하지 않은 경우 연산 중단
    if len(top6) == 0:
        return np.array([])
    
    # 연결 영역에 대한 Skeleton Point 계산
    point_list = []

    connect_mask = np.array(top6)
    for i in range(connect_mask.shape[0]):
        connect = connect_mask[i]
        skeleton = morphology.skeletonize(connect)

        y, x = np.nonzero(skeleton)

        if len(x) > 0:
            mid_x = x[np.abs((x-np.median(x))).argmin()].item()
            mid_x_idx = np.nonzero(x == mid_x)[0]
            mid_y_raw = y[mid_x_idx]
            mid_y = mid_y_raw[np.abs((mid_y_raw-np.median(y))).argmin()].item()
            point_list.append([top6_info[i][0], (mid_x, mid_y), top6_info[i][2]])
        
        if onlycenter == True:
            points = []
            for att in point_list:
                points.append(att[1])

    return point_list if onlycenter==False else points    



def calculate_thickness(region_mask: np.ndarray, 
                        pixel_resolution: float = 0.5, 
                        edge_threshold: int = 5
                        ): # -> tuple[list[float], list[tuple[int, int, int]]]:
    """
    영역의 두께를 계산하며, 경사와 가장자리 문제를 고려합니다.

    Args:
    region_mask: 영역 마스크
    pixel_resolution: 픽셀 해상도 (기본값: 0.5mm)
    edge_threshold: 가장자리로 간주할 픽셀 수 (기본값: 3)

    Returns:
    각 열 좌표별 두께 및 두께를 계산하는 데 사용된 행의 좌표
    """
    if region_mask.ndim != 2:
        raise ValueError("region_mask must be a 2D array")
    
    rows, cols = region_mask.shape
    thicknesses = []
    thickness_positions = []
    
    for x in range(cols):
        col_mask = region_mask[:, x]
        if np.any(col_mask):
            y_coords = np.where(col_mask)[0]
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            # 가장자리 체크
            if y_min < edge_threshold or y_max > rows - edge_threshold:
                continue
            
            # 경사 체크
            if y_max - y_min > 1:
                # 중간 지점 찾기
                mid_point = (y_min + y_max) // 2
                # 위쪽 경계 찾기
                upper_bound = mid_point
                while upper_bound > y_min and col_mask[upper_bound-1]:
                    upper_bound -= 1
                # 아래쪽 경계 찾기
                lower_bound = mid_point
                while lower_bound < y_max and col_mask[lower_bound+1]:
                    lower_bound += 1
                
                thickness = (lower_bound - upper_bound + 1) * pixel_resolution
                thicknesses.append(thickness)
                thickness_positions.append((x, upper_bound, lower_bound))
            else:
                thickness = (y_max - y_min + 1) * pixel_resolution
                thicknesses.append(thickness)
                thickness_positions.append((x, y_min, y_max))
    
    return thicknesses, thickness_positions


def calculate_mode(values, 
                   precision: int = 1) -> float:
    """
    주어진 값들의 최빈값을 계산합니다.

    Args:
    values: 값들의 리스트
    precision: 결과의 소수점 자리수 (기본값: 1)

    """
    rounded_values = np.round(values, precision)
    value_counts = Counter(rounded_values)
    mode = max(value_counts, key=value_counts.get)
    return mode
