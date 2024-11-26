import numpy as np

from .utils import *

from collections import Counter
from skimage import data, measure, morphology

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

def merge_close_ranges(ranges, threshold=5):
    if not ranges:
        return []
    
    merged = []
    current_start, current_end = ranges[0]
    
    for next_start, next_end in ranges[1:]:
        # 현재 구간의 끝점과 다음 구간의 시작점의 차이가 threshold보다 작으면
        if next_start - current_end <= threshold:
            # 구간을 확장
            current_end = next_end
        else:
            # 새로운 구간 시작
            merged.append((current_start, current_end))
            current_start, current_end = next_start, next_end
    
    # 마지막 구간 추가
    merged.append((current_start, current_end))
    
    return merged

def find_transition_ranges(mask):
    transition_ranges = []
    for i in range(mask.shape[0]):
        row = mask[i, :]
        ranges = []
        in_range = False
        start = -1
        for idx in range(len(row)):
            if not in_range and row[idx]:
                in_range = True
                start = idx
            elif in_range and not row[idx]:
                in_range = False
                ranges.append((start, idx))
        if in_range:
            ranges.append((start, len(row)))
            
        # 가까운 구간들을 병합
        merged_ranges = merge_close_ranges(ranges)
        transition_ranges.append(merged_ranges)
    return transition_ranges

def get_distances_by_segment(transition_ranges, threshold_ratio=0.9):
    # 먼저 각 구간별 거리 계산
    max_segments = max(len(ranges) for ranges in transition_ranges if ranges)
    segment_data = [[] for _ in range(max_segments)]  # 각 구간별로 거리와 좌표를 저장할 리스트
    
    # 각 행을 순회하면서 구간별 거리 및 좌표 수집
    for row_index, ranges in enumerate(transition_ranges):
        if ranges:
            for i, (start, end) in enumerate(ranges):
                distance = end - start
                segment_data[i].append({
                    'distance': distance,
                    'start': start,
                    'end': end,
                    'row': row_index  # 이 행(row_index)의 위치를 기록 (x 좌표 역할을 할 수 있음)
                })
    
    # 각 행의 구간 개수 확인 및 분포 확인
    segment_counts = [len(ranges) for ranges in transition_ranges if ranges]
    count_distribution = Counter(segment_counts)
    # print("구간 개수 분포:", dict(count_distribution))
    
    # 가장 많은 거리 개수 찾기
    max_distance_count = max(len(segment) for segment in segment_data)
    threshold_count = max_distance_count * threshold_ratio
    # print(f"최대 거리 개수: {max_distance_count}")
    # print(f"임계값(90%): {threshold_count}")
    
    # 임계값을 넘는 구간만 선택
    filtered_segment_data = []
    for segment in segment_data:
        if len(segment) >= threshold_count:
            filtered_segment_data.append(segment)
    
    # print("\n필터링 후 구간 통계:")
    # for i, segment in enumerate(filtered_segment_data):
    #     print(f"구간 {i+1}:")
    #     print(f"  거리 개수: {len(segment)}")
    #     if len(segment) > 0:
    #         distances = [data['distance'] for data in segment]
    #         print(f"  평균 거리: {np.mean(distances):.2f}")
    #         print(f"  거리 범위: {min(distances)} ~ {max(distances)}")
    
    return filtered_segment_data

def find_two_modes(distances):
    # 거리들의 빈도 계산
    counter = Counter(distances)
    
    # 첫 번째 최빈값 찾기
    first_mode = max(counter.items(), key=lambda x: x[1])[0]
    
    # 첫 번째 최빈값을 제외한 새로운 Counter 생성
    counter.pop(first_mode)
    
    # counter가 비어있지 않은 경우에만 두 번째 최빈값 찾기
    if counter:
        second_mode = max(counter.items(), key=lambda x: x[1])[0]
        return min(first_mode, second_mode)
    else:
        # 모든 값이 같은 빈도를 가진 경우 첫 번째 최빈값 반환
        print("  모든 값이 동일한 빈도를 가짐")
        return first_mode
    
def distance(mask):
    """
    DeeplabV3+에서 반환된 mask의 y방향 거리를 계산합니다

    Args:
    mask: 0과 1로 이루어진 binary mask

    Returns:
    max_distance: y방향 거리의 최댓값
    mean_distance: y방향 거리의 평균값
    min_distance: y방향 거리의 최솟값
    """

    # Definition Part
    distances = []

    # Calculate Distances
    for x in range(mask.shape[1]):
        y_coords = np.where(mask[:, x] == 1)[0]

        if len(y_coords) > 0:
            y_max = np.max(y_coords)
            y_min = np.min(y_coords)
            distance = y_max - y_min
            distances.append(distance)

    # Calculate Max, Mean, Min
    max_distance = np.max(distances)
    mean_distance = np.mean(distances)
    min_distance = np.min(distances)

    return max_distance, mean_distance, min_distance

def SelectWindow(windows, joint_mask, gap_mask):
    """
    SAM2를 이용하여 Inference할 영역들을 선택하는 계산을 수행합니다.

    Args:
    windows: Slidingwindow 라이브러리를 이용하여 연산된 slidingwindow 객체
    joint_mask: 딥랩에서 탐지하고, Grounding DINO를 이용하여 crop한 유간 외 영역에 대한 binary mask
    gap_mask: 딥랩에서 탐지하고, Grounding DINO를 이용하여 crop한 유간 영역에 대한 binary mask

    Returns:
    selected_windows: inference를 수행할 window들만 선택하여 (x1, y1, x2, y2) 형식으로 표현한 list
    """

    # Window Grouping
    grouped_windows = {}

    for wn in windows:
        left_x = wn.x
        if left_x not in grouped_windows:
            grouped_windows[left_x] = []
        grouped_windows[left_x].append(wn)

    grouped_windows = list(grouped_windows.values())

    # Select Window
    selected_windows = []

    for wns in grouped_windows:
        raw_windows = []
        joint_sums = []
        gap_sums = []

        wn_start = wns[0]
        wn_end = wns[-1]

        w, h = wn_start.w, wn_start.h
        x1 = wn_start.x 
        x2 = x1 + w

        y_start = wn_start.y
        y_end = wn_end.y + h

        if y_start == y_end - h:
            y1, y2 = y_start, y_start+h
            joint_sum = np.sum(joint_mask[y1:y2, x1:x2])
            gap_sum = np.sum(gap_mask[y1:y2, x1:x2])
            if joint_sum!=0 and gap_sum!=0:
                selected_windows.append((x1, y1, int(x2), int(y2)))
            else:
                continue

        for y in range(y_start, y_end-h, 1):
            y1, y2 = y, y+h
            joint_sum = np.sum(joint_mask[y1:y2, x1:x2])
            gap_sum = np.sum(gap_mask[y1:y2, x1:x2])

            if joint_sum!=0 and gap_sum!=0:
                joint_sums.append(joint_sum)
                gap_sums.append(gap_sum)
                raw_windows.append((x1, y1, int(x2), int(y2)))
        
        if len(raw_windows) > 0 :
            gap_sum_max = max(gap_sums)
            max_idxs = [idx for idx, val in enumerate(gap_sums) if val==gap_sum_max]
            mid_idx = int(len(max_idxs)/2)
            selected_window = raw_windows[max_idxs[mid_idx]]
            selected_windows.append(selected_window)
    
    if len(selected_windows) <= 0:
        print("There are no gap pixels detected in the Prompt Model.")
    return selected_windows

def getSKPoints(label, label_index, points_num, segment_size):
    """
    SAM2에 입력될 Point Prompt를 추출하는 연산을 수행합니다.
    (아래 getPoints 함수의 기능함수)

    Args:
    label: Point를 뽑아낼 영역의 0/1로 된 binary mask
    label_index: 해당 label이 몇 번째 분할의 label인지를 나타내는 인덱스 지표
    points_num: 해당 label에서 뽑아낼 point의 개수
    segment_size: 해당 label(분할)의 너비

    Returns:
    points: 해당 label에서 추출된 point의 좌표
    """

    ## 연결 영역 계산
    # Definition Part
    area_info = []
    area_idx = []
    selected_area = []

    # Calculate Connected Area
    labels = measure.label(label)
    props = measure.regionprops(labels)

    for region in props:
        min_x, min_y, max_x, max_y = region.bbox
        min_x = min_x+(label_index*segment_size)
        max_x = max_x+(label_index*segment_size)
        area_info.append((region.area, region.label, (min_x, min_y, max_x, max_y)))
    
    # Select Connected Area
    area_info.sort(reverse=True)
    area_info = area_info[0:points_num]
    area_idx = [a[1] for a in area_info]

    for idx in area_idx:
        mask = (labels == idx).astype(np.uint8)
        selected_area.append(mask)
    
    ## 연결 영역이 발생하지 않은 경우 빈 list 반환
    if len(selected_area) <= 0:
        return []

    ## Skeleton Point 계산
    points = []

    connect_mask = np.array(selected_area)
    for i in range(connect_mask.shape[0]):
        connect = connect_mask[i]
        skeleton = morphology.skeletonize(connect)

        y, x = np.nonzero(skeleton)

        if len(x) > 0:
            mid_x = x[np.abs((x-np.median(x))).argmin()].item()
            mid_x_idx = np.nonzero(x == mid_x)[0]
            mid_y_raw = y[mid_x_idx]
            mid_y = mid_y_raw[np.abs((mid_y_raw-np.median(y))).argmin()].item()
            points.append((area_info[i][0], (mid_x+(label_index*segment_size), mid_y), area_info[i][2]))
        else:
            continue
    
    return points

def getPoints(label, segments_num = 3, points_num = 3, onlypoint = False):
    """
    SAM2에 입력될 Point Prompt를 추출하는 연산을 수행합니다.

    Args:
    label: Point를 뽑아낼 영역의 0/1로 된 binary mask
    segments_num: 해당 label을 세로로 몇 분할 할지에 대한 숫자(기본값:3분할)
    points_num: 해당 label의 각 분할에서 최대 몇 개의 point를 뽑아낼지에 대한 숫자(기본값: 3개)

    Returns:
    only_points: 해당 label에서 추출된 point의 좌표
    """

    # Definition Part
    points = []

    # Segment label
    h, w = label.shape
    seg_w = int(np.floor((w/segments_num) * 10) / 10)
    labels = [label[:, (i*seg_w):((i+1)*seg_w)] for i in range(segments_num)]

    # Generate Points
    l_idx = 0
    for l in labels:
        points = points + getSKPoints(l, l_idx, points_num, seg_w)
        l_idx = l_idx + 1

    # Get Points Only
    only_points = [p[1] for p in points]

    return points if onlypoint==False else only_points


