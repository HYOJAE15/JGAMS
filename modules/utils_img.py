import numpy as np

from numba import njit

from .utils import imread, cvtArrayToQImage
from PySide6.QtGui import  QPixmap

import cv2

import torch
from torchvision.ops import box_convert
import supervision as sv

@njit(fastmath=True)
def _applyBrushSize(brushSize):
    """
    apply brush size to X, Y
    Args:
        brushSize (int): brush size
    
    Returns:
        tuple(list, list, int): x, y, width
    """

    width = int(brushSize // 2)

    _X, _Y = [], []

    if brushSize % 2 == 0:
        for _x in range(-width, width):
            for _y in range(-width,width):
                _X.append(_x)
                _Y.append(_y)
    else:
        for _x in range(-width, width+1):
            for _y in range(-width, width+1):
                _X.append(_x)
                _Y.append(_y)

    return _X, _Y, width


def applyBrushSize(X, Y, brushSize, max_x, max_y, brushType = 'rectangle'): 
    """
    apply brush size to X, Y
    Args:
        X (list): x coordinate
        Y (list): y coordinate
        brushSize (int): brush size
        max_x (int): max x
        max_y (int): max y
        brushType (str): brush type, rectangle or circle
    
    Returns:
        tuple(np.ndarray, np.ndarray): x, y
    """
    assert isinstance(X, list) or isinstance(X, np.ndarray), "X must be list or np.ndarray"
    assert isinstance(Y, list) or isinstance(Y, np.ndarray), "Y must be list or np.ndarray"
    assert isinstance(brushSize, int), "brushSize must be int"
    assert isinstance(max_x, int), "max_x must be int"
    assert isinstance(max_y, int), "max_y must be int"
    assert isinstance(brushType, str), "brushType must be str"

    _X, _Y, width = _applyBrushSize(brushSize)
    
    if brushType == 'circle' :
        _X, _Y = convetRectangleToCircle(_X, _Y, width)
        
    return_x = []
    return_y = []
    
    for x, y in zip(X, Y):
        _x = x + _X
        _y = y + _Y

        return_x += _x.tolist()
        return_y += _y.tolist()

    return_x = np.array(return_x)
    return_y = np.array(return_y)

    return_x = np.clip(return_x, 0, max_x-1)
    return_y = np.clip(return_y, 0, max_y-1)

    _return = np.vstack((return_x, return_y))
    _return = np.unique(_return, axis=1)
    return_x , return_y = _return[0, :], _return[1, :]
    
    return return_x, return_y

@njit(fastmath=True)
def convetRectangleToCircle(X, Y, width):
    """
    convert rectangle to circle
    Args:
        X (list): x coordinate
        Y (list): y coordinate
        width (int): width of the circle
    
    Returns:
        tuple(np.ndarray): X, Y
    """
    
    dist = [np.sqrt(_x**2 + _y**2) for _x, _y in zip(Y, X)]
    Y =  [_y for idx, _y in enumerate(Y) if dist[idx] < width]
    X = [_x for idx, _x in enumerate(X) if dist[idx] < width]
    return np.array(X), np.array(Y)

@njit(fastmath=True)
def fast_coloring(X, Y, array, label_palette, brush_class, alpha = 50):
    """
    fast coloring
    Args:
        X (list): x coordinate
        Y (list): y coordinate
        array (np.ndarray): image array (RGBA)
        label_palette (np.ndarray): label palette
        brush_class (int): brush class
        alpha (int): alpha value
    
    Returns:
        array (np.ndarray): image array
    """
    # assertion 
    assert len(X) == len(Y), "X and Y must have the same length"
    assert len(X) > 0, "X and Y must have at least one element"
    assert array.shape[2] == 4, "array must be RGBA"
    assert alpha >= 0 and alpha <= 255, "alpha must be between 0 and 255"

    for x, y in zip(X, Y): 
        array[y, x, :3] = label_palette[brush_class]
        array[y, x, 3] = alpha

    return array 

def annotate_GD(image_source, boxes, logits, phrases) -> np.ndarray:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    box_annotator = sv.BoxAnnotator(text_scale=3, text_thickness=5)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    return annotated_frame

def getScaledPoint(event, scale):
    """
    get scaled point
    Args:
        event: QMouseEvent
        scale: float

    Returns:
        tuple(int): x, y
    """

    x, y = round(event.x() / scale), round(event.y() / scale)

    return x, y

def getScaledPoint_mmdet(point, scale):
    """
    get mmdet point 
    Args:
        point: mmdet bbox
        scale: float

    Returns:
        tuple(int): x, y
    """

    min_x, min_y, max_x, max_y = round(float(point[0]) / scale), round(float(point[1]) / scale), round(float(point[2]) / scale), round(float(point[3]) / scale)

    return min_x, min_y, max_x, max_y

def getCoordBTWTwoPoints(x1, y1, x2, y2):
    """
    coordinate between two points
    Args: 
        x1, y1, x2, y2: int or list(int)

    Returns:
        tuple(np.ndarray): x, y
    """

    d0 = x2 - x1
    d1 = y2 - y1
    
    count = max(abs(d1)+1, abs(d0)+1)

    if d0 == 0:
        return (
            np.full(count, x1),
            np.round(np.linspace(y1, y2, count)).astype(np.int32)
        )

    if d1 == 0:
        return (
            np.round(np.linspace(x1, x2, count)).astype(np.int32),
            np.full(count, y1),  
        )

    return (
        np.round(np.linspace(x1, x2, count)).astype(np.int32),
        np.round(np.linspace(y1, y2, count)).astype(np.int32)
    )

def readImageToPixmap(path):
        """Read image to pixmap
        Args:
            path (str): Image path
        Returns:
            QPixmap: Image pixmap
        """
        img = imread(path)
        return QPixmap(cvtArrayToQImage(img))

def histEqualization_gr (img):
    
    src_gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst_gr = cv2.equalizeHist(src_gr)
    dst_gr_bgr = cv2.cvtColor(dst_gr, cv2.COLOR_GRAY2BGR)
    
    return dst_gr_bgr

def histEqualization_hsv (img):
    
    src_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(src_hsv)
    dst_hsv_v = cv2.equalizeHist(v)
    dst_hsv_merged = cv2.merge([h,s,dst_hsv_v])
    dst_hsv_merged_bgr = cv2.cvtColor(dst_hsv_merged, cv2.COLOR_HSV2BGR)

    return dst_hsv_merged_bgr

def histEqualization_ycc (img):
    
    src_ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, Cr, Cb = cv2.split(src_ycc)
    dst_ycc_y = cv2.equalizeHist(y)
    dst_ycc_merged = cv2.merge([dst_ycc_y, Cr, Cb])
    dst_ycc_merged_bgr = cv2.cvtColor(dst_ycc_merged, cv2.COLOR_YCrCb2BGR)

    return dst_ycc_merged_bgr
