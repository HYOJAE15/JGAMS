import os

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2,40).__str__()

import cv2

from PIL import Image

import numpy as np

from numba import njit

from PySide6.QtGui import QImage

import slidingwindow as sw 
import math 

import csv

import submodules.GroundingDINO.groundingdino.datasets.transforms as T

from skimage.measure import label, regionprops, regionprops_table
from skimage import data, measure, morphology



def imread(path: str, checkImg: bool=True) -> np.array:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

    # if img.ndim == 3 : 
    if checkImg :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    
    return img

def imread_GD(path: str):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    img_width, img_height = pil_img.size

    transform = T.Compose(
        [T.RandomResize([800], max_size=1333),
         T.ToTensor(),
         T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
         ])

    image_source = np.asarray(pil_img)
    image_tr, _ = transform(pil_img.convert("RGB"), None)


    return image_source, image_tr

def imwrite(path:str,
            img: np.array
            )-> None: 
    _, ext = os.path.splitext(path)

    if img.ndim == 3 : 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 

    _, label_to_file = cv2.imencode(ext, img)
    label_to_file.tofile(path)

def imwrite_colormap(path, img): 
    _, ext = os.path.splitext(path)
    _, label_to_file = cv2.imencode(ext, img)
    label_to_file.tofile(path)

def createLayersFromLabel(label: np.array, 
                          num_class: int
                          ) -> list([np.array]):
    layers = []

    for idx in range(num_class):
        layers.append(label == idx)
        
    return layers

def cvtArrayToQImage(array: np.array) -> QImage:

    if len(array.shape) == 3 : 

        h, w, c = array.shape
        if c == 3:
            return QImage(array.data, w, h, 3 * w, QImage.Format_RGB888)
        elif c == 4: 
            return QImage(array.data, w, h, 4 * w, QImage.Format_RGBA8888)

    elif len(array.shape) == 2 :
        h, w = array.shape
        return QImage(array.data, w, h, QImage.Format_Mono)
    
def cvtPixmapToArray(pixmap):
    """Convert a QPixmap to a numpy array
    Args: 
        pixmap (QPixmap): The QPixmap to convert
    
    Returns:
        img (np.array): The converted QPixmap
    """
    
    ## Get the size of the current pixmap
    size = pixmap.size()
    h = size.width()
    w = size.height()

    ## Get the QImage Item and convert it to a byte string
    qimg = pixmap.toImage()
    byte_str = qimg.bits().tobytes()

    ## Using the np.frombuffer function to convert the byte string into an np array
    img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w,h,4))

    return img

@njit
def mapLabelToColorMap(label: np.array, 
                       colormap: np.array, 
                       palette: list(list([int, int, int]))
                       )-> np.array:
    
    assert label.ndim == 2, "label must be 2D array"
    assert colormap.ndim == 3, "colormap must be 3D array"
    assert colormap.shape[2] == 4, "colormap must have 4 channels"

    for x in range(label.shape[0]):
        for y in range(label.shape[1]):
            colormap[x, y, :3] = palette[label[x, y]][:3]

    return colormap

def convertLabelToColorMap(
        label: np.array,
        palette: list(list([int, int, int])),
        alpha: int) -> np.array:
    assert label.ndim == 2, "label must be 2D array"
    assert alpha >= 0 and alpha <= 255, "alpha must be between 0 and 255"

    colormap = np.zeros((label.shape[0], label.shape[1], 4), dtype=np.uint8)
    colormap = mapLabelToColorMap(label, colormap, palette)
    colormap[:, :, 3] = alpha

    return colormap

def generateForNumberOfWindows(data, dimOrder, windowCount, overlapPercent, transforms=[]):
	"""
	Generates a set of sliding windows for the specified dataset, automatically determining the required window size in
	order to create the specified number of windows. `windowCount` must be a tuple specifying the desired number of windows
	along the Y and X axes, in the form (countY, countX).
	"""
	
	# Determine the dimensions of the input data
	width = data.shape[dimOrder.index('w')]
	height = data.shape[dimOrder.index('h')]
	
	# Determine the window size required to most closely match the desired window count along both axes
	countY, countX = windowCount
	windowSizeX = math.ceil(width / countX)
	windowSizeY = math.ceil(height / countY)
	
	# Generate the windows
	return sw.generateForSize(
		width,
		height,
		dimOrder,
		0,
		overlapPercent,
		transforms,
		overrideWidth = windowSizeX,
		overrideHeight = windowSizeY
	)

def blendImageWithColorMap(image,
                           label,
                           palette = np.array([[0, 0, 0],
                                               [0, 255, 0],
                                               [0, 0, 255],
                                               [0, 255, 255],
                                               [255, 0, 0]
                                               ]),
                           alpha = 0.5
        ):
    """ blend image with color map 
    Args: 
        image (3d np.array): RGB image
        label (2d np.array): 1 channel gray-scale image
        pallete (2d np.array) 
        alpha (float)

    Returns: 
        color_map (3d np.array): RGB image
    """

    color_map = np.zeros_like(image)
        
    for idx, color in enumerate(palette) : 
        
        if idx == 0 :
            color_map[label == idx, :] = image[label == idx, :] * 1
        else :
            color_map[label == idx, :] = image[label == idx, :] * alpha + color * (1-alpha)

    return color_map

def getPrompt(ROI, point, label, path, OR_Rect):
    """
    실험 결과 분석을 위하여 세부내용(Prompt)을 csv 파일로 저장한다. 
    """
    fields = ["File", "min X", "min Y", "max X", "max Y", "Overlap Rate", "Prompt(point)", "Prompt(label)"]
    csv_name = os.path.basename(path)
    
    if os.path.isfile(path) == False:
        samData_list = [csv_name, ROI[0], ROI[1], ROI[2], ROI[3], 0, point, label]
    

        with open(path, "x", encoding="cp949", newline="") as f:
            write = csv.writer(f)
            write.writerow(fields)

            write.writerow(samData_list)
            
    elif os.path.isfile(path) : 
        latestData_list = []
        
        with open(path, "r", encoding="cp949", newline="") as f:
            rows = csv.reader(f)
            for row in rows:
                latestData_list.append(row)
        
        for idx in latestData_list[1:] :
            start=[int(idx[1]), int(idx[2])]
            end = [int(idx[3]), int(idx[4])]
            OR_Rect = cv2.rectangle(OR_Rect, start, end, (1, 1, 1), -1)

        roi_last = np.zeros(OR_Rect.shape)
        roi_last_start = [ROI[0], ROI[1]]
        roi_last_end = [ROI[2], ROI[3]]
        roi_last = cv2.rectangle(roi_last, roi_last_start, roi_last_end, (1, 1, 1), -1)

        union = np.count_nonzero(OR_Rect)
        last_roi = cv2.countNonZero(roi_last)
        

        if union > 0 : 
            intersection = cv2.bitwise_and(roi_last, OR_Rect)
            intersection_roi = cv2.countNonZero(intersection)
            print(f"intersection_roi: {intersection_roi}")
            print(f"last_roi: {last_roi}")
            overlap_rate = intersection_roi/last_roi
            print(f"overlap rate: {overlap_rate}")
            
        elif union == 0 :
            overlap_rate = 0
            
        
        samData_list = [csv_name, ROI[0], ROI[1], ROI[2], ROI[3], overlap_rate, point, label]
    

        with open(path, "a", encoding="cp949", newline="") as f:
            write = csv.writer(f)
            write.writerow(samData_list)


