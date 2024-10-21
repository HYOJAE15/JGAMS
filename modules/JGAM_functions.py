import os

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2,40).__str__()

import cv2
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import  QPixmap
from PySide6.QtWidgets import (QMainWindow, QFileSystemModel) 

from .ui_main import Ui_MainWindow
from .ui_dino_prompt import Ui_DinoPrompt
from .ui_prompt_model import Ui_promptModel
from .ui_functions import UIFunctions
from .app_settings import Settings
from .dnn_functions import DNNFunctions

from .utils import *
from .utils_img import (annotate_GD)
from .utils_JGAM import *

from modules.utils import imwrite_colormap

from submodules.GroundingDINO.groundingdino.util import box_ops

import torch

from skimage.measure import label, regionprops
from skimage import morphology

import copy

class PromptModelWindow(QMainWindow, UIFunctions):
    def __init__(self):
        QMainWindow.__init__(self)

        self.ui = Ui_promptModel()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        self.settings = Settings()

        self.uiDefinitions()

    def resizeEvent(self, event):
        self.resize_grips()

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()

class DinoPromptWindow(QMainWindow, UIFunctions):
    def __init__(self):
        QMainWindow.__init__(self)

        self.ui = Ui_DinoPrompt()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        self.settings = Settings()

        self.uiDefinitions()

    def resizeEvent(self, event):
        self.resize_grips()

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()




class JGAMFunctions(DNNFunctions):
    def __init__(self):
        DNNFunctions.__init__(self)
        
        if not hasattr(self, 'ui'):
            QMainWindow.__init__(self)
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)

        global mainWidgets
        mainWidgets = self.ui
            
        self.fileModel = QFileSystemModel()
        
        """
        Attribute
        """
        self.pred_thr = 0.80
        self.area_thr = 250
        self.fill_thr = 250
        
        """
        Experiment
        """

        self.promptVerification = False
        self.promptErosion = False
        
        """
        Pompt Tool
        """

        self.use_GD = False
        self.use_PM = False
        
        mainWidgets.GDButton.clicked.connect(self.openGD)
        mainWidgets.PMButton.clicked.connect(self.openPM)

        # Grounding-DINO
        self.GD = DinoPromptWindow()
        ## Enter 키와 "확인" 버튼에 기능을 탑재하라
        self.GD.ui.lineEdit.returnPressed.connect(self.changePrompt)
        
        # Prompt Model (DeeplabV3+)
        self.PM = PromptModelWindow()
        self.PM.ui.thrSlider.valueChanged.connect(self.changeThreshold)
        self.PM.ui.verCheckBox.stateChanged.connect(self.changeThrVerification)
        

        """
        expansion Tool
        """

        self.use_jgam = False
    
        mainWidgets.jgamButton.clicked.connect(self.checkExpansionTools)

    def updateColorMap(self):
        """
        Update the color map
        """
        self.colormap = convertLabelToColorMap(self.label, self.label_palette, self.alpha)
        self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.color_pixmap_item.setPixmap(QPixmap())
        self.color_pixmap_item.setPixmap(self.color_pixmap)

    
    def openGD(self):
        """
        Open or Close Grounding DINO Prompt
        """
        if self.use_GD == False:
            self.GD.show()
            self.use_GD = True
            # if hasattr(self, 'PM'):
            #     self.PM.close()  
            self.set_button_state(use_GD=self.use_GD, use_PM=self.use_PM)

        elif self.use_GD == True:
            self.GD.close()
            self.use_GD = False
            self.set_button_state(use_GD=self.use_GD, use_PM=self.use_PM)

    
    def openPM(self):
        """
        Open or Close Prompt Model Threshold Menu
        """
        if self.use_PM == False:
            self.PM.show()
            self.use_PM = True
            # if hasattr(self, 'PM'):
            #     self.PM.close()  
            self.set_button_state(use_GD=self.use_GD, use_PM=self.use_PM)

        elif self.use_PM == True:
            self.PM.close()
            self.use_PM = False
            self.set_button_state(use_GD=self.use_GD, use_PM=self.use_PM)

    def changePrompt(self):
        self.GDPromptText = self.GD.ui.lineEdit.text()
        print(self.GDPromptText)

    def changeThreshold(self, value):
        self.pred_thr = float(value/100)
        self.PM.ui.brushSizeText.setText(str(f"{value} %"))
    
    def changeThrVerification(self):
        if self.PM.ui.verCheckBox.isChecked():
            self.promptVerification = True
            print(self.promptVerification)
            
        if self.PM.ui.verCheckBox.isChecked() == False:
            self.promptVerification = False
            print(self.promptVerification)
        
    def set_button_state(self,
                         use_jgam=False,
                         use_GD=False,
                         use_PM=False
                         ):
        """
        Set the state of the buttons
        """
        
        self.use_jgam = use_jgam
        self.use_GD = use_GD
        self.use_PM = use_PM
        
        mainWidgets.jgamButton.setChecked(use_jgam)
        mainWidgets.GDButton.setChecked(use_GD)
        mainWidgets.PMButton.setChecked(use_PM)
        
    def checkExpansionTools(self):
        
        ### JGAMS
        if hasattr(self, 'imgPath') :
            ## 1. Grounding DINO
            self.inferenceGroundingDino()

            if hasattr(self, 'GD_min_x') :
                ## 2. Create SAM's Prompt
                self.promptModel(self.promptVerification)
                ## 2.1 Point Sampling
                input_point, input_label, input_box = self.pointSampling(self.promptErosion)
                
                if len(input_point) > 0 :
                    ## 3. SAM inference
                    self.inferenceSAM(input_point, input_label, input_box)
                else :
                    print(f"No point")
    
            
            else :
                print(f"No expansion joint")

            ## 4. Measure joint gap
            image = cv2.imread(self.imgPath)
            gap_mask = self.label == 2
            # gap_mask = np.array(gap_mask, dtype=np.uint8)
            
            mask, image, region_data, all_thicknesses, all_thickness_positions = self.gap_measure(gap_mask, image)
            print(f"region_data: {region_data}")

        else:
            print(f"No image")     

        """
        Automatic Joint Gap Measurement 
        """

        ## 1. Grounding DINO
    def inferenceGroundingDino(self):
        
        self.load_groundingDino(self.groundingDino_config,
                                self.groundingDino_checkpoint
                                )
        
        GD_img_source, GD_img = imread_GD(self.imgPath)
        
        TEXT_PROMPT = "Steel joint" 
        BOX_TRESHOLD = 0.25
        TEXT_TRESHOLD = 0.20


        boxes, logits, phrases = self.inference_groundingDino(model=self.groundingDino_model, 
                                                              image=GD_img,
                                                              caption=TEXT_PROMPT,
                                                              box_threshold=BOX_TRESHOLD,
                                                              text_threshold=TEXT_TRESHOLD,
                                                              )
        
        annotated_frame = annotate_GD(image_source=GD_img_source,
                                      boxes=boxes, logits=logits,
                                      phrases=phrases
                                      )
        
        annotated_frame = annotated_frame[...,::-1]
        index = logits.argmax()
        box = boxes[index]

        H, W, _ = GD_img_source.shape
        box_xyxy = box_ops.box_cxcywh_to_xyxy(box) * torch.Tensor([W, H, W, H])

        min_x, min_y, max_x, max_y = box_xyxy.int().tolist()
        
        self.GD_min_x = min_x
        self.GD_min_y = min_y
        self.GD_max_x = max_x
        self.GD_max_y = max_y

        
        ## 2. Create SAM's Prompt 
    def promptModel(self, verification=False):
        
        if hasattr(self, 'mmseg_model') == False :
            self.load_mmseg(self.mmseg_config, self.mmseg_checkpoint)

        img = cvtPixmapToArray(self.pixmap)
        self.GD_img_roi = img[self.GD_min_y:self.GD_max_y, self.GD_min_x:self.GD_max_x, :3]
        back, joint, gap, logits = self.inference_mmseg(self.GD_img_roi)

        """
        nomalize the segmentation logits
        """
        
        if len(np.nonzero(self.label[0])) > 0:
                self.label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        print(f"promptModel status: {verification}, {self.pred_thr}")
                
        pred_thrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99] if verification else [self.pred_thr]
        
        for thr in pred_thrs:

            prompt_label = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

            # joint
            joint_logit = logits[1, :, :]
            joint_score = min_max_normalize(joint_logit)
            joint_bi = extract_values_above_threshold(joint_score, thr)
            joint_bi = morphology.remove_small_objects(joint_bi, self.area_thr)
            joint_bi = morphology.remove_small_holes(joint_bi, self.fill_thr)
            
            joint_idx = np.argwhere(joint_bi == 1)
            joint_y_idx, joint_x_idx = joint_idx[:, 0], joint_idx[:, 1]
            joint_x_idx = joint_x_idx + self.GD_min_x
            joint_y_idx = joint_y_idx + self.GD_min_y

            prompt_label[joint_y_idx, joint_x_idx] = 1
            if thr == 0.8:
                self.label[joint_y_idx, joint_x_idx] = 1
            # self.colormap[joint_y_idx, joint_x_idx, :3] = self.label_palette[1]

            # gap
            gap_logit = logits[2, :, :]
            gap_score = min_max_normalize(gap_logit)
            gap_bi = extract_values_above_threshold(gap_score, thr)
            gap_bi = morphology.remove_small_objects(gap_bi, self.area_thr)
            gap_bi = morphology.remove_small_holes(gap_bi, self.fill_thr)
            
            gap_idx = np.argwhere(gap_bi == 1)
            gap_y_idx, gap_x_idx = gap_idx[:, 0], gap_idx[:, 1]
            gap_x_idx = gap_x_idx + self.GD_min_x
            gap_y_idx = gap_y_idx + self.GD_min_y

            prompt_label[gap_y_idx, gap_x_idx] = 2
            if thr == self.pred_thr:
                self.label[gap_y_idx, gap_x_idx] = 2
            # self.colormap[gap_y_idx, gap_x_idx, :3] = self.label_palette[2]

            # visual
            img = img[:, :, :3]
            prompt_colormap = blendImageWithColorMap(img, prompt_label) 

            promptPath = self.imgPath.replace('/leftImg8bit/', '/promptLabelIds/')
            promptPath = promptPath.replace( '_leftImg8bit.png', f'_prompt({thr})_labelIds.png')
            promptColormapPath = promptPath.replace(f'_prompt({thr})_labelIds.png', f"_prompt({thr})_color.png")        
            os.makedirs(os.path.dirname(promptPath), exist_ok=True)
            
            imwrite(promptPath, prompt_label) 
            imwrite_colormap(promptColormapPath, prompt_colormap)

        ## 2.1. Point Sampling
    def pointSampling(self, erosion=False):
        
        joint = self.label == 1
        gap = self.label == 2
        
        if erosion == True:
            joint = morphology.erosion(joint, morphology.square(15))
            gap = morphology.erosion(gap, morphology.square(15))
        
        # joint
        top6_joint = getTop6Skeletonize(joint, onlycenter=True)
        self.top6_joint = np.array(top6_joint)
        self.top6_joint_label = np.zeros((self.top6_joint.shape[0]), dtype=int)
        # gap
        top6_gap = getTop6Skeletonize(gap, onlycenter=True)
        self.top6_gap = np.array(top6_gap)
        self.top6_gap_label = np.ones((self.top6_gap.shape[0]), dtype=int)

        input_point = np.concatenate((self.top6_gap, self.top6_joint), axis=0)
        input_label = np.concatenate((self.top6_gap_label, self.top6_joint_label), axis=0)
        
        input_box = np.array([self.GD_min_x, self.GD_min_y, self.GD_max_x, self.GD_max_y])

        return input_point, input_label, input_box
        
        ## 3. SAM inference
    def inferenceSAM(self, input_point, input_label, input_box):
        
        if hasattr(self, 'sam_model') == False :
            self.load_sam(self.sam_checkpoint) 

        img = cvtPixmapToArray(self.pixmap)
        img = img[:, :, :3]
        # img_roi = img[self.GD_min_y:self.GD_max_y, self.GD_min_x:self.GD_max_x, :3]
                
        self.sam_predictor.set_image(img)
        
        
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box, 
            multimask_output=True,
        )

        mask = masks[np.argmax(scores), :, :]
        self.sam_mask_input = logits[np.argmax(scores), :, :]

        # update label with result
        idx = np.argwhere(mask == 1)
        y_idx, x_idx = idx[:, 0], idx[:, 1]

        self.GD_sam_y_idx = y_idx
        self.GD_sam_x_idx = x_idx
        
        self.label[self.label==1] = 0
        
        self.updateColorMap()

        self.label[y_idx, x_idx] = 2
        self.colormap[y_idx, x_idx, :3] = self.label_palette[2]

        # imwrite(self.labelPath, self.label)

        _colormap = copy.deepcopy(self.colormap)

        for joint in self.top6_joint:
            
            # cv2.circle(_colormap, (joint[0], joint[1]), 50, (0, 0, 255, 255), 9)
            cv2.circle(_colormap, (joint[0], joint[1]), 9, (0, 0, 255, 255), -1)
        for gap in self.top6_gap:
            
            # cv2.circle(_colormap, (gap[0], gap[1]), 50, (255, 0, 0, 255), 9)
            cv2.circle(_colormap, (gap[0], gap[1]), 9, (255, 0, 0, 255), -1)
        


        self.color_pixmap = QPixmap(cvtArrayToQImage(_colormap))
        self.color_pixmap_item.setPixmap(QPixmap())
        self.color_pixmap_item.setPixmap(self.color_pixmap)

        sam_colormap = blendImageWithColorMap(img, self.label) 
        img = imread(self.imgPath)

        for joint in self.top6_joint:
            
            cv2.circle(sam_colormap, (joint[0], joint[1]), 9, (255, 0, 0, 255), -1)
            cv2.circle(img, (joint[0], joint[1]), 9, (255, 0, 0, 255), -1)
        for gap in self.top6_gap:
            
            cv2.circle(sam_colormap, (gap[0], gap[1]), 9, (0, 0, 255, 255), -1)
            cv2.circle(img, (gap[0], gap[1]), 9, (0, 0, 255, 255), -1)
        
        
        colormapPath = os.path.dirname(self.labelPath)
        colormapName = os.path.basename(self.labelPath)
        colormapPath = os.path.dirname(colormapPath)
        colormapPath = os.path.dirname(colormapPath)
        colormapPath = os.path.join(colormapPath, "JGAM_colormap")
        os.makedirs(colormapPath, exist_ok=True)
        colormapPath = os.path.join(colormapPath, colormapName)

        pointName = colormapName.replace("_labelIds.png", "point.png")
        pointmapPath = os.path.join(os.path.dirname(colormapPath), pointName)

        imwrite_colormap(colormapPath, sam_colormap)
        cv2.imwrite(pointmapPath, img)
        
        ## 4. Measure joint gpa
    def gap_measure(self,
                    mask: np.ndarray,
                    image: np.ndarray,
                    pixel_resolution: float=0.5,
                    min_region_size: int=100):
        """
        주어진 마스크와 이미지를 사용하여 유간을 계산합니다.

        Args:
        mask: 원본 탐지 마스크  
        image: 원본 이미지
        pixel_resolution: 픽셀 해상도 (기본값: 0.5mm)
        min_region_size: 최소 영역 크기 (기본값: 100)

        Returns:
        두께 계산 결과
        """
        labeled_mask = label(mask, connectivity=1)
        regions = regionprops(labeled_mask)
        
        if not regions:
            return float('-inf')
        
        region_data = []
        all_thicknesses = []
        all_thickness_positions = []
        
        for region in regions:
            if region.area >= min_region_size:
                region_mask = labeled_mask == region.label
                thicknesses, thickness_positions = calculate_thickness(region_mask, pixel_resolution)
                
                if thicknesses:
                    mean_thickness = np.mean(thicknesses)
                    mode_thickness = calculate_mode(thicknesses)
                    
                    mean_positions = [pos for thickness, pos in zip(thicknesses, thickness_positions) if np.isclose(thickness, mean_thickness, atol = 0.1)]
                    mode_positions = [pos for thickness, pos in zip(thicknesses, thickness_positions) if np.isclose(thickness, mode_thickness, atol = 0.1)]
                    
                    thickness_cv = np.std(thicknesses) / mean_thickness if mean_thickness != 0 else float('inf')
                    
                    region_data.append({
                        'label': region.label,
                        'thicknesses': thicknesses,
                        'positions' : thickness_positions,
                        'mean_thickness' : mean_thickness,
                        'mode_thickness' : mode_thickness,
                        'mean_positions' : mean_positions,
                        'mode_positions' : mode_positions,
                        'cv' : thickness_cv
                    })
                    
                    all_thicknesses.extend(thicknesses)
                    all_thickness_positions.extend(thickness_positions)
        
        return mask, image, region_data, all_thicknesses, all_thickness_positions
        
     

            
