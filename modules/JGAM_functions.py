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
from .ui_gap_measure import Ui_gapMeasurement
from .ui_functions import UIFunctions
from .app_settings import Settings
from .dnn_functions import DNNFunctions

from .utils import *
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

class GapMeasureWindow(QMainWindow, UIFunctions):
    def __init__(self):
        QMainWindow.__init__(self)

        self.ui = Ui_gapMeasurement()
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
        self.TEXT_PROMPT = "Steel joint" 

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

        self.GM = GapMeasureWindow()
        ## Enter 키와 "확인" 버튼에 기능을 탑재하라
        
        """
        expansion Tool
        """

        self.use_jgam = False
    
        mainWidgets.jgamButton.clicked.connect(self.checkExpansionTools)

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
        self.TEXT_PROMPT = self.GD.ui.lineEdit.text()

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
            self.GD_min_x = []
            self.GD_min_y = []
            self.GD_max_x = []
            self.GD_max_y = []

            self.inferenceGroundingDino()

            self.GM.close()

            if self.GD_min_x or self.GD_min_y or self.GD_max_x or self.GD_max_y :
                ## 2. Create SAM's Prompt
                self.promptModel(self.promptVerification)
                
                ## 2.1 Sliding Window Crop
                windows_selected = self.slidingWindowCrop()

                ## 2.2 Inference with SAM
                # Definition Part
                self.total_gap = []
                self.total_joint = []
                self.label[self.label!=0] = 0

                for wn in windows_selected:
                    # 2.2.1 Crop
                    x1, y1, x2, y2 = wn[0], wn[1], wn[2], wn[3]
                    crop_img = self.GD_img_roi[y1:y2, x1:x2]
                    crop_joint = self.joint_GD_roi[y1:y2, x1:x2]
                    crop_gap = self.gap_GD_roi[y1:y2, x1:x2]

                    # 2.2.2 Point Sampling
                    input_point, input_label = self.pointSampling(crop_img, crop_joint, crop_gap, x1, y1, erosion=False)

                    if len(input_point) > 0 :
                        ## 3. SAM inference
                        self.inferenceSAM(crop_img, input_point, input_label, x1, y1)
                        # self.inferenceSAM2(input_point, input_label, input_box)
                    else :
                        print(f"No point")
            
            else :
                print(f"No expansion joint")

            ## 4. Save Results
            self.SaveImg()

            ## 5. Measure joint gap
            # Parameters
            PIXEL_TO_MM = 0.2

            # get masks
            image = cv2.imread(self.imgPath)
            gap_mask = self.label == 2
            gap_mask_t = gap_mask.astype(np.bool).T

            # Calculate Gap Ranges
            transition_ranges = find_transition_ranges(gap_mask_t)

            # Calculate Gap Distances
            filtered_distances = get_distances_by_segment(transition_ranges)

            mm_distances = []
            for segment in filtered_distances:
                mm_distances.append([data['distance'] * PIXEL_TO_MM for data in segment])
            
            # Select Most Frequent Distance
            representative_distances = []

            # 필터링된 구간에 대해 최빈값 계산
            for i, distances in enumerate(mm_distances):
                if len(distances) > 0:
                    representative_distance = find_two_modes(distances)
                    representative_distances.append(representative_distance)

            # Calculate Gap Distance
            final_gap = np.sum(np.array(representative_distances))
            final_gap = round(final_gap, 3)
            print(f"최종 유간 측정 결과: {final_gap}mm")
            self.GM.show()
            self.GM.ui.gapLineEdit.setText(str(f"{final_gap} mm"))


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
        
        self.GD_img_source, GD_img = imread_GD(self.imgPath)
        
        print(f"Grounding DINO status: {self.TEXT_PROMPT}")

        BOX_TRESHOLD = 0.25
        TEXT_TRESHOLD = 0.20

        boxes, logits, phrases = self.inference_groundingDino(model=self.groundingDino_model, 
                                                              image=GD_img,
                                                              caption=self.TEXT_PROMPT,
                                                              box_threshold=BOX_TRESHOLD,
                                                              text_threshold=TEXT_TRESHOLD,
                                                              )
        
        index = logits.argmax()
        box = boxes[index]

        H, W, _ = self.GD_img_source.shape
        box_xyxy = box_ops.box_cxcywh_to_xyxy(box) * torch.Tensor([W, H, W, H])

        min_x, min_y, max_x, max_y = box_xyxy.int().tolist()
        
        self.GD_min_x = min_x
        self.GD_min_y = min_y
        self.GD_max_x = max_x
        self.GD_max_y = max_y

        
        ## 2. Create SAM's Prompt 
    def promptModel(self, verification=False):
        
        if hasattr(self, 'mmseg_model') == False :
            self.load_mmseg(self.mmseg_config_v2, self.mmseg_checkpoint_v2)

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
            
            if thr == self.pred_thr:
                self.label[joint_y_idx, joint_x_idx] = 1
            
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
            
            # visual
            img = img[:, :, :3]
            prompt_colormap = blendImageWithColorMap(img, prompt_label) 

            promptPath = self.imgPath.replace('/leftImg8bit/', '/promptLabelIds/')
            promptPath = promptPath.replace( '_leftImg8bit.png', f'_prompt({thr})_labelIds.png')
            promptColormapPath = promptPath.replace(f'_prompt({thr})_labelIds.png', f"_prompt({thr})_color.png")        
            os.makedirs(os.path.dirname(promptPath), exist_ok=True)
            
            imwrite(promptPath, prompt_label) 
            imwrite_colormap(promptColormapPath, prompt_colormap)

        ## 2.1. Sliding Window Crop
    def slidingWindowCrop(self):
        # Crop with GDINO
        joint = self.label == 1
        joint_bi = joint.astype(np.uint8)
        self.joint_GD_roi = joint_bi[self.GD_min_y:self.GD_max_y, self.GD_min_x:self.GD_max_x]

        gap = self.label == 2
        gap_bi = gap.astype(np.uint8)
        self.gap_GD_roi = gap_bi[self.GD_min_y:self.GD_max_y, self.GD_min_x:self.GD_max_x]

        # # Calculate Distance
        # jmax, jmean, jmin = distance(joint_GD_roi)
        # gamax, gmean, gmin = distance(gap_GD_roi)

        # Set Window Size
        wn_size = 300
        overlap_per = 0.5

        # Generate Windows
        windows = sw.generate(self.GD_img_roi, sw.DimOrder.HeightWidthChannel, wn_size, overlap_per)
        selected_windows = SelectWindow(windows, self.joint_GD_roi, self.gap_GD_roi)
        
        return selected_windows

        ## 2.2.2. Point Sampling
    def pointSampling(self, img, joint, gap, x1, y1, erosion=False):    
        if erosion == True:
            joint = morphology.erosion(joint, morphology.square(15))
            gap = morphology.erosion(gap, morphology.square(15))
        
        # Define Parameters
        segments_num = 3
        points_max_num = 4

        # joint
        joint_points = getPoints(joint, segments_num, points_max_num, onlypoint=True)
        joint_labels = [0] * len(joint_points)

        # gap
        gap_points = getPoints(gap, segments_num, points_max_num, onlypoint=True)
        gap_labels = [1] * len(gap_points)

        # Make Points for Window
        input_point = np.concatenate((np.array(gap_points), np.array(joint_points)), axis=0)
        input_label = np.concatenate((np.array(gap_labels), np.array(joint_labels)), axis=0)

        # Make Points For Image
        self.total_joint = self.total_joint + [(t[0]+x1+self.GD_min_x, t[1]+y1+self.GD_min_y) for t in joint_points]
        self.total_gap = self.total_gap + [(t[0]+x1+self.GD_min_x, t[1]+y1+self.GD_min_y) for t in gap_points]

        return input_point, input_label
        
        ## 3. SAM inference
    def inferenceSAM(self, img, input_point, input_label, x1, y1):
        
        if hasattr(self, 'sam_model') == False :
            self.load_sam(self.sam_checkpoint) 
                
        self.sam_predictor.set_image(img)
        
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        mask = masks[np.argmax(scores), :, :]
        
        # update label with result
        idx = np.argwhere(mask == 1)
        y_idx, x_idx = idx[:, 0]+y1+self.GD_min_y, idx[:, 1]+x1+self.GD_min_x

        self.label[y_idx, x_idx] = 2

        ## 4. Save Image
    def SaveImg(self):
        label_idx = np.argwhere(self.label == 2)
        y_idx, x_idx = label_idx[:, 0], label_idx[:, 1]
        self.colormap = convertLabelToColorMap(self.label, self.label_palette, self.alpha)
        self.colormap[y_idx, x_idx, :3] = self.label_palette[2]

        imwrite(self.labelPath, self.label)

        _colormap = copy.deepcopy(self.colormap)
        sam_colormap = blendImageWithColorMap(self.GD_img_source, self.label)
        img = imread(self.imgPath)

        for joint in self.total_joint:
            cv2.circle(_colormap, (joint[0], joint[1]), 1, (0, 0, 255, 255), -1)
            cv2.circle(sam_colormap, (joint[0], joint[1]), 1, (255, 0, 0, 255), -1)
            cv2.circle(img, (joint[0], joint[1]), 1, (255, 0, 0, 255), -1)
        
        for gap in self.total_gap:
            cv2.circle(_colormap, (gap[0], gap[1]), 1, (255, 0, 0, 255), -1)
            cv2.circle(sam_colormap, (gap[0], gap[1]), 1, (0, 0, 255, 255), -1)
            cv2.circle(img, (gap[0], gap[1]), 1, (0, 0, 255, 255), -1)

        self.color_pixmap = QPixmap(cvtArrayToQImage(_colormap))
        self.color_pixmap_item.setPixmap(QPixmap())
        self.color_pixmap_item.setPixmap(self.color_pixmap)

        colormapPath = os.path.dirname(self.labelPath)
        colormapName = os.path.basename(self.labelPath)
        colormapPath = os.path.dirname(colormapPath)
        colormapPath = os.path.dirname(colormapPath)
        colormapPath = os.path.join(colormapPath, "JGAM_colormap")
        os.makedirs(colormapPath, exist_ok=True)
        colormapPath = os.path.join(colormapPath, colormapName)

        pointName = colormapName.replace("_labelIds.png", "_point.png")
        pointmapPath = os.path.join(os.path.dirname(colormapPath), pointName)

        imwrite_colormap(colormapPath, sam_colormap)
        cv2.imwrite(pointmapPath, img)

        ## 3. SAM inference
    # def inferenceSAM2(self, input_point, input_label, input_box):
        
    #     if hasattr(self, 'sam2_model') == False :
    #         self.load_sam2(self.sam2_config, self.sam2_checkpoint) 

    #     img = cvtPixmapToArray(self.pixmap)
    #     img = img[:, :, :3]
    #     # img_roi = img[self.GD_min_y:self.GD_max_y, self.GD_min_x:self.GD_max_x, :3]
                
    #     self.sam2_predictor.set_image(img)
        
    #     masks, scores, logits = self.sam2_predictor.predict(
    #         point_coords=input_point,
    #         point_labels=input_label,
    #         box=input_box, 
    #         multimask_output=True,
    #     )

    #     mask = masks[np.argmax(scores), :, :]
        
    #     # update label with result
    #     idx = np.argwhere(mask == 1)
    #     y_idx, x_idx = idx[:, 0], idx[:, 1]

    #     self.label[self.label!=0] = 0
    #     self.label[y_idx, x_idx] = 2

    #     self.colormap = convertLabelToColorMap(self.label, self.label_palette, self.alpha)
    #     self.colormap[y_idx, x_idx, :3] = self.label_palette[2]

    #     imwrite(self.labelPath, self.label)

    #     _colormap = copy.deepcopy(self.colormap)
    #     sam_colormap = blendImageWithColorMap(img, self.label)
    #     img = imread(self.imgPath)

    #     for joint in self.top6_joint:
    #         cv2.circle(_colormap, (joint[0], joint[1]), 9, (0, 0, 255, 255), -1)
    #         cv2.circle(sam_colormap, (joint[0], joint[1]), 9, (255, 0, 0, 255), -1)
    #         cv2.circle(img, (joint[0], joint[1]), 9, (255, 0, 0, 255), -1)
        
    #     for gap in self.top6_gap:
    #         cv2.circle(_colormap, (gap[0], gap[1]), 9, (255, 0, 0, 255), -1)
    #         cv2.circle(sam_colormap, (gap[0], gap[1]), 9, (0, 0, 255, 255), -1)
    #         cv2.circle(img, (gap[0], gap[1]), 9, (0, 0, 255, 255), -1)

    #     self.color_pixmap = QPixmap(cvtArrayToQImage(_colormap))
    #     self.color_pixmap_item.setPixmap(QPixmap())
    #     self.color_pixmap_item.setPixmap(self.color_pixmap)

    #     colormapPath = os.path.dirname(self.labelPath)
    #     colormapName = os.path.basename(self.labelPath)
    #     colormapPath = os.path.dirname(colormapPath)
    #     colormapPath = os.path.dirname(colormapPath)
    #     colormapPath = os.path.join(colormapPath, "JGAM_colormap")
    #     os.makedirs(colormapPath, exist_ok=True)
    #     colormapPath = os.path.join(colormapPath, colormapName)

    #     pointName = colormapName.replace("_labelIds.png", "_point.png")
    #     pointmapPath = os.path.join(os.path.dirname(colormapPath), pointName)

    #     imwrite_colormap(colormapPath, sam_colormap)
    #     cv2.imwrite(pointmapPath, img)
        

