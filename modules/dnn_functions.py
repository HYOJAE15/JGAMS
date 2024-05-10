from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QGraphicsScene

from .ui_main import Ui_MainWindow
from .ui_sam_window import Ui_SAMWindow
from .ui_functions import UIFunctions
from .app_settings import Settings

from mmseg.apis import init_model, inference_model

from mmdet.apis import init_detector, inference_detector

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
from skimage.morphology import skeletonize

import pydensecrf.densecrf as dcrf 
import numpy as np

import skimage.morphology

from .utils import cvtPixmapToArray

from segment_anything import sam_model_registry, SamPredictor

class SAMWindow(QMainWindow, UIFunctions):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_SAMWindow()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        self.settings = Settings()

        self.uiDefinitions()

        # add qlabels to scroll area

    def resizeEvent(self, event):
        self.resize_grips()

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()

    def setScene(self, pixmap, color_pixmap, scale=1.0):
        """
        Set the scene of the image
        Args:
            pixmap (QPixmap): The pixmap of the image.
            color_pixmap (QPixmap): The pixmap of the color image.
            scale (float): The scale of the scene.
        """
        self.scene = QGraphicsScene()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.color_pixmap_item = self.scene.addPixmap(color_pixmap)
        self.ui.graphicsView.setScene(self.scene)
        self.scaleScene(scale=scale)

    def scaleScene(self, scale=1.0):
        """
        Scale the scene
        Args:
            scale (float): The scale of the scene.
        """
        self.ui.graphicsView.setFixedSize(scale * self.pixmap_item.pixmap().size())
        self.ui.graphicsView.fitInView(self.pixmap_item)



class DNNFunctions(object):
    def __init__(self):

        if not hasattr(self, 'ui'):
            QMainWindow.__init__(self)
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)

        self.SAMWindow = SAMWindow()

        ######################### 
        # Semantic Segmentation #
        #########################
        
        # MMSegmentation
        self.mmseg_config = 'dnn/configs/cgnet.py'
        self.mmseg_checkpoint = 'dnn/checkpoints/cgnet.pth'
        # Segment Anything
        self.sam_checkpoint = 'dnn/checkpoints/sam_vit_h_4b8939.pth'

        #################### 
        # Object Detection #
        ####################

        # Crack
        self.mmdet_crack_config = 'dnn/configs/fasterrcnn_r50_crack_2x.py'
        self.mmdet_crack_checkpoint = 'dnn/checkpoints/fasterrcnn_r50_crack_2x.pth'
        # Efflorescence
        self.mmdet_efflorescence_config = 'dnn/configs/fasterrcnn_r50_efflorescence_2x.py'
        self.mmdet_efflorescence_checkpoint = 'dnn/checkpoints/fasterrcnn_r50_efflorescence_2x.pth'
        # Rebar Exposure
        self.mmdet_rebarExposure_config = 'dnn/configs/fasterrcnn_r50_rebarExposure_2x.py'
        self.mmdet_rebarExposure_checkpoint = 'dnn/checkpoints/fasterrcnn_r50_rebarExposure_2x.pth'
        # Spalling
        self.mmdet_spalling_config = 'dnn/configs/fasterrcnn_r50_spalling_2x.py'
        self.mmdet_spalling_checkpoint = 'dnn/checkpoints/fasterrcnn_r50_spalling_2x.pth'
        
        ##############
        # Attributes #
        ##############
        
        self.scale = 1.0

    
    def load_sam(self, checkpoint, mode='default'):
        """
        Load the sam model
        Args:
            mode (str): The mode of the sam model.
        """
        self.sam_model = sam_model_registry[mode](checkpoint=checkpoint)
        self.sam_model.to(device='cuda:0')
        self.sam_predictor = SamPredictor(self.sam_model)
        self.set_sam_image()
        self.sam_status = True
        self.mmseg_status = False

        
    def set_sam_image(self):
        image = cvtPixmapToArray(self.pixmap)
        image = image[:, :, :3]
        
        self.sam_predictor.set_image(image)


    def load_mmseg(self, config_file, checkpoint_file):
        """
        Load the mmseg model
        Args:
            config_file (str): The path to the config file.
            checkpoint_file (str): The path to the checkpoint file.
        """
        self.mmseg_model = init_model(config_file, checkpoint_file, device='cuda:0')
        self.mmseg_status = True
        self.sam_status = False

    def inference_mmseg(self, img, do_crf=True):
        """
        Inference the image with the mmseg model

        Args:
            img (np.ndarray): The image to be processed.
            do_crf (bool): Whether to apply DenseCRF.

        Returns:
            mask (np.ndarray): The processed mask.

        """
        # filter image size too small or too large
        if img.shape[0] < 50 or img.shape[1] < 50 :
            print(f"too small")
            return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        elif img.shape[0] > 1000 or img.shape[1] > 1000 :
            print(f"too large")
            return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        img = self.cvtRGBATORGB(img)

        result = inference_model(self.mmseg_model, img)

        mask = result.pred_sem_seg.data.cpu().numpy()
        mask = np.squeeze(mask)

        if do_crf:
            crf = self.applyDenseCRF(img, mask)
            skel = skeletonize(mask)

            crf[skel] = 1
            mask = crf

        mask = skimage.morphology.binary_closing(mask, skimage.morphology.square(3))

        return mask
    
    def load_mmdet (self, config_file, checkpoint_file):
        """
        Load the mmdet model
        Args:
            config_file (str): The path to the config file.
            checkpoint_file (str): The path to the checkpoint file.
        """
        self.mmdet_model = init_detector(config_file, checkpoint_file, device='cpu')


    def inference_mmdet(self, img, model):
        """
        Inference the image with the mmseg model

        Args:
            img (np.ndarray): The image to be processed.
            model: mmdetection model

        Returns:
            bboxes (np.ndarray): The processed bboxes.
            scores (np.ndarray): The processed scores.
        """

        img = self.cvtRGBATORGB(img)
        
        result = inference_detector(model, img)

        bboxes = result.pred_instances.bboxes.cpu().numpy()
        bboxes = np.squeeze(bboxes)

        scores = result.pred_instances.scores.cpu().numpy()
        scores = np.squeeze(scores)


        return bboxes, scores


    @staticmethod
    def applyDenseCRF(img, label, num_iter=3):
        """
        Apply DenseCRF to the image and label

        Args:
            img (np.ndarray): The image to be processed.
            label (np.ndarray): The label to be processed.
            num_iter (int): The number of iterations.

        Returns:
            label (np.ndarray): The processed label.
        """
        num_labels = np.max(label) + 1

        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], num_labels)

        U = unary_from_labels(label, num_labels, gt_prob=0.7, zero_unsure=False)

        d.setUnaryEnergy(U)

        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(50, 50), schan=(13, 13, 13),
                                            img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        Q = d.inference(num_iter)

        MAP = np.argmax(Q, axis=0)

        return MAP.reshape((img.shape[0], img.shape[1]))

    
    @staticmethod
    def cvtRGBATORGB(img):
        """Convert a RGBA image to a RGB image
        Args:
            img (np.ndarray): The image to be converted.

        Returns:
            img (np.ndarray): The converted image.
        
        """
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return img
    

    


    