from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMainWindow, QGraphicsScene

from .ui_main import Ui_MainWindow
from .ui_sam_window import Ui_SAMWindow
from .ui_functions import UIFunctions
from .app_settings import Settings

from mmseg.apis import init_model, inference_model

from skimage.morphology import skeletonize

import numpy as np

import skimage.morphology

from .utils import cvtPixmapToArray

from segment_anything import sam_model_registry, SamPredictor

from submodules.GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate

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
        self.mmseg_config = 'dnn/configs/promptModel.py'
        self.mmseg_checkpoint = 'dnn/checkpoints/promptModel.pth'
        # Segment Anything
        self.sam_checkpoint = 'dnn/checkpoints/sam_vit_h_4b8939.pth'

        #################### 
        # Object Detection #
        ####################
        
        # GroundingDINO
        self.groundingDino_config = 'dnn/configs/GroundingDINO_SwinB_cfg.py'
        self.groundingDino_checkpoint = 'dnn/checkpoints/groundingdino_swinb_cogcoor.pth'

        ##############
        # Attributes #
        ##############
        
        self.scale = 1.0

    def load_groundingDino(self, config, checkpoint):
        self.groundingDino_model = load_model(config, checkpoint)
    
    def inference_groundingDino(self, model, image, caption, box_threshold, text_threshold, device="cuda"):

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=device
            )
        
        return boxes, logits, phrases


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
        # if img.shape[0] < 50 or img.shape[1] < 50 :
        #     print(f"too small")
        #     return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        # elif img.shape[0] > 1000 or img.shape[1] > 1000 :
        #     print(f"too large")
        #     return np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        img = self.cvtRGBATORGB(img)

        result = inference_model(self.mmseg_model, img)

        mask = result.pred_sem_seg.data.cpu().numpy()
        mask = np.squeeze(mask)

        logits = result.seg_logits.data.cpu().numpy()
        
        
        back = mask == 0
        joint = mask == 1
        gap = mask == 2

        print(f"bf: {mask}")
        print(f"logit: {logits}")
        

        # if do_crf:
        #     crf = self.applyDenseCRF(img, mask)
        #     skel = skeletonize(mask)

        #     crf[skel] = 1
        #     mask = crf

        back = skimage.morphology.binary_closing(back, skimage.morphology.square(3))
        joint = skimage.morphology.binary_closing(joint, skimage.morphology.square(3))
        gap = skimage.morphology.binary_closing(gap, skimage.morphology.square(3))
        

        print(f"af: {mask}")

        return back, joint, gap, logits
    
    
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
    

    


    