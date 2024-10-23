import os

os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = pow(2,40).__str__()

import cv2
import numpy as np

from PySide6.QtGui import  QPixmap
from PySide6.QtWidgets import (QMainWindow,
                               QFileSystemModel,
                               QGraphicsScene,
                               QFileDialog) 

from .ui_main import Ui_MainWindow
from .dnn_functions import DNNFunctions

from .utils import *
from .utils_img import (getScaledPoint, readImageToPixmap)

class ImageFunctions(DNNFunctions):
    def __init__(self):
        DNNFunctions.__init__(self)

        if not hasattr(self, 'ui'):
            QMainWindow.__init__(self)
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)

        global mainWidgets
        mainWidgets = self.ui
            
        self.fileModel = QFileSystemModel()
        self.alpha = 80
        self.scale = 1
        self.oldPos = None
        
        """
        Signals
        """
        mainWidgets.treeView.clicked.connect(self.openImage)
        
        mainWidgets.mainImageViewer.mousePressEvent = self._mousePressPoint
        
        mainWidgets.addImageButton.clicked.connect(self.addNewImage)
        mainWidgets.deleteImageButton.clicked.connect(self.deleteImage)

        
        self.sam_mode = False
        
        """
        Variables
        """
        # self.AltKey = False
        
        self.ControlKey = False
        
        self.input_point_list = []
        self.input_label_list = []

        self.sam_y_idx = []
        self.sam_x_idx = []

    def deleteImage(self, event):
        self.currentIndex = mainWidgets.treeView.currentIndex().data(QFileSystemModel.FilePathRole)
        self.imgFolderPath, _ = os.path.split(self.currentIndex)

        img_path = self.currentIndex
        label_path = self.convertImagePathToLabelPath(img_path)
        os.remove(img_path)
        os.remove(label_path)
        
        mainWidgets.treeView.model().removeRow(mainWidgets.treeView.currentIndex().row(), mainWidgets.treeView.currentIndex().parent())


    @staticmethod
    def convertImagePathToLabelPath(img_path):
        img_label_folder = img_path.replace('/leftImg8bit/', '/gtFine/')
        img_label_folder = img_label_folder.replace( '_leftImg8bit.png', '_gtFine_labelIds.png')
        return img_label_folder
    
    def addNewImage(self, event):
        
        self.currentIndex = mainWidgets.treeView.currentIndex().data(QFileSystemModel.FilePathRole)
        
        self.imgFolderPath, filename = os.path.split(self.currentIndex)

        if 'leftImg8bit.png' not in filename:
            self.imgFolderPath = self.currentIndex
        
        img_save_folder = self.imgFolderPath
        img_save_folder = img_save_folder.replace( '_leftImg8bit.png', '')  
    
        img_label_folder = img_save_folder.replace('/leftImg8bit/', '/gtFine/')
        img_label_folder = img_label_folder.replace( '_leftImg8bit.png', '')

        readFilePath = QFileDialog.getOpenFileNames(
                caption="Add images to current working directory", filter="Images (*.png *.jpg *.tiff)"
                )
        images = readFilePath[0]

        for img in images:
                
            temp_img = cv2.imdecode(np.fromfile(img, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            img_filename = os.path.basename(img) # -> basename is file name
            img_filename = img_filename.replace(' ', '')
            img_filename = img_filename.replace('.jpg', '.png')
            img_filename = img_filename.replace('.JPG', '.png')
            img_filename = img_filename.replace('.tiff', '.png')
            img_filename = img_filename.replace('.png', '_leftImg8bit.png')

            img_gt_filename = img_filename.replace( '_leftImg8bit.png', '_gtFine_labelIds.png')
            gt = np.zeros((temp_img.shape[0], temp_img.shape[1]), dtype=np.uint8)

            _, org_img = cv2.imencode(".png", temp_img)
            org_img.tofile(os.path.join(img_save_folder, img_filename))

            _, gt_img = cv2.imencode(".png", gt)
            gt_img.tofile(os.path.join(img_label_folder, img_gt_filename))

        # self.resetTreeView(refreshIndex=True)
        
    def resetTreeView(self, refreshIndex = False):
        """Reset the tree view
        Args:
            refreshIndex (bool, optional): Refresh the current index. Defaults to False.
        """
        
        mainWidgets.treeView.reset()
        self.fileModel = QFileSystemModel()
        _imgFolderPath = self.imgFolderPath.replace('/train', '')
        _imgFolderPath = _imgFolderPath.replace('/test', '')
        _imgFolderPath = _imgFolderPath.replace('/val', '')

        self.fileModel.setRootPath(_imgFolderPath)
        
        mainWidgets.treeView.setModel(self.fileModel)
        mainWidgets.treeView.setRootIndex(self.fileModel.index(_imgFolderPath))

        if refreshIndex:
            mainWidgets.treeView.setCurrentIndex(self.fileModel.index(self.currentIndex))

    
    def openImage(self, index):
        self.imgPath = self.fileModel.filePath(index)

        if os.path.isdir(self.imgPath):
            print(f"folder")
        
        elif os.path.isfile(self.imgPath):
            
            self.labelPath = self.imgPath.replace('/leftImg8bit/', '/gtFine/')
            self.labelPath = self.labelPath.replace( '_leftImg8bit.png', '_gtFine_labelIds.png')
            self.pixmap = readImageToPixmap(self.imgPath)        
            
            self.label = imread(self.labelPath, checkImg=False)
            
            self.colormap = convertLabelToColorMap(self.label, self.label_palette, self.alpha)
            self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))
            self.scene = QGraphicsScene()
            self.pixmap_item = self.scene.addPixmap(self.pixmap)

            self.color_pixmap_item = self.scene.addPixmap(self.color_pixmap)

            mainWidgets.mainImageViewer.setScene(self.scene)

            self.scale = mainWidgets.scrollAreaImage.height() / self.label.shape[0]
            mainWidgets.mainImageViewer.setFixedSize(self.scale * self.color_pixmap.size())
            mainWidgets.mainImageViewer.fitInView(self.pixmap_item)

    def _mousePressPoint(self, event):
        
        """
        Get the brush class and the point where the mouse is pressed
        """

        # Get the brush class
        self.brush_class = mainWidgets.classList.currentRow()
        
        event_global = mainWidgets.mainImageViewer.mapFromGlobal(event.globalPos())
        
        x, y = getScaledPoint(event_global, self.scale)
        
        self.x = x
        self.y = y

        self.fixed_x = x
        self.fixed_y = y 

    def updateColorMap(self):
        """
        Update the color map
        """
        self.colormap = convertLabelToColorMap(self.label, self.label_palette, self.alpha)
        self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))
        self.color_pixmap_item.setPixmap(QPixmap())
        self.color_pixmap_item.setPixmap(self.color_pixmap)


    def removeAllLabel(self):
        self.label = np.zeros((self.label.shape[0], self.label.shape[1]), dtype=np.uint8)
        
        self.colormap = convertLabelToColorMap(self.label, self.label_palette, self.alpha)
        self.color_pixmap = QPixmap(cvtArrayToQImage(self.colormap))

        self.color_pixmap_item.setPixmap(QPixmap())
        self.color_pixmap_item.setPixmap(self.color_pixmap)


    