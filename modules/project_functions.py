import os 
import json

import numpy as np 

from PySide6.QtCore import Qt, QParallelAnimationGroup
from PySide6.QtGui import QColor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QTableWidgetItem, QColorDialog,
    QFileSystemModel, QListWidgetItem
) 

from .app_settings import Settings
from .ui_main import Ui_MainWindow
from .ui_project_class import Ui_ProjectClass
from .ui_project_name import Ui_ProjectName
from .ui_functions import UIFunctions


class ProjectClassWindow(QMainWindow, UIFunctions):
    def __init__(self):
        QMainWindow.__init__(self)

        self.ui = Ui_ProjectClass()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)

        self.settings = Settings()

        self.uiDefinitions()

    def resizeEvent(self, event):
        self.resize_grips()

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()


class ProjectNameWindow(QMainWindow, UIFunctions):
    def __init__(self):
        QMainWindow.__init__(self)

        self.ui = Ui_ProjectName()
        self.ui.setupUi(self)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
        # self.show()

        self.settings = Settings()

        self.uiDefinitions()

    def resizeEvent(self, event):
        self.resize_grips()

    def mousePressEvent(self, event):
        self.dragPos = event.globalPos()


class ProjectFunctions(object):
    def __init__(self):

        """
        Init UI
        """
        if not hasattr(self, 'ui'):
            QMainWindow.__init__(self)
            self.ui = Ui_MainWindow()
            self.ui.setupUi(self)
        
        self.new_project_info = {}

        global mainWidgets
        mainWidgets = self.ui 

        """
        Button connections
        """
        # ProjectNameWindow
        self.ProjectName = ProjectNameWindow()
        UIProjectName = self.ProjectName.ui
        UIProjectName.selectFolderButton.clicked.connect(self.selectProjectFolder)
        UIProjectName.nextButton.clicked.connect(self.openProjectClassDialogue)
        UIProjectName.cancelButton.clicked.connect(self.ProjectName.close)

        # ProjectClassWindow
        self.projectClass = ProjectClassWindow()
        self.projectClass.ui.okButton.clicked.connect(self.createProjectHeader)
        self.projectClass.ui.cancelButton.clicked.connect(self.projectClass.close)
        self.projectClass.ui.addButton.clicked.connect(self.addRow)
        self.projectClass.ui.removeButton.clicked.connect(self.deleteRow)
        self.projectClass.ui.tableWidget.clicked.connect(self.eventTable)
        self.projectClass.ui.tableWidget.doubleClicked.connect(self.eventTable)

        # Main Window
        mainWidgets.btn_add_new_project.clicked.connect(
            lambda: self.ProjectName.show()
            )
        mainWidgets.btn_open_project.clicked.connect(self.openExistingProject)

        # Class List 
        mainWidgets.classList.itemClicked.connect(self.getListWidgetIndex)
        # mainWidgets.classList.itemSelectionChanged.connect(self.getListWidgetIndex_autolabel)
        
    def selectProjectFolder(self):
        self.project_folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        self.ProjectName.ui.lineFolder.setText(self.project_folder)
        self.ProjectName.close()
        self.ProjectName.show()
    
    def openProjectClassDialogue(self):

        self.project_name = self.ProjectName.ui.datasetComboBox.currentText()
        self.new_project_info['project_name'] = self.project_name
        self.new_project_info['project_folder'] = self.project_folder
        self.ProjectName.close()
        self.projectClass.show()
    
    def eventTable(self, item):
        if item.column() != 0: 
            color = QColorDialog.getColor()
            
            self.projectClass.ui.tableWidget.item(item.row(), 1).setText(f'[{color.red()}, {color.green()}, {color.blue()}]')

            self.projectClass.ui.tableWidget.item(item.row(), 2).setForeground(QColor(color.red(), color.green(), color.blue()))

    def addRow(self):
        rowPosition = self.projectClass.ui.tableWidget.rowCount()
        self.projectClass.ui.tableWidget.insertRow(rowPosition) #insert new row
        
        for i in range(0, 3):

            item = QTableWidgetItem()
            if i == 0 : 
                item.setText(f"Class{rowPosition}")
            elif i == 1 : 
                item.setText(f"[{rowPosition}, {rowPosition}, {rowPosition}]")
            elif i == 2 : 
                item.setText(f"Color Preview")

            self.projectClass.ui.tableWidget.setItem(rowPosition, i, item)
            self.projectClass.ui.tableWidget.item(rowPosition, i).setTextAlignment(Qt.AlignCenter)
        
        self.projectClass.ui.tableWidget.item(rowPosition, 1).setBackground(QColor(rowPosition, rowPosition, rowPosition))

    def deleteRow(self):
        self.projectClass.ui.tableWidget.removeRow(self.projectClass.ui.tableWidget.currentRow())

    def createProjectHeader(self):
        createProjectFile_name = self.new_project_info['project_name'] + ".hdr"

        path = self.new_project_info['project_folder']
        n_row = self.projectClass.ui.tableWidget.rowCount()

        self.new_project_info['categories'] = []

        for i in range(n_row):
            self.new_project_info['categories'].append(
                [
                    self.projectClass.ui.tableWidget.item(i, 0).text(),
                    self.projectClass.ui.tableWidget.item(i, 1).text()
                ]
                )
            
        with open(os.path.join(path, createProjectFile_name), 'w') as fp:
            json.dump(self.new_project_info, fp)

        self.projectClass.close()


    def openExistingProject(self):

        btn = self.ui.imageButton
        btnName = btn.objectName()

        self.resetStyle(btnName) # RESET ANOTHERS BUTTONS SELECTED
        btn.setStyleSheet(self.selectMenu(btn.styleSheet())) # SELECT MENU

        self.ui.stackedWidget.setCurrentWidget(self.ui.imagePage) # SET PAGE

        readFilePath = QFileDialog.getOpenFileName(
            caption="Select Project File", filter="*.hdr"
            )
        hdr_path = readFilePath[0]
        
        folderPath = os.path.dirname(hdr_path)
        cityscapeDataset_folderPath = os.path.join(folderPath, "leftImg8bit")
            # openFolderPath 를 None 으로 받고 treeView 에서 선택한 파일 또는 폴더 주소를 받는다.
        self.openFolderPath = None
        # self.fileNameLabel.setText(cityscapeDataset_folderPath)
        
        self.fileModel = QFileSystemModel()
        self.fileModel.setRootPath(os.path.join(folderPath, 'leftImg8bit'))
        
        self.ui.treeView.setModel(self.fileModel)
        self.ui.treeView.setRootIndex(self.fileModel.index(os.path.join(folderPath, 'leftImg8bit')))

        with open(hdr_path) as f:
            hdr = json.load(f)

        self.ui.classList.clear()

        self.label_palette = []

        for idx, cat in enumerate(hdr['categories']):
            name, color = cat[0], cat[1]
            color = json.loads(color)
            self.ui.classList.addItem(name)
            iconPixmap = QPixmap(20, 20)
            iconPixmap.fill(QColor(color[0], color[1], color[2]))
            self.ui.classList.item(idx).setIcon(QIcon(iconPixmap))
            self.label_palette.append(color)
            self.ui.classList.item(idx).setCheckState(Qt.Checked)

        self.label_palette = np.array(self.label_palette)

        self.openImageMenu()

        mainWidgets.classList.setCurrentRow(0)
        
    def getListWidgetIndex (self):
        self.brush_class = mainWidgets.classList.currentRow()
        
    