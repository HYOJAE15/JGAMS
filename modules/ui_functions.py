# ///////////////////////////////////////////////////////////////
#
# BY: WANDERSON M.PIMENTA
# PROJECT MADE WITH: Qt Designer and PySide6
# V: 1.0.0
#
# This project can be used freely for all uses, as long as they maintain the
# respective credits only in the Python scripts, any information in the visual
# interface (GUI) can be modified without any implication.
#
# There are limitations on Qt licenses if you want to use your products
# commercially, I recommend reading them on the official website:
# https://doc.qt.io/qtforpython/licenses.html
#
# ///////////////////////////////////////////////////////////////

# GLOBALS
# ///////////////////////////////////////////////////////////////

from widgets import CustomGrip

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QColor
from PySide6.QtWidgets import (
    QPushButton, QSizeGrip, QGraphicsDropShadowEffect, QSizePolicy
)
from PySide6.QtCore import (
    QPropertyAnimation, QEasingCurve, QParallelAnimationGroup, QTimer, QEvent
)



GLOBAL_STATE = False
GLOBAL_TITLE_BAR = True

class UIFunctions(object):
    # MAXIMIZE/RESTORE
    # ///////////////////////////////////////////////////////////////
    def maximize_restore(self):
        global GLOBAL_STATE
        status = GLOBAL_STATE
        if status == False:
            self.showMaximized()
            GLOBAL_STATE = True
            self.ui.appMargins.setContentsMargins(0, 0, 0, 0)
            self.ui.maximizeRestoreAppBtn.setToolTip("Restore")
            self.ui.maximizeRestoreAppBtn.setIcon(QIcon(u":/icons/images/icons/icon_restore.png"))
            self.ui.frame_size_grip.hide()
            self.left_grip.hide()
            self.right_grip.hide()
            self.top_grip.hide()
            self.bottom_grip.hide()
        else:
            GLOBAL_STATE = False
            self.showNormal()
            self.resize(self.width()+1, self.height()+1)
            self.ui.appMargins.setContentsMargins(10, 10, 10, 10)
            self.ui.maximizeRestoreAppBtn.setToolTip("Maximize")
            self.ui.maximizeRestoreAppBtn.setIcon(QIcon(u":/icons/images/icons/icon_maximize.png"))
            self.ui.frame_size_grip.show()
            self.left_grip.show()
            self.right_grip.show()
            self.top_grip.show()
            self.bottom_grip.show()

    # RETURN STATUS
    # ///////////////////////////////////////////////////////////////
    def returStatus(self):
        return GLOBAL_STATE

    # SET STATUS
    # ///////////////////////////////////////////////////////////////
    def setStatus(self, status):
        global GLOBAL_STATE
        GLOBAL_STATE = status

    # TOGGLE MENU
    # ///////////////////////////////////////////////////////////////
    def toggleMenu(self, enable):
        if enable:
            # GET WIDTH
            width = self.ui.menuBox.width()
            maxExtend = self.settings.MENU_WIDTH
            standard = 60

            # SET MAX WIDTH
            if width == 60:
                widthExtended = maxExtend
            else:
                widthExtended = standard

            # ANIMATION
            self.animation = QPropertyAnimation(self.ui.menuBox, b"minimumWidth")
            self.animation.setDuration(self.settings.TIME_ANIMATION)
            self.animation.setStartValue(width)
            self.animation.setEndValue(widthExtended)
            self.animation.setEasingCurve(QEasingCurve.InOutQuart)
            self.animation.start()


    def toggleLeftBoxAnimation(self, _left_box, to_standard=False):
        global left_box
        # GET WIDTH
        left_box = _left_box
        width = left_box.width()
        maxExtend = self.settings.LEFT_BOX_WIDTH
        standard = 0

        # SET MAX WIDTH
        if width == 0:
            widthExtended = maxExtend

            left_box.grip.setFixedWidth(10)
            left_box.grip.setGeometry(maxExtend - 10, 0, 10, left_box.height() - 20)
            left_box.grip.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint)
            left_box.grip.show()
            left_box.resizeEvent = lambda event: left_box.grip.setFixedHeight(100000)

            def resizeLeftBoxAnimation(event):
                global left_box

                width = left_box.width()
                event_local = left_box.grip.mapFromGlobal(event.globalPos())
                
                widthExtended = width + event_local.x() 
                print("width widthExtended", width, widthExtended)

                left_box.grip.setGeometry(widthExtended - 10, 0, 10, 100000)

                animation = QPropertyAnimation(left_box, b"minimumWidth")
                animation.setDuration(0.01)
                animation.setStartValue(width)
                animation.setEndValue(widthExtended)
                animation.setEasingCurve(QEasingCurve.InOutQuart)
                animation.start()

                return animation        

            left_box.grip.wi.rightgrip.mouseMoveEvent = resizeLeftBoxAnimation
        else:
            left_box.grip.hide()
            widthExtended = standard

        if to_standard: 
            left_box.grip.hide()
            widthExtended = standard

        # ANIMATION
        animation = QPropertyAnimation(left_box, b"minimumWidth")
        animation.setDuration(self.settings.TIME_ANIMATION)
        animation.setStartValue(width)
        animation.setEndValue(widthExtended)
        animation.setEasingCurve(QEasingCurve.InOutQuart)
        animation.start()

        return animation

    

    
    def toggleRightBoxAnimation(self, right_box, to_standard=False):
        
        # GET WIDTH
        width = right_box.width()
        maxExtend = self.settings.RIGHT_TOOL_BOX_WIDTH
        standard = 0

        # # GET BTN STYLE
        # style = self.ui.settingsTopBtn.styleSheet()

        # SET MAX WIDTH
        if width == 0:
            widthExtended = maxExtend
    
        else:
            widthExtended = standard
            
        if to_standard: 
            widthExtended = standard

        # ANIMATION
        animation = QPropertyAnimation(right_box, b"minimumWidth")
        animation.setDuration(self.settings.TIME_ANIMATION)
        animation.setStartValue(width)
        animation.setEndValue(widthExtended)
        animation.setEasingCurve(QEasingCurve.InOutQuart)
        animation.start()

        return animation


    # # TOGGLE LEFT BOX
    # # ///////////////////////////////////////////////////////////////
    # def toggleLeftBox(self, enable, target):
    #     if enable:
    #         # GET WIDTH
    #         width = target.width()
    #         widthRightBox = self.ui.extraRightBox.width()
    #         maxExtend = self.settings.LEFT_BOX_WIDTH
    #         color = self.settings.BTN_LEFT_BOX_COLOR
    #         standard = 0

    #         # GET BTN STYLE
    #         style = target.styleSheet()

    #         # SET MAX WIDTH
    #         if width == 0:
    #             widthExtended = maxExtend
    #             # SELECT BTN
    #             target.setStyleSheet(style + color)
    #             if widthRightBox != 0:
    #                 style = self.ui.settingsTopBtn.styleSheet()
    #                 self.ui.settingsTopBtn.setStyleSheet(style.replace(self.settings.BTN_RIGHT_BOX_COLOR, ''))
    #         else:
    #             widthExtended = standard
    #             # RESET BTN
    #             target.setStyleSheet(style.replace(color, ''))
                
    #     self.start_box_animation(target, width, widthRightBox, "left")


    # # TOGGLE RIGHT BOX
    # # ///////////////////////////////////////////////////////////////
    # def toggleRightBox(self, enable):
    #     if enable:
    #         # GET WIDTH
    #         width = self.ui.extraRightBox.width()
    #         widthLeftBox = self.ui.extraLeftBox.width()
    #         maxExtend = self.settings.RIGHT_BOX_WIDTH
    #         color = self.settings.BTN_RIGHT_BOX_COLOR
    #         standard = 0

    #         # GET BTN STYLE
    #         style = self.ui.settingsTopBtn.styleSheet()

    #         # SET MAX WIDTH
    #         if width == 0:
    #             widthExtended = maxExtend
    #             # SELECT BTN
    #             self.ui.settingsTopBtn.setStyleSheet(style + color)
    #             if widthLeftBox != 0:
    #                 style = self.ui.toggleLeftBox.styleSheet()
    #                 self.ui.toggleLeftBox.setStyleSheet(style.replace(self.settings.BTN_LEFT_BOX_COLOR, ''))
    #         else:
    #             widthExtended = standard
    #             # RESET BTN
    #             self.ui.settingsTopBtn.setStyleSheet(style.replace(color, ''))

    #         self.start_box_animation(widthLeftBox, width, "right")
    
    
    # def start_box_animation(self, left_box, left_box_width, right_box_width, direction):
    #     right_width = 0
    #     left_width = 0 

    #     # Check values
    #     if left_box_width == 0 and direction == "left":
    #         left_width = 240
    #     else:
    #         left_width = 0
    #     # Check values
    #     if right_box_width == 0 and direction == "right":
    #         right_width = 240
    #     else:
    #         right_width = 0       

    #     # ANIMATION LEFT BOX        
    #     self.left_box = QPropertyAnimation(left_box, b"minimumWidth")
    #     self.left_box.setDuration(self.settings.TIME_ANIMATION)
    #     self.left_box.setStartValue(left_box_width)
    #     self.left_box.setEndValue(left_width)
    #     self.left_box.setEasingCurve(QEasingCurve.InOutQuart)

    #     # ANIMATION RIGHT BOX        
    #     self.right_box = QPropertyAnimation(self.ui.extraRightBox, b"minimumWidth")
    #     self.right_box.setDuration(self.settings.TIME_ANIMATION)
    #     self.right_box.setStartValue(right_box_width)
    #     self.right_box.setEndValue(right_width)
    #     self.right_box.setEasingCurve(QEasingCurve.InOutQuart)

    #     # GROUP ANIMATION
    #     self.group = QParallelAnimationGroup()
    #     self.group.addAnimation(self.left_box)
    #     self.group.addAnimation(self.right_box)
    #     self.group.start()

    # SELECT/DESELECT MENU
    # ///////////////////////////////////////////////////////////////
    # SELECT
    def selectMenu(self, getStyle):
        select = getStyle + self.settings.MENU_SELECTED_STYLESHEET
        return select

    def selectTopMenu(self, getStyle):
        select = getStyle + self.settings.TOP_BUTTON_SELECTED_STYLESHEET
        return select

    # DESELECT
    def deselectMenu(self, getStyle):
        deselect = getStyle.replace(self.settings.MENU_SELECTED_STYLESHEET, "")
        return deselect

    def deselectTopMenu(self, getStyle):
        deselect = getStyle.replace(self.settings.TOP_BUTTON_SELECTED_STYLESHEET, "")
        return deselect

    # START SELECTION
    def selectStandardMenu(self, widget):
        for w in self.ui.topMenu.findChildren(QPushButton):
            if w.objectName() == widget:
                w.setStyleSheet(self.selectMenu(w.styleSheet()))

    # RESET SELECTION
    def resetStyle(self, widget):
        for w in self.ui.topMenu.findChildren(QPushButton):
            if w.objectName() != widget:
                w.setStyleSheet(self.deselectMenu(w.styleSheet()))

        for w in self.ui.rightButtons.findChildren(QPushButton):
            if w.objectName() != widget:
                w.setStyleSheet(self.deselectMenu(w.styleSheet()))

    # IMPORT THEMES FILES QSS/CSS
    # ///////////////////////////////////////////////////////////////
    def theme(self, file, useCustomTheme):
        if useCustomTheme:
            str = open(file, 'r').read()
            self.ui.styleSheet.setStyleSheet(str)

    # START - GUI DEFINITIONS
    # ///////////////////////////////////////////////////////////////
    def uiDefinitions(self):
        def dobleClickMaximizeRestore(event):
            # IF DOUBLE CLICK CHANGE STATUS
            if event.type() == QEvent.MouseButtonDblClick:
                QTimer.singleShot(250, lambda: self.maximize_restore())
        self.ui.titleRightInfo.mouseDoubleClickEvent = dobleClickMaximizeRestore

        if self.settings.ENABLE_CUSTOM_TITLE_BAR:
            #STANDARD TITLE BAR
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
            self.setAttribute(Qt.WA_TranslucentBackground)

            # MOVE WINDOW / MAXIMIZE / RESTORE
            def moveWindow(event):
                # IF MAXIMIZED CHANGE TO NORMAL
                if self.returStatus():
                    self.maximize_restore()
                # MOVE WINDOW
                if event.buttons() == Qt.LeftButton:
                    self.move(self.pos() + event.globalPos() - self.dragPos)
                    self.dragPos = event.globalPos()
                    event.accept()
            self.ui.titleRightInfo.mouseMoveEvent = moveWindow

            # CUSTOM GRIPS
            self.left_grip = CustomGrip(self, Qt.LeftEdge, True)
            self.right_grip = CustomGrip(self, Qt.RightEdge, True)
            self.top_grip = CustomGrip(self, Qt.TopEdge, True)
            self.bottom_grip = CustomGrip(self, Qt.BottomEdge, True)

        else:
            self.ui.appMargins.setContentsMargins(0, 0, 0, 0)
            self.ui.minimizeAppBtn.hide()
            self.ui.maximizeRestoreAppBtn.hide()
            self.ui.closeAppBtn.hide()
            self.ui.frame_size_grip.hide()

        # DROP SHADOW
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(17)
        self.shadow.setXOffset(0)
        self.shadow.setYOffset(0)
        self.shadow.setColor(QColor(0, 0, 0, 150))
        self.ui.bgApp.setGraphicsEffect(self.shadow)

        # RESIZE WINDOW
        self.sizegrip = QSizeGrip(self.ui.frame_size_grip)
        self.sizegrip.setStyleSheet("width: 20px; height: 20px; margin 0px; padding: 0px;")

        # MINIMIZE
        self.ui.minimizeAppBtn.clicked.connect(lambda: self.showMinimized())

        # MAXIMIZE/RESTORE
        self.ui.maximizeRestoreAppBtn.clicked.connect(lambda: self.maximize_restore())

        # CLOSE APPLICATION
        self.ui.closeAppBtn.clicked.connect(lambda: self.close())

    def resize_grips(self):
        if self.settings.ENABLE_CUSTOM_TITLE_BAR:
            self.left_grip.setGeometry(0, 10, 10, self.height())
            self.right_grip.setGeometry(self.width() - 10, 10, 10, self.height())
            self.top_grip.setGeometry(0, 0, self.width(), 10)
            self.bottom_grip.setGeometry(0, self.height() - 10, self.width(), 10)

    # ///////////////////////////////////////////////////////////////
    # END - GUI DEFINITIONS


    
