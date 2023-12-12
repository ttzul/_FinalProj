# run the UI file through pyuic5

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
import matting
from ui import Ui_Dialog
import cv2
import numpy as np

class MyDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self):
        super().__init__()
        self.fg_image = None
        self.trimap = None
        self.bg_image = None

        self.setupUi(self)
        self.pushButton.clicked.connect(self.select_foreground)
        self.pushButton_2.clicked.connect(self.change_fg_to_gray)    

    def change_fg_to_gray(self):
        print("Change Foreground to Gray button clicked")
        if self.fg_image is None:
            print("Foreground image is not selected")
            return
        fg_gray = cv2.cvtColor(self.fg_image, cv2.COLOR_BGR2GRAY)
        fg_gray = cv2.cvtColor(fg_gray, cv2.COLOR_GRAY2BGR)
        self.show_image(fg_gray, self.graphicsView)

    def qimg2np(self, qimg):
        qimg = qimg.convertToFormat(QtGui.QImage.Format.Format_RGB888)
        width = qimg.width()
        height = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.array(ptr).reshape(height, width, 3)
        return arr

    def np2qimg(self, arr):
        height, width, channel = arr.shape
        bytesPerLine = 3 * width
        qimg = QtGui.QImage(arr.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        return qimg

    def select_image(self):
        print("Select Foreground button clicked")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if fileName:
            pixmap = QtGui.QPixmap(fileName)
            if not pixmap.isNull():
                image = pixmap.toImage()
                image = self.qimg2np(image)
                return image

    def show_image(self, image, graphicsView):
        pixmap = QtGui.QPixmap.fromImage(self.np2qimg(image))
        pixmap.scaled(graphicsView.size(), QtCore.Qt.KeepAspectRatio)
        scene = QtWidgets.QGraphicsScene()
        scene.addPixmap(pixmap)
        graphicsView.setScene(scene)
        graphicsView.fitInView(scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    
    def select_foreground(self):
        self.fg_image = self.select_image()
        if self.fg_image is None:
            return
        self.show_image(self.fg_image, self.graphicsView)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    sys.exit(app.exec_())
