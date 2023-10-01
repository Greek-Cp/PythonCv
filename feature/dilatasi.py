from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QLabel, QGridLayout,QInputDialog ,QRubberBand 
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QRect, QPoint
import sys
import cv2
import numpy as np

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        dark_stylesheet = '''
        QWidget {
            background-color: #1F1F1F;
            color: #FFFFFF;
        }
        QLabel {
            border: 2px solid #FFFFFF;
            border-radius: 4px;
        }
        QMenu::item:selected {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #C147E9, stop:1 #8823D5);
            color: #FFFFFF;
        }
        QMenuBar::item:selected {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #C147E9, stop:1 #8823D5);
            color: #FFFFFF;
            border-radius: 8px;
            border: 1px solid #FFFFFF;
        }
        QMenuBar::item {
            padding: 5px 10px;
        }
        '''
        self.setStyleSheet(dark_stylesheet)
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 800)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        MainWindow.setCentralWidget(self.centralwidget)

        layout = QGridLayout()
        self.centralwidget.setLayout(layout)

        self.menubar = QtWidgets.QMenuBar(MainWindow)
        MainWindow.setMenuBar(self.menubar)

        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setTitle("File")
        
        self.menuDilasi = QtWidgets.QMenu(self.menubar)
        self.menuDilasi.setTitle("Dilasi")

        self.actionOpen = QtWidgets.QAction("Open Image", MainWindow)
        self.actionOpen.triggered.connect(self.open_image)

        self.actionSave = QtWidgets.QAction("Save Image", MainWindow)
        self.actionSave.triggered.connect(self.save_image)

        self.actionSquare3 = QtWidgets.QAction("Square 3", MainWindow)
        self.actionSquare3.triggered.connect(lambda: self.dilate_image((3, 3)))

        self.actionSquare5 = QtWidgets.QAction("Square 5", MainWindow)
        self.actionSquare5.triggered.connect(lambda: self.dilate_image((5, 5)))

        self.actionCross3 = QtWidgets.QAction("Cross 3", MainWindow)
        self.actionCross3.triggered.connect(lambda: self.dilate_image((3, 3), shape=cv2.MORPH_CROSS))

        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)

        self.menuDilasi.addAction(self.actionSquare3)
        self.menuDilasi.addAction(self.actionSquare5)
        self.menuDilasi.addAction(self.actionCross3)

        self.menubar.addMenu(self.menuFile)
        self.menubar.addMenu(self.menuDilasi)
        
        self.menuScaling = QtWidgets.QMenu(self.menubar)
        self.menuScaling.setTitle("Scaling")
        
        self.actionUniform = QtWidgets.QAction("Uniform", MainWindow)
        self.actionUniform.triggered.connect(lambda: self.scale_image('uniform'))

        self.actionNonUniform = QtWidgets.QAction("Non Uniform", MainWindow)
        self.actionNonUniform.triggered.connect(lambda: self.scale_image('non-uniform'))

        self.menuScaling.addAction(self.actionUniform)
        self.menuScaling.addAction(self.actionNonUniform)
        
        self.menubar.addMenu(self.menuFile)
        self.menubar.addMenu(self.menuDilasi)
        self.menubar.addMenu(self.menuScaling)

        self.beforeImageView = QLabel(self.centralwidget)
        layout.addWidget(self.beforeImageView, 0, 0)

        self.afterImageView = QLabel(self.centralwidget)
        layout.addWidget(self.afterImageView, 0, 1)
        
        
    def scale_image(self, method):
        if hasattr(self, 'image'):
            factor, ok = QInputDialog.getDouble(self, "Input", "Enter scale factor:", 1.5, 0.1, 10, 2)
            if ok:
                if method == 'uniform':
                    self.processed_image = cv2.resize(self.image, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
                elif method == 'non-uniform':
                    factor_y, ok = QInputDialog.getDouble(self, "Input", "Enter Y scale factor:", 1, 0.1, 10, 2)
                    if ok:
                        self.processed_image = cv2.resize(self.image, None, fx=factor, fy=factor_y, interpolation=cv2.INTER_LINEAR)
                self.show_image(self.processed_image, 'after')
    def open_image(self):
        options = QFileDialog.Options()
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)", options=options)
        if filepath:
            self.image = cv2.imread(filepath)
            h, w, _ = self.image.shape
            # Ubah ukuran jendela sesuai dengan ukuran gambar
            self.resize(w ,h) # Penyesuaian dilakukan untuk menampung dua gambar dan beberapa padding
            self.show_image(self.image, 'before')


    def save_image(self):
        if hasattr(self, 'processed_image'):
            options = QFileDialog.Options()
            filepath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "Images (*.png *.jpg *.jpeg)", options=options)
            if filepath:
                cv2.imwrite(filepath, self.processed_image)

    def show_image(self, image, position='before'):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_qt_format.scaled(512, 512, QtCore.Qt.KeepAspectRatio)
        pixmap = QPixmap.fromImage(p)
        if position == 'before':
            self.beforeImageView.setPixmap(pixmap)
        else:
            self.afterImageView.setPixmap(pixmap)

    def dilate_image(self, kernel_size, shape=cv2.MORPH_RECT):
        kernel = cv2.getStructuringElement(shape, kernel_size)
        self.processed_image = cv2.dilate(self.image, kernel, iterations=1)
        self.show_image(self.processed_image, 'after')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Ui_MainWindow()
    window.show()
    sys.exit(app.exec_())
