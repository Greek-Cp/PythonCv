from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QLabel, QMessageBox, QWidget, QGridLayout
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import sys

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.setStyleSheet("background-color: lightgrey;")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200,800)

        grid_layout = QGridLayout()

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setLayout(grid_layout)

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = self.menuBar()

        self.menuFile = self.menubar.addMenu("File")

        self.actionOpen = self.menuFile.addAction("Open Image")
        self.actionOpen.triggered.connect(self.open_image)

        self.actionSave = self.menuFile.addAction("Save Image")
        self.actionSave.triggered.connect(self.save_image)

        self.beforeImageView = QLabel(self.centralwidget)
        self.beforeImageView.setStyleSheet("border: 2px solid black;")
        grid_layout.addWidget(self.beforeImageView, 0, 0)

        self.afterImageView = QLabel(self.centralwidget)
        self.afterImageView.setStyleSheet("border: 2px solid black;")
        grid_layout.addWidget(self.afterImageView, 0, 1)
        
        self.menuThreshold = self.menubar.addMenu("Threshold")

        self.actionBinary = self.menuThreshold.addAction("Binary")
        self.actionBinary.triggered.connect(lambda: self.threshold_image("BINARY"))

        self.actionBinary_Inv = self.menuThreshold.addAction("Binary Inverted")
        self.actionBinary_Inv.triggered.connect(lambda: self.threshold_image("BINARY_INV"))

        self.actionTrunc = self.menuThreshold.addAction("Truncate")
        self.actionTrunc.triggered.connect(lambda: self.threshold_image("TRUNC"))

        self.actionToZero = self.menuThreshold.addAction("To Zero")
        self.actionToZero.triggered.connect(lambda: self.threshold_image("TOZERO"))

        self.actionToZero_Inv = self.menuThreshold.addAction("To Zero Inverted")
        self.actionToZero_Inv.triggered.connect(lambda: self.threshold_image("TOZERO_INV"))

        self.actionZeroBasedBinary = self.menuThreshold.addAction("Zero-Based Binary")
        self.actionZeroBasedBinary.triggered.connect(lambda: self.threshold_image("ZERO_BASED_BINARY"))

        self.actionZeroBasedBinary_Inv = self.menuThreshold.addAction("Zero-Based Binary Inverted")
        self.actionZeroBasedBinary_Inv.triggered.connect(lambda: self.threshold_image("ZERO_BASED_BINARY_INV"))
            
        
    def threshold_image(self, method):
        if hasattr(self, 'image'):
            # Convert image to grayscale
            r, g, b = self.image[:,:,0], self.image[:,:,1], self.image[:,:,2]
            gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
            gray = gray.astype(np.uint8)
            
            output = np.zeros_like(gray)
            
            # Apply thresholding
            for i in range(gray.shape[0]):
                for j in range(gray.shape[1]):
                    pixel = gray[i, j]
                    if method == "BINARY":
                        output[i, j] = 255 if pixel >= 128 else 0
                    elif method == "BINARY_INV":
                        output[i, j] = 0 if pixel >= 128 else 255
                    elif method == "TRUNC":
                        output[i, j] = min(pixel, 128)
                    elif method == "TOZERO":
                        output[i, j] = pixel if pixel >= 128 else 0
                    elif method == "TOZERO_INV":
                        output[i, j] = pixel if pixel < 128 else 0
                    elif method == "ZERO_BASED_BINARY":
                        output[i, j] = pixel if pixel >= 128 else 0
                    elif method == "ZERO_BASED_BINARY_INV":
                        output[i, j] = pixel if pixel < 128 else 0
            
            # Convert back to 3-channel image
            self.processed_image = np.stack([output, output, output], axis=2)
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
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Image")
        if filepath:
            cv2.imwrite(filepath, self.processed_image)

    def show_image(self, image, pos='before'):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(image)

        if pos == 'before':
            self.beforeImageView.setPixmap(pixmap)
        else:
            self.afterImageView.setPixmap(pixmap)

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ui_MainWindow()
    window.show()
    sys.exit(app.exec_())
