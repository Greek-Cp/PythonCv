from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QLabel, QMessageBox, QWidget, QGridLayout, QSlider, QVBoxLayout, QMenu
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import sys

class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.image = None
        self.processed_image = None
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

        self.menuWarna = self.menubar.addMenu("Warna")
        self.actionRGB = self.menuWarna.addAction("RGB")
        self.actionRGB.triggered.connect(self.apply_rgb)
        
        self.actionRGB2HSV = self.menuWarna.addAction("RGB to HSV")
        self.actionRGB2HSV.triggered.connect(self.apply_rgb_to_hsv)
        
        self.actionRGB2YCrCb = self.menuWarna.addAction("RGB to YCrCb")
        self.actionRGB2YCrCb.triggered.connect(self.apply_rgb_to_ycrcb)

        self.beforeImageView = QLabel(self.centralwidget)
        self.beforeImageView.setStyleSheet("border: 2px solid black;")
        grid_layout.addWidget(self.beforeImageView, 0, 0)

        self.afterImageView = QLabel(self.centralwidget)
        self.afterImageView.setStyleSheet("border: 2px solid black;")
        grid_layout.addWidget(self.afterImageView, 0, 1)

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

    def apply_rgb(self):
        self.processed_image = self.image.copy()
        self.show_image(self.processed_image, 'after')

    def apply_rgb_to_hsv(self):
        r, g, b = self.image[:,:,0], self.image[:,:,1], self.image[:,:,2]
        r, g, b = r / 255.0, g / 255.0, b / 255.0

        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        delta = max_val - min_val

        h = np.zeros_like(r)
        s = np.zeros_like(r)
        v = max_val

        # Calculate Hue
        h[max_val == r] = 60.0 * (((g[max_val == r] - b[max_val == r]) / delta[max_val == r]) % 6)
        h[max_val == g] = 60.0 * (((b[max_val == g] - r[max_val == g]) / delta[max_val == g]) + 2)
        h[max_val == b] = 60.0 * (((r[max_val == b] - g[max_val == b]) / delta[max_val == b]) + 4)
        h[delta == 0] = 0

        # Calculate Saturation
        s[max_val != 0] = (delta[max_val != 0] / max_val[max_val != 0])
        s[max_val == 0] = 0

        # Scale h, s, v to 0-255
        h = (h / 360 * 255).astype(np.uint8)
        s = (s * 255).astype(np.uint8)
        v = (v * 255).astype(np.uint8)

        self.processed_image = np.stack([h, s, v], axis=2)
        self.show_image(self.processed_image, 'after')

    def apply_rgb_to_ycrcb(self):
        r, g, b = cv2.split(self.image)
        r, g, b = r / 255.0, g / 255.0, b / 255.0  # Normalize to [0, 1]

        # Compute Y, Cr, and Cb components based on standard equations
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
        cb = 128 - 0.168736 * r - 0.331364 * g + 0.5 * b

        # Scale back to 0-255
        y = (y * 255).astype(np.uint8)
        cr = (cr * 255).astype(np.uint8)
        cb = (cb * 255).astype(np.uint8)

        self.processed_image = cv2.merge([y, cr, cb])
        self.show_image(self.processed_image, 'after')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ui_MainWindow()
    window.show()
    sys.exit(app.exec_())
