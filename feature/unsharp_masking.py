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
 # Create a new menu for Unsharp Masking
        self.menuUnsharpMasking = self.menubar.addMenu("Unsharp Masking")
        
        # Add sub-menu options for Unsharp Masking
        self.actionBasicUnsharp = self.menuUnsharpMasking.addAction("Basic Unsharp Mask")
        self.actionBasicUnsharp.triggered.connect(self.basic_unsharp_mask)

        self.actionGaussianUnsharp = self.menuUnsharpMasking.addAction("Gaussian Unsharp Mask")
        self.actionGaussianUnsharp.triggered.connect(self.gaussian_unsharp_mask)


        self.beforeImageView = QLabel(self.centralwidget)
        self.beforeImageView.setStyleSheet("border: 2px solid black;")
        grid_layout.addWidget(self.beforeImageView, 0, 0)

        self.afterImageView = QLabel(self.centralwidget)
        self.afterImageView.setStyleSheet("border: 2px solid black;")
        grid_layout.addWidget(self.afterImageView, 0, 1)
        self.actionLaplacianUnsharp = self.menuUnsharpMasking.addAction("Laplacian Unsharp Mask")
        self.actionLaplacianUnsharp.triggered.connect(self.laplacian_unsharp_mask)

        self.actionHighBoost = self.menuUnsharpMasking.addAction("High-Boost Filtering")
        self.actionHighBoost.triggered.connect(self.high_boost_filtering)
        
    def high_boost_filtering(self):
        # First, blur the image using a simple blur kernel
        blurred = cv2.blur(self.image, (3, 3))

        # Create a mask by subtracting the blurred image from the original image
        mask = cv2.subtract(self.image, blurred)

        # Perform the high-boost filtering using addWeighted function
        k = 0.5  # Boost factor
        self.processed_image = cv2.addWeighted(self.image, 1, mask, k, 0)

        # Display the processed image
        self.show_image(self.processed_image, 'after')


    def gaussian_unsharp_mask(self):
        # Create a Gaussian blur of the image
        gaussian_blur = cv2.GaussianBlur(self.image, (5, 5), 1.5)
        
        # Perform the unsharp masking
        self.processed_image = cv2.addWeighted(self.image, 1.5, gaussian_blur, -0.5, 0)
        
        # Display the processed image
        self.show_image(self.processed_image, 'after')



    def laplacian_unsharp_mask(self):
        laplacian_filter = np.array([[-1, -1, -1],
                                     [-1,  8, -1],
                                     [-1, -1, -1]])

        laplacian = self.convolve2D(self.image, laplacian_filter)
        self.processed_image = np.clip(self.image + laplacian, 0, 255)
        self.show_image(self.processed_image, 'after')

    def basic_unsharp_mask(self):
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])

        self.processed_image = self.convolve2D(self.image, kernel)
        self.show_image(self.processed_image, 'after')

    def convolve2D(self, img, kernel):
        # Get dimensions
        h, w, _ = img.shape
        kh, kw = kernel.shape[:2]
        
        # Calculate the padding
        pad_h = kh // 2
        pad_w = kw // 2
        
        # Create a padded copy of the input image
        img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), 'constant')
        
        # Prepare the output image
        output = np.zeros_like(img)
        
        # Perform the convolution
        for i in range(h):
            for j in range(w):
                for c in range(3):  # Assuming a 3-channel color image
                    output[i, j, c] = np.sum(kernel * img_padded[i:i+kh, j:j+kw, c])
        
        return np.uint8(np.clip(output, 0, 255))


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
