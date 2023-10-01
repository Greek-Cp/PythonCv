from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QLabel, QMessageBox, QWidget, QGridLayout, QMenu, QAction
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

        self.menuSegmentasiCitra = self.menubar.addMenu("Segmentasi Citra")

        # Add submenus and actions for different segmentation techniques
        self.thresholding_menu = QMenu("Thresholding", self)
        self.watershed_menu = QMenu("Manual Watershed", self)
        self.kmeans_menu = QMenu("Manual K-Means", self)

        # Add actions for thresholding
        self.actionBinaryThreshold = QAction("Binary Threshold", self)
        self.actionBinaryThreshold.triggered.connect(self.binary_threshold)
        self.thresholding_menu.addAction(self.actionBinaryThreshold)

        self.actionOtsuThreshold = QAction("Otsu Threshold", self)
        self.actionOtsuThreshold.triggered.connect(self.otsu_threshold)
        self.thresholding_menu.addAction(self.actionOtsuThreshold)

        # Add actions for manual watershed
        self.actionManualWatershed = QAction("Manual Watershed", self)
        self.actionManualWatershed.triggered.connect(self.manual_watershed)
        self.watershed_menu.addAction(self.actionManualWatershed)

        # Add actions for manual K-means
        self.actionManualKMeans = QAction("Manual K-Means", self)
        self.actionManualKMeans.triggered.connect(self.manual_kmeans)
        self.kmeans_menu.addAction(self.actionManualKMeans)

        # Add submenus to the "Segmentasi Citra" menu
        self.menuSegmentasiCitra.addMenu(self.thresholding_menu)
        self.menuSegmentasiCitra.addMenu(self.watershed_menu)
        self.menuSegmentasiCitra.addMenu(self.kmeans_menu)
        self.menuSegmentation = self.menubar.addMenu("Segmentation")
        # Add sub-items under "Segmentation" menu
        self.actionContourSegmentation = self.menuSegmentation.addAction("Contour-Based Segmentation")
        self.actionContourSegmentation.triggered.connect(self.contour_based_segmentation)

        # ... (your existing code)
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
    def binary_threshold(self):
        if hasattr(self, 'image'):
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, self.processed_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
            self.show_image(cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR), 'after')
            QMessageBox.information(self, 'Success', 'Binary Thresholding Applied!')

    def otsu_threshold(self):
        if hasattr(self, 'image'):
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, self.processed_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.show_image(cv2.cvtColor(self.processed_image, cv2.COLOR_GRAY2BGR), 'after')
            QMessageBox.information(self, 'Success', 'Otsu Thresholding Applied!')
    def contour_based_segmentation(self):
        if hasattr(self, 'image'):
            # Convert the image to grayscale
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise and improve contour detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Detect contours in the image
            contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) > 0:
                # Create a black image to draw the contours on
                segmented_image = np.zeros_like(self.image)

                # Draw the detected contours on the segmented image
                cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 2)

                self.processed_image = segmented_image
                self.show_image(self.processed_image, 'after')
                QMessageBox.information(self, 'Success', 'Contour-Based Segmentation Applied!')
            else:
                QMessageBox.warning(self, 'Warning', 'No contours found in the image.')


    def manual_watershed(self):
        if hasattr(self, 'image'):
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel = np.ones((3, 3), np.uint8)
            sure_bg = cv2.dilate(thresh, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv2.watershed(self.image, markers)
            self.processed_image = self.image.copy()
            self.processed_image[markers == -1] = [0, 0, 255]
            self.show_image(self.processed_image, 'after')
            QMessageBox.information(self, 'Success', 'Manual Watershed Applied!')

    def manual_kmeans(self):
        if hasattr(self, 'image'):
            Z = self.image.reshape((-1, 3))
            Z = np.float32(Z)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 2  # Number of clusters (you can adjust this)
            _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center)
            res = center[label.flatten()]
            self.processed_image = res.reshape(self.image.shape)
            self.show_image(self.processed_image, 'after')
            QMessageBox.information(self, 'Success', 'Manual K-Means Applied!')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ui_MainWindow()
    window.show()
    sys.exit(app.exec_())
