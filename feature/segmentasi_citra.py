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
            # Convert image to grayscale
            r, g, b = self.image[:,:,0], self.image[:,:,1], self.image[:,:,2]
            gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)

            # Apply binary thresholding
            self.processed_image = np.where(gray >= 128, 255, 0).astype(np.uint8)

            # Convert back to 3-channel image
            self.processed_image = np.stack([self.processed_image] * 3, axis=2)
            self.show_image(self.processed_image, 'after')
            QMessageBox.information(self, 'Success', 'Binary Thresholding Applied!')

    def otsu_threshold(self):
        if hasattr(self, 'image'):
            # Convert image to grayscale
            r, g, b = self.image[:,:,0], self.image[:,:,1], self.image[:,:,2]
            gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)

            # Compute histogram
            hist = np.histogram(gray, bins=np.arange(0, 256), range=(0, 255))[0]

            # Compute Otsu threshold
            total = gray.size
            current_max, threshold = 0, 0
            sumT, sumF, countB, countF = 0, np.sum(hist), 0, total
            for i in range(0, 256):
                countB += hist[i]
                countF = total - countB
                if countF == 0:
                    break
                sumT += i * hist[i]
                mB = sumT / countB
                mF = (sumF - sumT) / countF
                varBetween = countB * countF * (mB - mF) ** 2
                if varBetween > current_max:
                    current_max, threshold = varBetween, i

            # Apply Otsu thresholding
            self.processed_image = np.where(gray > threshold, 255, 0).astype(np.uint8)

            # Convert back to 3-channel image
            self.processed_image = np.stack([self.processed_image] * 3, axis=2)
            self.show_image(self.processed_image, 'after')
            QMessageBox.information(self, 'Success', 'Otsu Thresholding Applied!')
    def contour_based_segmentation(self):
        if hasattr(self, 'image'):
            r, g, b = self.image[:,:,0], self.image[:,:,1], self.image[:,:,2]
            gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)

            # Gaussian blur for denoising
            blurred = ndimage.gaussian_filter(gray, sigma=1.0)

            # Use Sobel operators to get gradients
            sx = ndimage.sobel(blurred, axis=0, mode='constant')
            sy = ndimage.sobel(blurred, axis=1, mode='constant')
            sobel = np.hypot(sx, sy)

            # Binarize
            segmented_image = (sobel > np.percentile(sobel, 90)).astype(np.uint8) * 255

            self.processed_image = np.stack([segmented_image] * 3, axis=2)
            self.show_image(self.processed_image, 'after')
            QMessageBox.information(self, 'Success', 'Contour-Based Segmentation Applied!')
        else:
            QMessageBox.warning(self, 'Warning', 'No image loaded.')


    def manual_watershed(self):
        if hasattr(self, 'image'):
            r, g, b = self.image[:,:,0], self.image[:,:,1], self.image[:,:,2]
            gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)

            # Get binary image
            threshold = np.percentile(gray, 50)
            binary = (gray > threshold).astype(np.uint8)

            # Dilation
            kernel = np.ones((3, 3), np.uint8)
            sure_bg = ndimage.binary_dilation(binary, structure=kernel, iterations=3).astype(np.uint8)

            # Distance transform
            dist_transform = ndimage.distance_transform_edt(binary)

            # Get sure foreground
            sure_fg = (dist_transform > 0.7 * dist_transform.max()).astype(np.uint8)

            # Unknown region
            unknown = sure_bg - sure_fg

            # Watershed
            markers, _ = ndimage.label(sure_fg)
            markers += 1
            markers[unknown == 1] = 0
            markers = ndimage.watershed_ift(input=sure_bg, markers=markers)
            
            self.processed_image = self.image.copy()
            self.processed_image[markers == -1] = [0, 0, 255]
            
            self.show_image(self.processed_image, 'after')
            QMessageBox.information(self, 'Success', 'Manual Watershed Applied!')
        else:
            QMessageBox.warning(self, 'Warning', 'No image loaded.')

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
