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
        self.menuTranslation = self.menubar.addMenu("Translation")  # New menu for translation

        self.actionOpen = self.menuFile.addAction("Open Image")
        self.actionOpen.triggered.connect(self.open_image)

        self.actionSave = self.menuFile.addAction("Save Image")
        self.actionSave.triggered.connect(self.save_image)


        # New translation actions
        self.actionUp = self.menuTranslation.addAction("Up")
        self.actionUp.triggered.connect(lambda: self.translate_image("up"))

        self.actionDown = self.menuTranslation.addAction("Down")
        self.actionDown.triggered.connect(lambda: self.translate_image("down"))

        self.actionLeft = self.menuTranslation.addAction("Left")
        self.actionLeft.triggered.connect(lambda: self.translate_image("left"))

        self.actionRight = self.menuTranslation.addAction("Right")
        self.actionRight.triggered.connect(lambda: self.translate_image("right"))

        self.beforeImageView = QLabel(self.centralwidget)
        self.beforeImageView.setStyleSheet("border: 2px solid black;")
        grid_layout.addWidget(self.beforeImageView, 0, 0)

        self.afterImageView = QLabel(self.centralwidget)
        self.afterImageView.setStyleSheet("border: 2px solid black;")
        grid_layout.addWidget(self.afterImageView, 0, 1)

    def open_image(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Open Image")
        if filepath:
            self.image = cv2.imread(filepath)
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

    def dilate_image(self, kernel_size, shape=cv2.MORPH_RECT):
        kernel = cv2.getStructuringElement(shape, kernel_size)
        self.processed_image = cv2.dilate(self.image, kernel, iterations=1)
        self.show_image(self.processed_image, 'after')
        QMessageBox.information(self, 'Success', 'Image processed successfully!')

    def translate_image(self, direction):
        h, w, _ = self.image.shape
        M = np.float32([[1, 0, 0], [0, 1, 0]])  # Identity matrix

        if direction == 'up':
            M[1, 2] = -50  # Move up
        elif direction == 'down':
            M[1, 2] = 50  # Move down
        elif direction == 'left':
            M[0, 2] = -50  # Move left
        elif direction == 'right':
            M[0, 2] = 50  # Move right

        self.processed_image = cv2.warpAffine(self.image, M, (w, h))
        self.show_image(self.processed_image, 'after')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ui_MainWindow()
    window.show()
    sys.exit(app.exec_())
