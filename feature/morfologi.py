from PyQt5.QtWidgets import QMainWindow, QFileDialog, QApplication, QLabel, QMessageBox, QWidget, QGridLayout,QAction
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import sys

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
        MainWindow.resize(1200,800)

        grid_layout = QGridLayout()

        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setLayout(grid_layout)

        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = self.menuBar()

        self.menuFile = self.menubar.addMenu("File")
        self.menuErosion = self.menubar.addMenu("Erosion")
        self.menuDilation = self.menubar.addMenu("Dilation")
        self.menuOpening = self.menubar.addMenu("Opening")
        self.menuClosing = self.menubar.addMenu("Closing")

        # File menu
        self.actionOpen = QAction("Open Image", self)
        self.actionOpen.triggered.connect(self.open_image)
        self.menuFile.addAction(self.actionOpen)

        self.actionSave = QAction("Save Image", self)
        self.actionSave.triggered.connect(self.save_image)
        self.menuFile.addAction(self.actionSave)

        # Erosion submenu
        self.addActionToMenu(self.menuErosion, 'Square 3', self.morph_erosion_square3)
        self.addActionToMenu(self.menuErosion, 'Square 5', self.morph_erosion_square5)
        self.addActionToMenu(self.menuErosion, 'Cross 3', self.morph_erosion_cross3)

        # Dilation submenu
        self.addActionToMenu(self.menuDilation, 'Square 3', self.morph_dilation_square3)
        self.addActionToMenu(self.menuDilation, 'Square 5', self.morph_dilation_square5)
        self.addActionToMenu(self.menuDilation, 'Cross 3', self.morph_dilation_cross3)

        # Opening submenu
        self.addActionToMenu(self.menuOpening, 'Square 3', self.morph_opening_square3)
        self.addActionToMenu(self.menuOpening, 'Square 5', self.morph_opening_square5)
        self.addActionToMenu(self.menuOpening, 'Cross 3', self.morph_opening_cross3)

        # Closing submenu
        self.addActionToMenu(self.menuClosing, 'Square 3', self.morph_closing_square3)
        self.addActionToMenu(self.menuClosing, 'Square 5', self.morph_closing_square5)
        self.addActionToMenu(self.menuClosing, 'Cross 3', self.morph_closing_cross3)

        self.beforeImageView = QLabel(self.centralwidget)
        self.beforeImageView.setStyleSheet("border: 2px solid black;")
        grid_layout.addWidget(self.beforeImageView, 0, 0)

        self.afterImageView = QLabel(self.centralwidget)
        self.afterImageView.setStyleSheet("border: 2px solid black;")
        grid_layout.addWidget(self.afterImageView, 0, 1)
        
        
    def addActionToMenu(self, menu, label, func):
        action = QAction(label, self)
        action.triggered.connect(func)
        menu.addAction(action)
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
        if len(image.shape) == 3:
            h, w, ch = image.shape
            bytes_per_line = ch * w
            image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        else:
            h, w = image.shape
            image = QImage(image.data, w, h, w, QImage.Format_Grayscale8)
            
        pixmap = QPixmap.fromImage(image)

        if pos == 'before':
            self.beforeImageView.setPixmap(pixmap)
        else:
            self.afterImageView.setPixmap(pixmap)


  # Placeholder functions for each morphological operation
    def morph_erosion_square3(self):
        if hasattr(self, 'image') and self.image is not None:
            kernel = np.ones((3, 3), np.uint8)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            processed_image = self.manual_erosion(gray_image, kernel)
            self.show_image(processed_image, 'after')
        else:
            print("Image not loaded properly")


    def morph_erosion_square5(self):
        if hasattr(self, 'image') and self.image is not None:
            kernel = np.ones((5, 5), np.uint8)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            processed_image = self.manual_erosion(gray_image, kernel)
            self.show_image(processed_image, 'after')
        else:
            print("Image not loaded properly")

    def morph_erosion_cross3(self):
        if hasattr(self, 'image') and self.image is not None:
            kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], np.uint8)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            processed_image = self.manual_erosion(gray_image, kernel)
            self.show_image(processed_image, 'after')
        else:
            print("Image not loaded properly")

    def manual_erosion(self, image, kernel):
        k_height, k_width = kernel.shape
        height, width = image.shape
        
        pad_height = k_height // 2
        pad_width = k_width // 2
        
        padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'constant')
        output_image = np.zeros_like(image)
        
        for i in range(height):
            for j in range(width):
                sub_matrix = padded_image[i:i + k_height, j:j + k_width]
                result = np.min(sub_matrix[kernel == 1])
                output_image[i, j] = result
                
        return output_image
    def morph_dilation_square3(self):
        if hasattr(self, 'image') and self.image is not None:
            kernel = np.ones((3, 3), np.uint8)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            processed_image = self.manual_dilation(gray_image, kernel)
            self.show_image(processed_image, 'after')
        else:
            print("Image not loaded properly")

    def morph_dilation_square5(self):
        if hasattr(self, 'image') and self.image is not None:
            kernel = np.ones((5, 5), np.uint8)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            processed_image = self.manual_dilation(gray_image, kernel)
            self.show_image(processed_image, 'after')
        else:
            print("Image not loaded properly")

    def morph_dilation_cross3(self):
        if hasattr(self, 'image') and self.image is not None:
            kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], np.uint8)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            processed_image = self.manual_dilation(gray_image, kernel)
            self.show_image(processed_image, 'after')
        else:
            print("Image not loaded properly")


    def morph_opening_square3(self):
        if hasattr(self, 'image') and self.image is not None:
            kernel = np.ones((3, 3), np.uint8)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            processed_image = self.manual_dilation(self.manual_erosion(gray_image, kernel), kernel)
            self.show_image(processed_image, 'after')
        else:
            print("Image not loaded properly")

    def morph_opening_square5(self):
        if hasattr(self, 'image') and self.image is not None:
            kernel = np.ones((5, 5), np.uint8)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            processed_image = self.manual_dilation(self.manual_erosion(gray_image, kernel), kernel)
            self.show_image(processed_image, 'after')
        else:
            print("Image not loaded properly")

    def morph_opening_cross3(self):
        if hasattr(self, 'image') and self.image is not None:
            kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], np.uint8)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            processed_image = self.manual_dilation(self.manual_erosion(gray_image, kernel), kernel)
            self.show_image(processed_image, 'after')
        else:
            print("Image not loaded properly")

    def morph_closing_square3(self):
        if hasattr(self, 'image') and self.image is not None:
            kernel = np.ones((3, 3), np.uint8)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            processed_image = self.manual_erosion(self.manual_dilation(gray_image, kernel), kernel)
            self.show_image(processed_image, 'after')
        else:
            print("Image not loaded properly")

    def morph_closing_square5(self):
        if hasattr(self, 'image') and self.image is not None:
            kernel = np.ones((5, 5), np.uint8)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            processed_image = self.manual_erosion(self.manual_dilation(gray_image, kernel), kernel)
            self.show_image(processed_image, 'after')
        else:
            print("Image not loaded properly")

    def morph_closing_cross3(self):
        if hasattr(self, 'image') and self.image is not None:
            kernel = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]], np.uint8)
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            processed_image = self.manual_erosion(self.manual_dilation(gray_image, kernel), kernel)
            self.show_image(processed_image, 'after')
        else:
            print("Image not loaded properly")
    def manual_dilation(self, image, kernel):
        kernel_center = (kernel.shape[0] // 2, kernel.shape[1] // 2)
        height, width = image.shape
        dilated_image = np.zeros((height, width), np.uint8)

        for y in range(height):
            for x in range(width):
                max_value = 0
                for ky in range(-kernel_center[0], kernel_center[0] + 1):
                    for kx in range(-kernel_center[1], kernel_center[1] + 1):
                        new_y = y + ky
                        new_x = x + kx
                        if 0 <= new_y < height and 0 <= new_x < width:
                            pixel = image[new_y, new_x]
                            max_value = max(max_value, pixel * kernel[ky + kernel_center[0], kx + kernel_center[1]])

                dilated_image[y, x] = max_value

        return dilated_image

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ui_MainWindow()
    window.show()
    sys.exit(app.exec_())
