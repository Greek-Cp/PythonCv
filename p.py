from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QScrollArea, QPushButton, QAction , QGridLayout,QApplication
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from aritmatika import Ui_Aritmatika as p
import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os
import subprocess
import platform



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
    def animate_title(self):
        titles = ["Mata Kuliah PCV", "Yanuar Tri Laksono(E41210753)", "TESTING", "APP"]
        self.setWindowTitle(titles[self.counter])
        self.counter = (self.counter + 1) % len(titles)
        
    def setupUi(self,MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200,800)
        
        
        # Setup timer for title animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.animate_title)
        self.counter = 0
        self.timer.start(1000)  # Update every 1 second
        grid_layout = QGridLayout()
        
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setLayout(grid_layout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.beforeImageView = QtWidgets.QLabel(self.centralwidget)
        self.beforeImageView.setObjectName("beforeImageView")
        self.beforeImageView.setStyleSheet("border: 2px solid black;")
        self.afterImageView = QtWidgets.QLabel(self.centralwidget)
        self.afterImageView.setObjectName("afterImageView")
        grid_layout.addWidget(self.beforeImageView, 0, 0)
        self.afterImageView.setStyleSheet("border: 2px solid black;")
        grid_layout.addWidget(self.afterImageView, 0, 1)

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuHistogram = QtWidgets.QMenu(self.menuView)
        self.menuHistogram.setObjectName("menuHistogram")
        self.menuColors = QtWidgets.QMenu(self.menubar)
        self.menuColors.setObjectName("menuColors")
        self.menuRGB = QtWidgets.QMenu(self.menuColors)
        self.menuRGB.setObjectName("menuRGB")
        self.menuRGB_to_Grayscale = QtWidgets.QMenu(self.menuColors)
        self.menuRGB_to_Grayscale.setObjectName("menuRGB_to_Grayscale")
        self.menuBrightness = QtWidgets.QMenu(self.menuColors)
        self.menuBrightness.setObjectName("menuBrightness")
        self.menuBit_Depth = QtWidgets.QMenu(self.menuColors)
        self.menuBit_Depth.setObjectName("menuBit_Depth")
        self.menuTentang = QtWidgets.QMenu(self.menubar)
        self.menuTentang.setObjectName("menuTentang")
        self.menuImage_Processing = QtWidgets.QMenu(self.menubar)
        self.menuImage_Processing.setObjectName("menuImage_Processing")
        self.menuAritmatical_Operation = QtWidgets.QMenu(self.menubar)
        self.menuAritmatical_Operation.setObjectName("menuAritmatical_Operation")
        arithmetic_action = self.menuAritmatical_Operation.addAction("Aritmatika")
        arithmetic_action.triggered.connect(self.frameArimatika)
        self.menuIler = QtWidgets.QMenu(self.menubar)
        self.menuIler.setObjectName("menuIler")
        self.menuEdge_Detection_2 = QtWidgets.QMenu(self.menuIler)
        self.menuEdge_Detection_2.setObjectName("menuEdge_Detection_2")
        self.menuGaussian_Blur = QtWidgets.QMenu(self.menuIler)
        self.menuGaussian_Blur.setObjectName("menuGaussian_Blur")
        self.menuEdge_Detection = QtWidgets.QMenu(self.menubar)
        self.menuEdge_Detection.setObjectName("menuEdge_Detection")

        self.menuMorfologi = QtWidgets.QMenu(self.menubar)
        self.menuMorfologi.setObjectName("menuMorfologi")
             
        #untuk menambahkan object pada menu  bar
        self.menuUniform = QtWidgets.QMenu(self.menubar)
        self.menuUniform.setObjectName("menuUniform")
        
     
        
        self.actionA = QtWidgets.QAction(MainWindow)
        self.actionA.setObjectName("actionSquareopant")
     
        
        
        #menambahkan item sub menu berupa  action
        self.actionUniformScaling = QtWidgets.QAction(MainWindow)  
        #menambahkan action ke menu
        self.menuUniform.addAction(self.actionUniformScaling)
        py_files = ['crop.py', 'dilatasi.py', 'morfologi.py', 'ROI.py', 'segmentasi_citra.py', 'translasi.py', 'treshold.py', 'unsharp_masking.py']
        for file in py_files:
            # Removing .py extension
            name = os.path.splitext(file)[0]
            action = QAction(name, self)
            action.triggered.connect(lambda checked, file=file: self.runPythonScript(file))
            self.menuUniform.addAction(action)
        self.menuErosion = QtWidgets.QMenu(self.menuMorfologi)
        self.menuErosion.setObjectName("menuErosion")
        self.menuDilation = QtWidgets.QMenu(self.menuMorfologi)
        self.menuDilation.setObjectName("menuDilation")
        self.menuOpening = QtWidgets.QMenu(self.menuMorfologi)
        self.menuOpening.setObjectName("menuOpening")
        self.menuClosing = QtWidgets.QMenu(self.menuMorfologi)
        self.menuClosing.setObjectName("menuClosing")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.actionInput = QtWidgets.QAction(MainWindow)
        self.actionInput.setObjectName("actionInput")
        self.actionInput.triggered.connect(self.showInputHistogram)
        self.actionfliphorizontal = QtWidgets.QAction(MainWindow)
        self.actionfliphorizontal.setObjectName("actionFlipHorizontal")
        self.actionfliphorizontal.triggered.connect(self.flip_horizontal)
        self.actionflipvertical = QtWidgets.QAction(MainWindow)
        self.actionflipvertical.setObjectName("actionFlipVertical")
        self.actionflipvertical.triggered.connect(self.flip_vertical)
        self.actionRotate = QtWidgets.QAction(MainWindow)
        self.actionRotate.setObjectName("ActionRotate")
        self.actionRotate.triggered.connect(self.rotateImage)
        self.actionOutput = QtWidgets.QAction(MainWindow)
        self.actionOutput.setObjectName("actionOutput")
        self.actionOutput.triggered.connect(self.showOutputHistogram)
        self.actionInput_Output = QtWidgets.QAction(MainWindow)
        self.actionInput_Output.setObjectName("actionInput_Output")
        self.actionBrightness_Contrast = QtWidgets.QAction(MainWindow)
        self.actionBrightness_Contrast.setObjectName("actionBrightness_Contrast")
        self.actionBrightness_Contrast.triggered.connect(self.open_brightness_contrast_dialog)
        self.actionInvers = QtWidgets.QAction(MainWindow)
        self.actionInvers.setObjectName("actionInvers")
        self.actionInvers.triggered.connect(self.Invers)
        self.actionLog_Brightness = QtWidgets.QAction(MainWindow)
        self.actionLog_Brightness.setObjectName("actionLog_Brightness")
        self.actionGamma_Corretion = QtWidgets.QAction(MainWindow)
        self.actionGamma_Corretion.setObjectName("actionGamma_Corretion")
        self.actionKuning = QtWidgets.QAction(MainWindow)
        self.actionKuning.setObjectName("actionKuning")
        self.actionOrange = QtWidgets.QAction(MainWindow)
        self.actionOrange.setObjectName("actionOrange")
        self.actionCyan = QtWidgets.QAction(MainWindow)
        self.actionCyan.setObjectName("actionCyan")
        self.actionPurple = QtWidgets.QAction(MainWindow)
        self.actionPurple.setObjectName("actionPurple")
        self.actionGray = QtWidgets.QAction(MainWindow)
        self.actionGray.setObjectName("actionGray")
        self.actionCoklat = QtWidgets.QAction(MainWindow)
        self.actionCoklat.setObjectName("actionCoklat")
        self.actionMerah = QtWidgets.QAction(MainWindow)
        self.actionMerah.setObjectName("actionMerah")
        self.actionAverage = QtWidgets.QAction(MainWindow)
        self.actionAverage.setObjectName("actionAverage")
        self.actionAverage.triggered.connect(self.showAverage)
        self.actionLightness = QtWidgets.QAction(MainWindow)
        self.actionLightness.setObjectName("actionLightness")
        self.actionLightness.triggered.connect(self.showLightness)
        self.actionLuminance = QtWidgets.QAction(MainWindow)
        self.actionLuminance.setObjectName("actionLuminance")
        self.actionLuminance.triggered.connect(self.showLuminance)
        self.actionContrast = QtWidgets.QAction(MainWindow)
        self.actionContrast.setObjectName("actionContrast")
        self.action1_bit = QtWidgets.QAction(MainWindow)
        self.action1_bit.setObjectName("action1_bit")
        self.action1_bit.triggered.connect(self.onConvertClicked1)
        self.action2_bit = QtWidgets.QAction(MainWindow)
        self.action2_bit.setObjectName("action2_bit")
        self.action2_bit.triggered.connect(self.onConvertClicked2)
        self.action3_bit = QtWidgets.QAction(MainWindow)
        self.action3_bit.setObjectName("action3_bit")
        self.action3_bit.triggered.connect(self.onConvertClicked3)
        self.action4_bit = QtWidgets.QAction(MainWindow)
        self.action4_bit.setObjectName("action4_bit")
        self.action4_bit.triggered.connect(self.onConvertClicked4)
        self.action5_bit = QtWidgets.QAction(MainWindow)
        self.action5_bit.setObjectName("action5_bit")
        self.action5_bit.triggered.connect(self.onConvertClicked5)
        self.action6_bit = QtWidgets.QAction(MainWindow)
        self.action6_bit.setObjectName("action6_bit")
        self.action6_bit.triggered.connect(self.onConvertClicked6)
        self.action7_bit = QtWidgets.QAction(MainWindow)
        self.action7_bit.setObjectName("action7_bit")
        self.action7_bit.triggered.connect(self.onConvertClicked7)
        self.actionHistogram_Equalization = QtWidgets.QAction(MainWindow)
        self.actionHistogram_Equalization.setObjectName("actionHistogram_Equalization")
        self.actionHistogram_Equalization.triggered.connect(self.HistogramEqGrayscale)
        self.actionFuzzy_HE_RGB = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_HE_RGB.setObjectName("actionFuzzy_HE_RGB")
        self.actionFuzzy_HE_RGB.triggered.connect(self.FuzzyToRgb)
        self.actionFuzzy_Grayscale = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_Grayscale.setObjectName("actionFuzzy_Grayscale")
        self.actionFuzzy_Grayscale.triggered.connect(self.fuzzyGrayScale)
        self.actionIdentity = QtWidgets.QAction(MainWindow)
        self.actionIdentity.setObjectName("actionIdentity")
        self.actionSharpen = QtWidgets.QAction(MainWindow)
        self.actionSharpen.setObjectName("actionSharpen")
        self.actionSharpen.triggered.connect(self.sharpen)
        self.actionUnsharp_Masking = QtWidgets.QAction(MainWindow)
        self.actionUnsharp_Masking.setObjectName("actionUnsharp_Masking")
        self.actionAvarage_Filter = QtWidgets.QAction(MainWindow)
        self.actionAvarage_Filter.setObjectName("actionAvarage_Filter")
        self.actionLow_Pass_Filler = QtWidgets.QAction(MainWindow)
        self.actionLow_Pass_Filler.setObjectName("actionLow_Pass_Filler")
        self.actionLow_Pass_Filler.triggered.connect(self.lowPassFilter)
        self.actionHight_Pass_Filter = QtWidgets.QAction(MainWindow)
        self.actionHight_Pass_Filter.setObjectName("actionHight_Pass_Filter")
        self.actionHight_Pass_Filter.triggered.connect(self.HighPassFilter)
        self.actionBandstop_Filter = QtWidgets.QAction(MainWindow)
        self.actionBandstop_Filter.setObjectName("actionBandstop_Filter")
        self.actionEdge_Detection_1 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_1.setObjectName("actionEdge_Detection_1")
        self.actionEdge_Detection_2 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_2.setObjectName("actionEdge_Detection_2")
        self.actionEdge_Detection_3 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_3.setObjectName("actionEdge_Detection_3")
        self.actionGaussian_Blur_3_x_3 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_3_x_3.setObjectName("actionGaussian_Blur_3_x_3")
        self.actionGaussian_Blur_3_x_3.triggered.connect(self.gaussian_blur3x3)
        self.actionGaussian_Blur_5_X_5 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_5_X_5.setObjectName("actionGaussian_Blur_5_X_5")
        self.actionGaussian_Blur_5_X_5.triggered.connect(self.gaussian_blur5x5)
        self.actionPrewitt = QtWidgets.QAction(MainWindow)
        self.actionPrewitt.setObjectName("actionPrewitt")
        self.actionSobel = QtWidgets.QAction(MainWindow)
        self.actionSobel.setObjectName("actionSobel")
        self.actionRobert = QtWidgets.QAction(MainWindow)
        self.actionRobert.setObjectName("actionRobert")
        self.actionSquare_3 = QtWidgets.QAction(MainWindow)
        self.actionSquare_3.setObjectName("actionSquare_3")
        self.actionSquare_5 = QtWidgets.QAction(MainWindow)
        self.actionSquare_5.setObjectName("actionSquare_5")
        self.actionCross_3 = QtWidgets.QAction(MainWindow)
        self.actionCross_3.setObjectName("actionCross_3")
        self.actionSquare_4 = QtWidgets.QAction(MainWindow)
        self.actionSquare_4.setObjectName("actionSquare_4")
        self.actionSquare_6 = QtWidgets.QAction(MainWindow)
        self.actionSquare_6.setObjectName("actionSquare_6")
        self.actionCross = QtWidgets.QAction(MainWindow)
        self.actionCross.setObjectName("actionCross")
        self.actionSquare_9 = QtWidgets.QAction(MainWindow)
        self.actionSquare_9.setObjectName("actionSquare_9")
        self.actionSquare_10 = QtWidgets.QAction(MainWindow)
        self.actionSquare_10.setObjectName("actionSquare_10")
        self.actionOpen_File = QtWidgets.QAction(MainWindow)
        self.actionOpen_File.setObjectName("actionOpen_File")
        self.actionOpen_File.triggered.connect(self.openFile)
        self.actionSave_As = QtWidgets.QAction(MainWindow)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionSave_As.triggered.connect(self.saveImage)
        self.actionKeluar = QtWidgets.QAction(MainWindow)
        self.actionKeluar.setObjectName("actionKeluar")
        self.actionKeluar.triggered.connect(self.exitApplication)
        self.menuHistogram.addAction(self.actionInput)
        self.menuHistogram.addAction(self.actionOutput)
        self.menuHistogram.addAction(self.actionInput_Output) 
        self.menuView.addAction(self.menuHistogram.menuAction())
        self.menuRGB.addAction(self.actionKuning)
        self.menuRGB.addAction(self.actionOrange)
        self.menuRGB.addAction(self.actionCyan)
        self.menuRGB.addAction(self.actionPurple)
        self.menuRGB.addAction(self.actionGray)
        self.menuRGB.addAction(self.actionCoklat)
        self.menuRGB.addAction(self.actionMerah)
        self.menuRGB_to_Grayscale.addAction(self.actionAverage)
        self.menuRGB_to_Grayscale.addAction(self.actionLightness)
        self.menuRGB_to_Grayscale.addAction(self.actionLuminance)
        self.menuBrightness.addAction(self.actionContrast)
        self.menuBit_Depth.addAction(self.action1_bit) 
        self.menuBit_Depth.addAction(self.action2_bit)
        self.menuBit_Depth.addAction(self.action3_bit)
        self.menuBit_Depth.addAction(self.action4_bit)
        self.menuBit_Depth.addAction(self.action5_bit)
        self.menuBit_Depth.addAction(self.action6_bit)
        self.menuBit_Depth.addAction(self.action7_bit)
        self.menuColors.addAction(self.menuRGB.menuAction())
        self.menuColors.addAction(self.menuRGB_to_Grayscale.menuAction())
        self.menuColors.addAction(self.menuBrightness.menuAction())
        self.menuColors.addAction(self.actionBrightness_Contrast)
        self.menuColors.addAction(self.actionInvers)
        self.menuColors.addAction(self.actionLog_Brightness)
        self.menuColors.addAction(self.menuBit_Depth.menuAction())
        self.menuColors.addAction(self.actionGamma_Corretion)
        self.menuImage_Processing.addAction(self.actionHistogram_Equalization)
        self.menuImage_Processing.addAction(self.actionFuzzy_HE_RGB)
        self.menuImage_Processing.addAction(self.actionFuzzy_Grayscale)
        self.menuEdge_Detection_2.addAction(self.actionEdge_Detection_1)
        self.menuEdge_Detection_2.addAction(self.actionEdge_Detection_2)
        self.menuEdge_Detection_2.addAction(self.actionEdge_Detection_3)
        self.menuGaussian_Blur.addAction(self.actionGaussian_Blur_3_x_3)
        self.menuGaussian_Blur.addAction(self.actionGaussian_Blur_5_X_5)
        self.menuIler.addAction(self.actionIdentity)
        self.menuIler.addAction(self.menuEdge_Detection_2.menuAction())
        self.menuIler.addAction(self.actionSharpen)
        self.menuIler.addAction(self.menuGaussian_Blur.menuAction())
        self.menuIler.addAction(self.actionUnsharp_Masking)
        self.menuIler.addAction(self.actionAvarage_Filter)
        self.menuIler.addAction(self.actionLow_Pass_Filler)
        self.menuIler.addAction(self.actionHight_Pass_Filter)
        self.menuIler.addAction(self.actionBandstop_Filter)
        self.menuEdge_Detection.addAction(self.actionPrewitt)
        self.menuEdge_Detection.addAction(self.actionSobel)
        self.menuEdge_Detection.addAction(self.actionRobert)
        self.menuErosion.addAction(self.actionSquare_3)
        self.menuErosion.addAction(self.actionSquare_5)
        self.menuErosion.addAction(self.actionCross_3)
        self.menuDilation.addAction(self.actionSquare_4)
        self.menuDilation.addAction(self.actionSquare_6)
        self.menuDilation.addAction(self.actionCross)
        self.menuOpening.addAction(self.actionSquare_9)
        self.menuClosing.addAction(self.actionSquare_10)

        self.menuMorfologi.addAction(self.menuErosion.menuAction())
        self.menuMorfologi.addAction(self.menuDilation.menuAction())
        self.menuMorfologi.addAction(self.menuOpening.menuAction())
        self.menuMorfologi.addAction(self.menuClosing.menuAction())


        self.menuFile.addAction(self.actionOpen_File)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addAction(self.actionKeluar)
        self.menuTentang.addAction(self.actionfliphorizontal)
        self.menuTentang.addAction(self.actionflipvertical)
        self.menuTentang.addAction(self.actionRotate)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuColors.menuAction())
        self.menubar.addAction(self.menuTentang.menuAction())
        self.menubar.addAction(self.menuImage_Processing.menuAction())
        self.menubar.addAction(self.menuAritmatical_Operation.menuAction())
        self.menubar.addAction(self.menuIler.menuAction())
        self.menubar.addAction(self.menuEdge_Detection.menuAction())
        self.menubar.addAction(self.menuMorfologi.menuAction())
        self.menubar.addAction(self.menuUniform.menuAction())
        
        self.actionRobert.triggered.connect(self.RobertFilter)
        self.actionPrewitt.triggered.connect(self.prewitt)
        self.actionSobel.triggered.connect(self.sobelEdgeDetection)
        self.actionIdentity.triggered.connect(self.identity)
        
        self.actionSquare_3.triggered.connect(self.erosikotak3)
        self.actionSquare_5.triggered.connect(self.erosikotak5)
        self.actionCross_3.triggered.connect(self.ErosiX)
        
        self.actionSquare_4.triggered.connect(self.dilasikotak3)
        self.actionSquare_6.triggered.connect(self.dilasikotak5)
        self.actionCross.triggered.connect(self.DilasiX)
        
        self.actionKuning.triggered.connect(self.Kuning)
        self.actionOrange.triggered.connect(self.Orange)
        self.actionCyan.triggered.connect(self.Cyan)
        self.actionPurple.triggered.connect(self.Purple)
        self.actionGray.triggered.connect(self.Gray)
        self.actionCoklat.triggered.connect(self.Coklat)
        self.actionMerah.triggered.connect(self.Merah)
        
        self.setGeometry(100, 100, 800, 600)
        self.show()
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
    def runPythonScript(self, file_name):
        script_path = os.path.join('feature', file_name)
        
        # Cek sistem operasi
        os_type = platform.system()
        
        if os_type == "Windows":
            # Jalankan script untuk Windows
            subprocess.run(['python', script_path])
        elif os_type == "Darwin":
            # Jalankan script untuk Mac
            
            # Anda bisa menggunakan 'brew' untuk menginstall python jika belum ada
            # Ini hanyalah contoh, sesuaikan perintahnya sesuai kebutuhan
            # subprocess.run(['brew', 'install', 'python3'])
            
            # Kemudian jalankan script python
            subprocess.run(['/opt/homebrew/bin/python3', script_path])
        else:
            print("Sistem operasi tidak dikenali")

    # Pengujian fungsi
    # runPythonScript(ObjekSelfPalsu, 'nama_file.py')  # Gantikan ObjekSelfPalsu dengan objek self yang sesuai

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuHistogram.setTitle(_translate("MainWindow", "Histogram"))
        self.menuColors.setTitle(_translate("MainWindow", "Colors"))
        self.menuRGB.setTitle(_translate("MainWindow", "RGB"))
        self.menuRGB_to_Grayscale.setTitle(_translate("MainWindow", "RGB to Grayscale"))
        self.menuBrightness.setTitle(_translate("MainWindow", "Brightness"))
        self.menuBit_Depth.setTitle(_translate("MainWindow", "Bit Depth"))
        self.menuTentang.setTitle(_translate("MainWindow", "Geometri"))
        self.menuImage_Processing.setTitle(_translate("MainWindow", "Image Processing"))
        self.menuAritmatical_Operation.setTitle(_translate("MainWindow", "Aritmatical Operation"))
        self.menuIler.setTitle(_translate("MainWindow", "Filter"))
        self.menuEdge_Detection_2.setTitle(_translate("MainWindow", "Edge Detection"))
        self.menuGaussian_Blur.setTitle(_translate("MainWindow", "Gaussian Blur"))
        self.menuEdge_Detection.setTitle(_translate("MainWindow", "Edge Detection"))
        self.menuMorfologi.setTitle(_translate("MainWindow", "Morfologi"))

        
        #untuk menambahkan nama pada menu item 
        self.menuUniform.setTitle(_translate("MainWindow","Other"))
        #untuk mengeset nama action
        self.actionUniformScaling.setText(_translate("MainWindow","Uniform Scaling"))
        #untuk mengeset nama action
        self.actionA.setText(_translate("MainWindow","Action Tes"))
        
        self.menuErosion.setTitle(_translate("MainWindow", "Erosion"))
        self.menuDilation.setTitle(_translate("MainWindow", "Dilation"))
        self.menuOpening.setTitle(_translate("MainWindow", "Opening"))
        self.menuClosing.setTitle(_translate("MainWindow", "Closing"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionInput.setText(_translate("MainWindow", "Input"))
        self.actionOutput.setText(_translate("MainWindow", "Output"))
        self.actionInput_Output.setText(_translate("MainWindow", "Input Output"))
        self.actionBrightness_Contrast.setText(_translate("MainWindow", "Brightness - Contrast"))
        self.actionInvers.setText(_translate("MainWindow", "Invers"))
        self.actionLog_Brightness.setText(_translate("MainWindow", "Log Brightness"))
        self.actionGamma_Corretion.setText(_translate("MainWindow", "Gamma Corretion"))
        self.actionKuning.setText(_translate("MainWindow", "Kuning"))
 
        self.actionOrange.setText(_translate("MainWindow", "Orange"))
        self.actionCyan.setText(_translate("MainWindow", "Cyan"))
        self.actionPurple.setText(_translate("MainWindow", "Purple"))
        self.actionGray.setText(_translate("MainWindow", "Gray"))
        self.actionCoklat.setText(_translate("MainWindow", "Coklat"))
        self.actionMerah.setText(_translate("MainWindow", "Merah"))
        self.actionAverage.setText(_translate("MainWindow", "Average"))
        self.actionLightness.setText(_translate("MainWindow", "Lightness"))
        self.actionLuminance.setText(_translate("MainWindow", "Luminance"))
        self.actionContrast.setText(_translate("MainWindow", "Contrast"))
        self.action1_bit.setText(_translate("MainWindow", "1 bit"))
        self.action2_bit.setText(_translate("MainWindow", "2 bit"))
        self.action3_bit.setText(_translate("MainWindow", "3 bit"))
        self.action4_bit.setText(_translate("MainWindow", "4 bit"))
        self.action5_bit.setText(_translate("MainWindow", "5 bit"))
        self.action6_bit.setText(_translate("MainWindow", "6 bit"))
        self.action7_bit.setText(_translate("MainWindow", "7 bit"))
        self.actionHistogram_Equalization.setText(_translate("MainWindow", "Histogram Equalization"))
        self.actionFuzzy_HE_RGB.setText(_translate("MainWindow", "Fuzzy HE RGB"))
        self.actionFuzzy_Grayscale.setText(_translate("MainWindow", "Fuzzy Grayscale"))
        self.actionIdentity.setText(_translate("MainWindow", "Identity"))
        self.actionSharpen.setText(_translate("MainWindow", "Sharpen"))
        self.actionUnsharp_Masking.setText(_translate("MainWindow", "Unsharp Masking"))
        self.actionAvarage_Filter.setText(_translate("MainWindow", "Avarage Filter"))
        self.actionLow_Pass_Filler.setText(_translate("MainWindow", "Low Pass Filter"))
        self.actionHight_Pass_Filter.setText(_translate("MainWindow", "Hight Pass Filter"))
        self.actionBandstop_Filter.setText(_translate("MainWindow", "Bandstop Filter"))
        self.actionEdge_Detection_1.setText(_translate("MainWindow", "Edge Detection 1"))
        self.actionEdge_Detection_2.setText(_translate("MainWindow", "Edge Detection 2"))
        self.actionEdge_Detection_3.setText(_translate("MainWindow", "Edge Detection 3"))
        self.actionGaussian_Blur_3_x_3.setText(_translate("MainWindow", "Gaussian Blur 3 x 3"))
        self.actionGaussian_Blur_5_X_5.setText(_translate("MainWindow", "Gaussian Blur 5 X 5 "))
        self.actionPrewitt.setText(_translate("MainWindow", "Prewitt"))
        self.actionSobel.setText(_translate("MainWindow", "Sobel"))
        self.actionSquare_3.setText(_translate("MainWindow", "Square 3"))
        self.actionSquare_5.setText(_translate("MainWindow", "Square 5"))
        self.actionCross_3.setText(_translate("MainWindow", "Cross 3"))
        self.actionSquare_4.setText(_translate("MainWindow", "Square 3"))
        self.actionSquare_6.setText(_translate("MainWindow", "Square 5"))
        self.actionCross.setText(_translate("MainWindow", "Cross 3"))
        self.actionSquare_9.setText(_translate("MainWindow", "Square 9"))
        self.actionSquare_10.setText(_translate("MainWindow", "Square 9"))
        self.actionOpen_File.setText(_translate("MainWindow", "Open File"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As"))
        self.actionKeluar.setText(_translate("MainWindow", "Keluar"))
        self.actionfliphorizontal.setText(_translate("mainWindow" , "Flip Horizontal"))
        self.actionflipvertical.setText(_translate("MainWindow","Flip Vertical"))
        self.actionRotate.setText(_translate("MainWindow","Rotate"))
        self.actionRobert.setText(_translate("mainWindow" , "Robert"))
    
    def prewitt(self):
        if self.beforeImageView.pixmap() is not None:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.beforeImageView.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    gx = 0
                    gy = 0

                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            pixel_color = QtGui.QColor(input_qimage.pixel(x + dx, y + dy))
                            intensity = pixel_color.red()  # Using red channel for grayscale

                            # Prewitt masks
                            gx += dx * intensity
                            gy += dy * intensity

                    edge_intensity = min(int(abs(gx) + abs(gy)), 255)
                    output_qimage.setPixelColor(x, y, QtGui.QColor(edge_intensity, edge_intensity, edge_intensity))

            # Convert QImage to QPixmap for display
            p = QtGui.QPixmap.fromImage(output_qimage)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)
            
    def Kuning(self):
        if self.beforeImageView is not None:
            image = self.beforeImageView.pixmap()
            inputan_image = image.toImage()

            for x in range(image.width()):
                for y in range(image.height()):
                    pixel = QtGui.QColor(inputan_image.pixel(x, y))

                    # Add yellow color to the pixel
                    r = min(pixel.red() + 100, 255)
                    g = min(pixel.green() + 100, 255)

                    # Set the new pixel color
                    inputan_image.setPixelColor(x, y, QtGui.QColor(r, g, pixel.blue()))

            p = QtGui.QPixmap.fromImage(inputan_image)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)

    def Orange(self):
        if self.beforeImageView is not None:
            image = self.beforeImageView.pixmap()   
            inputan_image = image.toImage()

            for x in range(image.width()):
                for y in range(image.height()):
                    pixel = QtGui.QColor(inputan_image.pixel(x, y))
                    

                    # Sample orange filter: increase red, decrease blue
                    r = min(pixel.red() + 100, 255)
                    g = min(pixel.green() + 50, 255)
                    b = min(pixel.blue() - 50, 255)

                    # Set the new pixel color
                    inputan_image.setPixelColor(x, y, QtGui.QColor(r, g, b))

            p= QtGui.QPixmap.fromImage(inputan_image)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)

    def Cyan(self):
        if self.beforeImageView is not None:
            image = self.beforeImageView.pixmap()
            inputan_image = image.toImage()

            for x in range(image.width()):
                for y in range(image.height()):
                    pixel = QtGui.QColor(inputan_image.pixel(x, y))

                # Set red and green components to 0, leave blue component unchanged
                    r = 0
                    g = pixel.green()
                    b = pixel.blue()

                # Set the new pixel color
                    inputan_image.setPixelColor(x, y, QtGui.QColor(r, g, b))

            p= QtGui.QPixmap.fromImage(inputan_image)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)

    def Purple(self):
        if self.beforeImageView is not None:
            image = self.beforeImageView.pixmap()
            inputan_image = image.toImage()

            for x in range(image.width()):
                for y in range(image.height()):
                    pixel = QtGui.QColor(inputan_image.pixel(x, y))

                # Add purple color to the pixel
                    r = min(pixel.red() + 100, 255)  # Ubah komponen merah (R)
                    g = 0  # Atur komponen hijau (G) menjadi 0 untuk membuat warna ungu
                    b = min(pixel.blue() + 100, 255)  # Ubah komponen biru (B)

                # Setel warna piksel yang baru
                    inputan_image.setPixelColor(x, y, QtGui.QColor(r, g, b))

            p = QtGui.QPixmap.fromImage(inputan_image)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)

    def Gray(self):
        if self.beforeImageView is not None:
            image = self.beforeImageView.pixmap()
            inputan_image = image.toImage()

            for x in range(image.width()):
                for y in range(image.height()):
                    pixel = QtGui.QColor(inputan_image.pixel(x, y))

                    # Calculate grayscale value using the average of R, G, and B values
                    grayscale_value = (pixel.red() + pixel.green() + pixel.blue()) // 3

                    # Set the new pixel color as grayscale
                    inputan_image.setPixelColor(x, y, QtGui.QColor(grayscale_value, grayscale_value, grayscale_value))

            p= QtGui.QPixmap.fromImage(inputan_image)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)
            
    def Coklat(self):
        if self.beforeImageView is not None:
            image = self.beforeImageView.pixmap()
            inputan_image = image.toImage()

            for x in range(image.width()):
                for y in range(image.height()):
                    pixel = QtGui.QColor(inputan_image.pixel(x, y))

                    # Ubah ke warna coklat (R lebih tinggi, G lebih rendah, B rendah)
                    r = min(pixel.red() + 50, 255)  # Ubah sesuai kebutuhan
                    g = max(pixel.green() - 50, 0)   # Ubah sesuai kebutuhan
                    b = max(pixel.blue() - 100, 0)  # Ubah sesuai kebutuhan

                    # Setel warna piksel baru
                    inputan_image.setPixelColor(x, y, QtGui.QColor(r, g, b))

            p = QtGui.QPixmap.fromImage(inputan_image)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)

    def Merah(self):
        if self.beforeImageView is not None:
            image = self.beforeImageView.pixmap()
            inputan_image = image.toImage()

            for x in range(image.width()):
                for y in range(image.height()):
                    pixel = QtGui.QColor(inputan_image.pixel(x, y))

                    # Set red color to the pixel (R=255, G=0, B=0)
                    r = pixel.red()
                    g = 0
                    b = 0

                    # Set the new pixel color
                    inputan_image.setPixelColor(x, y, QtGui.QColor(r, g, b))

            p= QtGui.QPixmap.fromImage(inputan_image)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)
                    
    def identity(self):
        if self.beforeImageView.pixmap() is not None:
            inputImage = self.beforeImageView.pixmap().toImage()
            width = inputImage.width()
            height = inputImage.height()
            outputImage = QtGui.QImage(width , height , QtGui.QImage.Format_RGB32)

            avg_filter = [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0]
            ]

            for i in range(width):
                for j in range(height):
                    c1 = QtGui.QColor(inputImage.pixel(i, j))
                    c1r, c1g, c1b = c1.red(), c1.green(), c1.blue()

                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            posX, posY = i + k, j + l

                            if posX < 0 or posY < 0 or posX >= width or posY >= height:
                                c1r += avg_filter[k + 1][l + 1] * 0
                                c1g += avg_filter[k + 1][l + 1] * 0
                                c1b += avg_filter[k + 1][l + 1] * 0
                            else:
                                c2 = QtGui.QColor(inputImage.pixel(posX, posY))
                                c1r += avg_filter[k + 1][l + 1] * c2.red()
                                c1g += avg_filter[k + 1][l + 1] * c2.green()
                                c1b += avg_filter[k + 1][l + 1] * c2.blue()

                    c1r = self.truncate(c1r)
                    c1g = self.truncate(c1g)
                    c1b = self.truncate(c1b)

                    outputImage.setPixel(i, j, QtGui.QColor(c1r, c1g, c1b).rgb())

            p = QtGui.QPixmap.fromImage(outputImage)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)


    @staticmethod
    def truncate(x):
        if x > 255:
            return 255
        elif x < 0:
            return 0
        return x
    
    def erosiKotak(self, size):
        if self.beforeImageView:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.beforeImageView.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Define the erosion element (square or cross)
            element = np.ones((size, size), dtype=np.uint8)

            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(input_qimage.pixel(x, y))
                    r, g, b, _ = pixel_color.getRgb()

                    # Apply erosion operation
                    if r == 0 and g == 0 and b == 0:
                        neighborhood = np.zeros((size, size), dtype=np.uint8)

                        # Check if the neighborhood matches the element
                        if np.array_equal(neighborhood, element):
                            r = 255
                            g = 255
                            b = 255

                    # Set the new pixel color
                    output_qimage.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Convert QImage to QPixmap for display
            p = QtGui.QPixmap.fromImage(output_qimage)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)   
            
    def erosikotak3(self):
        self.erosiKotak(3)
        
    def erosikotak5(self):
        self.erosiKotak(5)
    
    def dilasiKotak(self, size):
        if self.beforeImageView:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.beforeImageView.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Define the dilation element (square or cross)
            element = np.ones((size, size), dtype=np.uint8)

            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(input_qimage.pixel(x, y))
                    r, g, b, _ = pixel_color.getRgb()

                    # Apply dilation operation
                    if r == 255 and g == 255 and b == 255:
                        neighborhood = np.ones((size, size), dtype=np.uint8)

                        # Set the new pixel color if any element in the neighborhood matches
                        if np.any(np.logical_and(neighborhood, element)):
                            r = 0
                            g = 0
                            b = 0

                    # Set the new pixel color
                    output_qimage.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Convert QImage to QPixmap for display
            p = QtGui.QPixmap.fromImage(output_qimage)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)
    def dilasikotak3(self):
        self.dilasiKotak(3)
        
    def dilasikotak5(self):
        self.dilasiKotak(5)
        
    def ErosiX(self):
        if self.beforeImageView:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.beforeImageView.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Define the 3x3 cross element
            element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(input_qimage.pixel(x, y))
                    r, g, b, _ = pixel_color.getRgb()

                    # Apply erosion operation
                    if r == 0 and g == 0 and b == 0:
                        neighborhood = np.zeros((3, 3), dtype=np.uint8)

                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                if 0 <= x + i < width and 0 <= y + j < height:
                                    if element[i + 1][j + 1] == 1:
                                        neighbor_color = QtGui.QColor(input_qimage.pixel(x + i, y + j))
                                        neighbor_r, neighbor_g, neighbor_b, _ = neighbor_color.getRgb()

                                        if neighbor_r == 0 and neighbor_g == 0 and neighbor_b == 0:
                                            neighborhood[i + 1][j + 1] = 1

                        # Check if the neighborhood matches the element
                        if np.array_equal(neighborhood, element):
                            r = 255
                            g = 255
                            b = 255

                    # Set the new pixel color
                    output_qimage.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Convert QImage to QPixmap for display
            p= QtGui.QPixmap.fromImage(output_qimage)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)
            
    def DilasiX(self):
        if self.beforeImageView:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.beforeImageView.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Define the 3x3 cross element
            element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)

            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(input_qimage.pixel(x, y))
                    r, g, b, _ = pixel_color.getRgb()

                    # Apply dilation operation
                    if r == 255 and g == 255 and b == 255:
                        neighborhood = np.zeros((3, 3), dtype=np.uint8)

                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                if 0 <= x + i < width and 0 <= y + j < height:
                                    if element[i + 1][j + 1] == 1:
                                        neighbor_color = QtGui.QColor(input_qimage.pixel(x + i, y + j))
                                        neighbor_r, neighbor_g, neighbor_b, _ = neighbor_color.getRgb()

                                        if neighbor_r == 255 and neighbor_g == 255 and neighbor_b == 255:
                                            neighborhood[i + 1][j + 1] = 1

                        # Set the new pixel color if any element in the neighborhood matches
                        if np.any(np.logical_and(neighborhood, element)):
                            r = 0
                            g = 0
                            b = 0

                    # Set the new pixel color
                    output_qimage.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Convert QImage to QPixmap for display
            p= QtGui.QPixmap.fromImage(output_qimage)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)
                                                          
    def sobelEdgeDetection(self):
        if self.beforeImageView.pixmap() is not None:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.beforeImageView.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

            # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    gx = 0
                    gy = 0

                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            pixel_color = QtGui.QColor(input_qimage.pixel(x + dx, y + dy))
                            intensity = pixel_color.red()  # Using red channel for grayscale

                            # Sobel masks
                            if dx == -1:
                                gx -= intensity
                            elif dx == 1:
                                gx += intensity
                            if dy == -1:
                                gy -= intensity
                            elif dy == 1:
                                gy += intensity

                    edge_intensity = min(int(abs(gx) + abs(gy)), 255)
                    output_qimage.setPixelColor(x, y, QtGui.QColor(edge_intensity, edge_intensity, edge_intensity))

            # Convert QImage to QPixmap for display
            p = QtGui.QPixmap.fromImage(output_qimage)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)
            
    def RobertFilter(self):
        if self.beforeImageView.pixmap() is not None:
            # Convert QPixmap to QImage for image processing
            input_qimage = self.beforeImageView.pixmap().toImage()
            width = input_qimage.width()
            height = input_qimage.height()

             # Create a new QImage for the processed image
            output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Define Robert Filter kernels for two directions
            kernel_x = np.array([[-1, 0], [0, 1]])
            kernel_y = np.array([[0, -1], [1, 0]])

            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    sum_x = sum_y = 0

                    for i in range(2):
                        for j in range(2):
                            pixel_color = QtGui.QColor(input_qimage.pixel(x - 1 + i, y - 1 + j))
                            gray = pixel_color.red()  # Assuming it's a grayscale image

                            sum_x += gray * kernel_x[i][j]
                            sum_y += gray * kernel_y[i][j]

             # Calculate the gradient magnitude
                    magnitude = int(np.sqrt(sum_x**2 + sum_y**2))
                    output_qimage.setPixelColor(x, y, QtGui.QColor(magnitude, magnitude, magnitude))

            # Convert QImage to QPixmap for display
            p = QtGui.QPixmap.fromImage(output_qimage)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)
                
    def HistogramEqGrayscale(self):
            width = self.image.width()
            height = self.image.height()
            equalized_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            # Menghitung histogram
            histogram = [0] * 256
            total_pixels = width * height

            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(*self.image.pixelColor(x, y).getRgb())
                    intensity = pixel_color.red()  # Kami asumsikan gambar grayscale
                    histogram[intensity] += 1

            # Menghitung distribusi kumulatif
            cumulative_distribution = [0] * 256
            cumulative_distribution[0] = histogram[0] / total_pixels

            for i in range(1, 256):
                cumulative_distribution[i] = cumulative_distribution[i - 1] + histogram[i] / total_pixels

            # Menyesuaikan nilai pixel pada gambar hasil
            for x in range(width):
                for y in range(height):
                    pixel_color = QtGui.QColor(*self.image.pixelColor(x, y).getRgb())
                    intensity = pixel_color.red()  # Kami asumsikan gambar grayscale
                    new_intensity = int(255 * cumulative_distribution[intensity])
                    new_color = QtGui.QColor(new_intensity, new_intensity, new_intensity)
                    equalized_image.setPixelColor(x, y, new_color)
            
            p = QtGui.QPixmap.fromImage(equalized_image)
            self.afterImageView.setPixmap(p)
            self.image = equalized_image
            
    def FuzzyToRgb(self):
            width = self.image.width()
            height = self.image.height()

            # Mengambil data piksel dari gambar input
            input_data = np.zeros((height, width, 3), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    color = QtGui.QColor(self.image.pixel(x, y))
                    input_data[y, x, 0] = color.red()
                    input_data[y, x, 1] = color.green()
                    input_data[y, x, 2] = color.blue()

            # Menerapkan rumus Fuzzy HE RGB
            output_data = np.zeros_like(input_data)
            for i in range(3):  # Loop untuk saluran warna (R, G, B)
                for y in range(height):
                    for x in range(width):
                        val = input_data[y, x, i]
                        if val < 128:
                            output_data[y, x, i] = int(2 * val ** 2 / 255.0)
                        else:
                            output_data[y, x, i] = int(255 - 2 * (255 - val) ** 2 / 255.0)

            # Membuat gambar output dan menampilkannya di pbOutput
            output_image = QtGui.QImage(output_data.data, width, height, width * 3, QtGui.QImage.Format_RGB888)
            p = QtGui.QPixmap.fromImage(output_image)
            self.afterImageView.setPixmap(p)
            self.image = output_image
            
    def fuzzyGrayScale(self):
        width = self.image.width()
        height = self.image.height()

        # Mengambil data piksel dari gambar input
        input_data = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                color = QtGui.QColor(self.image.pixel(x, y))
                # Menghitung nilai greyscale menggunakan rumus Fuzzy
                val = int(0.3 * color.red() + 0.59 * color.green() + 0.11 * color.blue())
                input_data[y, x] = val

        # Membuat gambar output dan menampilkannya di pbOutput
        output_image = QtGui.QImage(input_data.data, width, height, width, QtGui.QImage.Format_Grayscale8)
        p = QtGui.QPixmap.fromImage(output_image)
        self.afterImageView.setPixmap(p)
        self.image = output_image
        
    def gaussian_blur3x3(self):
        avgFilter = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        output_image = QtGui.QImage(self.image)

        for i in range(output_image.width()):
            for j in range(output_image.height()):
                # c1 = QtGui.QColor(self.input_image.pixel(i, j))
                c1r, c1g, c1b = 0, 0, 0

                for k in range(3):
                    for l in range(3):
                        posX = i + k
                        posY = j + l

                        if posX < 0 or posY < 0 or posX >= output_image.width() or posY >= output_image.height():
                            c1r += 0
                            c1g += 0
                            c1b += 0
                        else:
                            pixel_color = QtGui.QColor(self.image.pixel(posX, posY))
                            c1r += int((avgFilter[k][l] / 16) * pixel_color.red())
                            c1g += int((avgFilter[k][l] / 16) * pixel_color.green())
                            c1b += int((avgFilter[k][l] / 16) * pixel_color.blue())

                output_image.setPixelColor(i, j, QtGui.QColor(c1r, c1g, c1b))  # Use setPixelColor instead of setPixel

        p = QtGui.QPixmap.fromImage(output_image)
        self.afterImageView.setPixmap(p)
        self.afterImageView.setScaledContents(True)
        
    def HighPassFilter(self):
        input_qimage = self.beforeImageView.pixmap().toImage()
        width = input_qimage.width()
        height = input_qimage.height()

        # Create a new QImage for the processed image
        output_qimage = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

        for x in range(1, width - 1):
            for y in range(1, height - 1):
                # Get the colors of the surrounding pixels
                pixel_center = QtGui.QColor(input_qimage.pixel(x, y))
                pixel_left = QtGui.QColor(input_qimage.pixel(x - 1, y))
                pixel_right = QtGui.QColor(input_qimage.pixel(x + 1, y))
                pixel_up = QtGui.QColor(input_qimage.pixel(x, y - 1))
                pixel_down = QtGui.QColor(input_qimage.pixel(x, y + 1))

                # Calculate the high-pass filtered pixel value
                r = 5 * pixel_center.red() - pixel_left.red() - pixel_right.red() - pixel_up.red() - pixel_down.red()
                g = 5 * pixel_center.green() - pixel_left.green() - pixel_right.green() - pixel_up.green() - pixel_down.green()
                b = 5 * pixel_center.blue() - pixel_left.blue() - pixel_right.blue() - pixel_up.blue() - pixel_down.blue()

                # Clip the values to the valid color range (0-255)
                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))

                # Set the new pixel color
                output_qimage.setPixel(x, y, QtGui.qRgb(r, g, b))

        p = QtGui.QPixmap.fromImage(output_qimage)
        self.afterImageView.setPixmap(p)
        self.afterImageView.setScaledContents(True)
            
    def lowPassFilter(self):
        filterMatrix = [
            [1, 1, 1],
            [1, 4, 1],
            [1, 1, 1]
        ]
        imagee = self.beforeImageView.pixmap().toImage()
        output_image = QtGui.QImage(imagee)
        # width, height = self.beforeImageView.width(), self.beforeImageView.height()

        for x in range(output_image.width()):
            for y in range(output_image.height()):
                
                r, g, b = 0, 0, 0

                for i in range(3):
                    for j in range(3):
                        xi = x + i - 1
                        yi = y + j - 1
                        
                        if xi >= 0 and xi < output_image.width() and yi >= 0 and yi < output_image.height():
                            pixelColor = QtGui.QColor(imagee.pixel(xi, yi))
                            r += filterMatrix[i][j] * pixelColor.red()
                            g += filterMatrix[i][j] * pixelColor.green()
                            b += filterMatrix[i][j] * pixelColor.blue()

                r = min(max(0, r // 12), 255)
                g = min(max(0, g // 12), 255)
                b = min(max(0, b // 12), 255)
                output_image.setPixel(x, y, QtGui.qRgb(r, g, b))

        p = QtGui.QPixmap.fromImage(output_image)
        self.afterImageView.setPixmap(p)
        self.afterImageView.setScaledContents(True)
        
        
    def sharpen(self):
        avgFilter = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
        output_image = QtGui.QImage(self.image)

        for i in range(output_image.width()):
            for j in range(output_image.height()):
                # c1 = QtGui.QColor(self.input_image.pixel(i, j))
                c1r, c1g, c1b = 0, 0, 0

                for k in range(3):
                    for l in range(3):
                        posX = i + k
                        posY = j + l

                        if posX < 0 or posY < 0 or posX >= output_image.width() or posY >= output_image.height():
                            c1r += 0
                            c1g += 0
                            c1b += 0
                        else:
                            pixel_color = QtGui.QColor(self.image.pixel(posX, posY))
                            c1r += int((avgFilter[k][l]) * pixel_color.red())
                            c1g += int((avgFilter[k][l]) * pixel_color.green())
                            c1b += int((avgFilter[k][l]) * pixel_color.blue())
                
                c1r = max(0, min(c1r, 255))
                c1g = max(0, min(c1g, 255))
                c1b = max(0, min(c1b, 255))
                output_image.setPixelColor(i, j, QtGui.QColor(c1r, c1g, c1b))  # Use setPixelColor instead of setPixel

        p = QtGui.QPixmap.fromImage(output_image)
        self.afterImageView.setPixmap(p)
        self.afterImageView.setScaledContents(True)    
    
    def gaussian_blur5x5(self):
        avgFilter = [[1, 4, 6,4,1], [4, 16, 24,16,4], [6, 24, 36,24,6],[4,16,24,16,3],[1,4,6,4,1]]
        output_image = QtGui.QImage(self.image)

        for i in range(output_image.width()):
            for j in range(output_image.height()):
                # c1 = QtGui.QColor(self.input_image.pixel(i, j))
                c1r, c1g, c1b = 0, 0, 0

                for k in range(5):
                    for l in range(5):
                        posX = i + k
                        posY = j + l

                        if posX < 0 or posY < 0 or posX >= output_image.width() or posY >= output_image.height():
                            c1r += 0
                            c1g += 0
                            c1b += 0
                        else:
                            pixel_color = QtGui.QColor(self.image.pixel(posX, posY))
                            c1r += int((avgFilter[k][l] / 256) * pixel_color.red())
                            c1g += int((avgFilter[k][l] / 256) * pixel_color.green())
                            c1b += int((avgFilter[k][l] / 256) * pixel_color.blue())

                output_image.setPixelColor(i, j, QtGui.QColor(c1r, c1g, c1b))  # Use setPixelColor instead of setPixel

        p = QtGui.QPixmap.fromImage(output_image)
        self.afterImageView.setPixmap(p)
        self.afterImageView.setScaledContents(True)
    
    def showInputHistogram(self):
            pbalap = self.beforeImageView.pixmap()
            input_image = pbalap.toImage()
            width = input_image.width()
            height = input_image.height()
            input_data = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    color = QtGui.QColor(input_image.pixel(x, y))
                    val = int(0.3 * color.red() + 0.59 * color.green() + 0.11 * color.blue())
                    input_data[y, x] = val
            # Menghitung histogram
            histogram, bins = np.histogram(input_data, bins=256, range=(0, 256))
            # Menampilkan grafik histogram
            plt.figure(figsize=(8, 6))
            plt.bar(bins[:-1], histogram, width=1, color='blue')
            plt.title('Histogram Input')
            plt.xlabel('Intensitas Piksel')
            plt.ylabel('Frekuensi')
            plt.show()
            
    def bitdepth(self, bit):
            io = self.beforeImageView.pixmap()
            pbalappp = io.toImage()
            level = 255 / (2 ** bit - 1)
            
            for i in range(pbalappp.width()):
                for j in range(pbalappp.height()):
                    color = QtGui.QColor(pbalappp.pixel(i, j))
                    R = int(round(color.red() / level) * level)
                    G = int(round(color.green() / level) * level)
                    B = int(round(color.blue() / level) * level)
                    pbalappp.setPixel(i, j, QtGui.QColor(R, G, B).rgb())

            p = QtGui.QPixmap.fromImage(pbalappp)
            self.afterImageView.setPixmap(p)
            self.afterImageView.setScaledContents(True)
            
    def onConvertClicked1(self):
        bit_depth = 1
        self.bitdepth(bit_depth)  
         
    def onConvertClicked2(self):
        bit_depth = 2
        self.bitdepth(bit_depth)  
        
    def onConvertClicked3(self):
        bit_depth = 3
        self.bitdepth(bit_depth)
            
    def onConvertClicked4(self):
        bit_depth = 4
        self.bitdepth(bit_depth)  
         
    def onConvertClicked5(self):
        bit_depth = 5
        self.bitdepth(bit_depth)  
        
    def onConvertClicked6(self):
        bit_depth = 6
        self.bitdepth(bit_depth)   
    
    def onConvertClicked7(self):
        bit_depth = 7
        self.bitdepth(bit_depth)   
            
    def showOutputHistogram(self):
        
            pbalap = self.afterImageView.pixmap()
            input_image = pbalap.toImage()
            width = input_image.width()
            height = input_image.height()
            input_data = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    color = QtGui.QColor(input_image.pixel(x, y))
                    val = int(0.3 * color.red() + 0.59 * color.green() + 0.11 * color.blue())
                    input_data[y, x] = val
            # Menghitung histogram
            histogram, bins = np.histogram(input_data, bins=256, range=(0, 256))
            # Menampilkan grafik histogram
            plt.figure(figsize=(8, 6))
            plt.bar(bins[:-1], histogram, width=1, color='blue')
            plt.title('Histogram Output')
            plt.xlabel('Intensitas Piksel')
            plt.ylabel('Frekuensi')
            plt.show()
                                        
    def frameArimatika(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = p()
        self.ui.setupUi(self.window)
        self.window.show() 
        
    def saveImage(self):
        # Inisialisasi opsi untuk dialog pemilihan berkas
        options = QFileDialog.Options()
        # Menambahkan opsi mode baca saja ke dalam opsi dialog
        options |= QFileDialog.ReadOnly 
        # menampung file path dari dialog open file dan difilter hanya format png , jpg , bmp
        file_name, _ = QFileDialog.getSaveFileName(None, "Save Image File", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
        # check apakah terdapat path file
        if file_name:
            #Simpan gambar yang telah diformat
            pixmap = self.afterImageView.pixmap()
            pixmap.save(file_name)
           # self.label_4.setText(file_name)
            
    def rotateImage(self):
        rotation , ok = QtWidgets.QInputDialog.getInt(None , "Rotate Image","Enter rotation angle (degress):",0,-360,360)
        if ok:
            current_pixmap = self.beforeImageView.pixmap()
            second_pixmap = self.afterImageView.pixmap()
            
            if self.afterImageView.pixmap() is None:
                    rotated_image = current_pixmap.transformed(QtGui.QTransform().rotate(rotation))
                    self.afterImageView.setPixmap(rotated_image)
                    self.afterImageView.setAlignment(QtCore.Qt.AlignCenter)
                    self.afterImageView.setScaledContents(True)
                    self.image = rotated_image.toImage()
                    
            else:    
                    rotated_image = second_pixmap.transformed(QtGui.QTransform().rotate(rotation))
                    self.afterImageView.setPixmap(rotated_image)
                    self.afterImageView.setAlignment(QtCore.Qt.AlignCenter)
                    self.afterImageView.setScaledContents(True)
                    self.image = rotated_image.toImage()
                                   
    def Invers(self):
        # Ambil pixmap dari pbInput
        pixmap = self.beforeImageView.pixmap()
        if pixmap:
            img = pixmap.toImage()
            width, height = img.width(), img.height()

            for y in range(height):
                for x in range(width):
                    pixel = img.pixel(x, y)
                    # Dapatkan nilai merah (red), hijau (green), dan biru (blue) dari pixel
                    red = QtGui.qRed(pixel)
                    green = QtGui.qGreen(pixel)
                    blue = QtGui.qBlue(pixel)
                    # Inversi warna
                    inverted_color = QtGui.QColor(255 - red, 255 - green, 255 - blue)
                    # Set pixel ke warna invers pada gambar
                    img.setPixel(x, y, inverted_color.rgb())

            # Terapkan gambar invers pada pbOutput
            pixmap = QtGui.QPixmap.fromImage(img)
            self.afterImageView.setPixmap(pixmap)
            self.afterImageView.setScaledContents(True)
            self.image = img
                        
    def showAverage(self):
        width = self.image.width()
        height = self.image.height()
        average = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
        
        for y in range(height):
            for x in range(width):
                pixel_color = QtGui.QColor(self.image.pixel(x, y))
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                rumusAverage = int((r + g + b)/3)
                grayscale_color = QtGui.QColor(rumusAverage, rumusAverage, rumusAverage)
                average.setPixelColor(x, y, grayscale_color)
        
        p = QtGui.QPixmap.fromImage(average) 
        
        # self.image = average
          

    def showLuminance(self):
        width = self.image.width()
        height = self.image.height()
        grayscale_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

        for y in range(height):
            for x in range(width):
                pixel_color = QtGui.QColor(self.image.pixel(x, y))
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                rumusLuminance = int(0.299 * r + 0.587 * g + 0.114 * b)
                grayscale_color = QtGui.QColor(rumusLuminance, rumusLuminance, rumusLuminance)
                grayscale_image.setPixelColor(x, y, grayscale_color)
        
        p = QtGui.QPixmap.fromImage(grayscale_image) 
        self.afterImageView.setPixmap(p)
        # self.image = grayscale_image
        

    def openFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        filepath, _ = QFileDialog.getOpenFileName(None, "Open Image", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
        
        if filepath:
            # Directly use QImage to read the image
            self.image = QImage(filepath)
            
            # If the image isn't null, proceed with setting it up
            if not self.image.isNull():
                h = self.image.height()
                w = self.image.width()
                
                self.resize(w, h)  # Adjust window size to fit the image
                
                pixmap = QPixmap.fromImage(self.image)
                self.beforeImageView.setPixmap(pixmap)
                self.beforeImageView.setScaledContents(True)

                # If you still need a numpy array version of the image,
                # convert QImage to numpy array and store in a separate attribute
                ptr = self.image.bits()
                ptr.setsize(self.image.byteCount())
                self.image_np = np.array(ptr).reshape(h, w, 4)  # 4 for RGBA

    def show_image(self, image, pos='before'):
        pixmap = QPixmap.fromImage(image)
        if pos == 'before':
            self.beforeImageView.setPixmap(pixmap)
        else:
            self.afterImageView.setPixmap(pixmap)
    def showLightness(self):
        width = self.image.width()
        height = self.image.height()
        lightness = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
        
        for y in range(height):
            for x in range(width):
                pixel_color = QtGui.QColor(self.image.pixel(x, y))
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                rumusLightness = int((max(r,g,b)+ min(r,g,b))/2)
                grayscale_color = QtGui.QColor(rumusLightness, rumusLightness, rumusLightness)
                lightness.setPixelColor(x, y, grayscale_color)
        
        p = QtGui.QPixmap.fromImage(lightness) 
        self.afterImageView.setPixmap(p)
        # self.image = lightness
        
    def histogram(self):
        img = cv2.imread(self.checkHisto , 1)
        # alternative way to find histogram of an image
        plt.hist(img.ravel(),256,[0,256])
        plt.show()
           
    def apply_brightness_contrast(self, brightness, contrast):
        width = self.image.width()
        height = self.image.height()
        # Create a numpy array to store the adjusted image
        adjusted_image = np.zeros((height, width, 4), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                # Get the RGB pixel values
                r, g, b, a = QtGui.QColor(self.image.pixel(x, y)).getRgb()
                # Adjust brightness
                adjusted_r = min(max(r + brightness, 0), 255)
                adjusted_g = min(max(g + brightness, 0), 255)
                adjusted_b = min(max(b + brightness, 0), 255)
                # Adjust contrast
                adjusted_r = min(max(((adjusted_r - 127) * contrast) + 127, 0), 255)
                adjusted_g = min(max(((adjusted_g - 127) * contrast) + 127, 0), 255)
                adjusted_b = min(max(((adjusted_b - 127) * contrast) + 127, 0), 255)

                # Set the adjusted color values in the numpy array
                adjusted_image[y][x] = [adjusted_r, adjusted_g, adjusted_b, a]

        # Create a QImage from the numpy array
        adjusted_qimage = QtGui.QImage(adjusted_image.data, width, height, width * 4, QtGui.QImage.Format_RGBA8888)

        # Create a QPixmap from the QImage and set it to self.afterImageView
        adjusted_pixmap = QtGui.QPixmap.fromImage(adjusted_qimage)
        self.afterImageView.setPixmap(adjusted_pixmap)
        self.afterImageView.setAlignment(QtCore.Qt.AlignCenter)
        self.image = adjusted_qimage

    def open_brightness_contrast_dialog(self):
    # Open a dialog to get user input for brightness and contrast
        brightness, ok1 = QtWidgets.QInputDialog.getInt(None, "Brightness", "Enter brightness (-255 to 255):", 0, -255, 255)
        contrast, ok2 = QtWidgets.QInputDialog.getDouble(None, "Contrast", "Enter contrast (0.01 to 4.0):", 1.0, 0.01, 4.0)

        if ok1 and ok2:
            # Apply brightness and contrast adjustments
            self.apply_brightness_contrast(brightness, contrast)
            
    def flip_horizontal(self):
            width = self.image.width()
            height = self.image.height()

            # Create a numpy array to store the flipped image
            flipped_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGBA8888)

            for y in range(height):
                for x in range(width):
                    pixel_color = QtGui.QColor(self.image.pixel(x, y))
                    flipped_image.setPixelColor(width - 1 - x, y, pixel_color)       

            flipped_pixmap = QtGui.QPixmap.fromImage(flipped_image)
            self.afterImageView.setPixmap(flipped_pixmap)
            self.afterImageView.setAlignment(QtCore.Qt.AlignCenter)
            self.image = flipped_image
            
    def flip_vertical(self):
            width = self.image.width()
            height = self.image.height()
            flipped_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGBA8888)

            for y in range(height):
                for x in range(width):
                    pixel_color = QtGui.QColor(self.image.pixel(x, y))
                    flipped_image.setPixelColor(x, height - 1 - y, pixel_color) 

            flipped_pixmap = QtGui.QPixmap.fromImage(flipped_image)
            self.afterImageView.setPixmap(flipped_pixmap)
            self.afterImageView.setAlignment(QtCore.Qt.AlignCenter)
            self.image = flipped_image        
             
    def exitApplication(self):
        # untuk keluar dari aplikasi
        QtWidgets.qApp.quit()    
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Ui_MainWindow()
    window.show()
    sys.exit(app.exec_())
