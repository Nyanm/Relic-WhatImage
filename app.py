# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'app.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Main_App(object):
    def setupUi(self, Main_App):
        Main_App.setObjectName("Main_App")
        Main_App.resize(1200, 900)
        self.centralwidget = QtWidgets.QWidget(Main_App)
        self.centralwidget.setObjectName("centralwidget")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralwidget)
        self.graphicsView.setGeometry(QtCore.QRect(10, 10, 1181, 821))
        self.graphicsView.setObjectName("graphicsView")
        Main_App.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Main_App)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1200, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuImage = QtWidgets.QMenu(self.menubar)
        self.menuImage.setObjectName("menuImage")
        self.menuRotate = QtWidgets.QMenu(self.menuImage)
        self.menuRotate.setObjectName("menuRotate")
        self.menuScaling = QtWidgets.QMenu(self.menuImage)
        self.menuScaling.setObjectName("menuScaling")
        self.menuAdjustment = QtWidgets.QMenu(self.menuImage)
        self.menuAdjustment.setObjectName("menuAdjustment")
        self.menuConvert_to_Gray = QtWidgets.QMenu(self.menuAdjustment)
        self.menuConvert_to_Gray.setObjectName("menuConvert_to_Gray")
        self.menuFlip = QtWidgets.QMenu(self.menuImage)
        self.menuFlip.setObjectName("menuFlip")
        self.menuQuantization = QtWidgets.QMenu(self.menuImage)
        self.menuQuantization.setObjectName("menuQuantization")
        self.menuHistogram = QtWidgets.QMenu(self.menuImage)
        self.menuHistogram.setObjectName("menuHistogram")
        self.menuFilter = QtWidgets.QMenu(self.menubar)
        self.menuFilter.setObjectName("menuFilter")
        self.menuNoise = QtWidgets.QMenu(self.menuFilter)
        self.menuNoise.setObjectName("menuNoise")
        self.menuStylize = QtWidgets.QMenu(self.menuFilter)
        self.menuStylize.setObjectName("menuStylize")
        self.menuBlur = QtWidgets.QMenu(self.menuFilter)
        self.menuBlur.setObjectName("menuBlur")
        self.menuFT = QtWidgets.QMenu(self.menuFilter)
        self.menuFT.setObjectName("menuFT")
        self.menuSharpen = QtWidgets.QMenu(self.menuFilter)
        self.menuSharpen.setObjectName("menuSharpen")
        self.menuSpecial = QtWidgets.QMenu(self.menuFilter)
        self.menuSpecial.setObjectName("menuSpecial")
        self.menuFind_Edges_2 = QtWidgets.QMenu(self.menuFilter)
        self.menuFind_Edges_2.setObjectName("menuFind_Edges_2")
        self.menuAbout = QtWidgets.QMenu(self.menubar)
        self.menuAbout.setObjectName("menuAbout")
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName("menuEdit")
        Main_App.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Main_App)
        self.statusbar.setObjectName("statusbar")
        Main_App.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(Main_App)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(Main_App)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_As = QtWidgets.QAction(Main_App)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionQuit = QtWidgets.QAction(Main_App)
        self.actionQuit.setObjectName("actionQuit")
        self.actionSource_Code = QtWidgets.QAction(Main_App)
        self.actionSource_Code.setObjectName("actionSource_Code")
        self.actionAbout_me = QtWidgets.QAction(Main_App)
        self.actionAbout_me.setObjectName("actionAbout_me")
        self.actionTranslation = QtWidgets.QAction(Main_App)
        self.actionTranslation.setObjectName("actionTranslation")
        self.actionClockwise_90 = QtWidgets.QAction(Main_App)
        self.actionClockwise_90.setObjectName("actionClockwise_90")
        self.actionAnticlockwise_90 = QtWidgets.QAction(Main_App)
        self.actionAnticlockwise_90.setObjectName("actionAnticlockwise_90")
        self.action180 = QtWidgets.QAction(Main_App)
        self.action180.setObjectName("action180")
        self.actionAny_Angle = QtWidgets.QAction(Main_App)
        self.actionAny_Angle.setObjectName("actionAny_Angle")
        self.actionPyramid_Up = QtWidgets.QAction(Main_App)
        self.actionPyramid_Up.setObjectName("actionPyramid_Up")
        self.actionPyramid_Down = QtWidgets.QAction(Main_App)
        self.actionPyramid_Down.setObjectName("actionPyramid_Down")
        self.actionBrightness_Contrast = QtWidgets.QAction(Main_App)
        self.actionBrightness_Contrast.setObjectName("actionBrightness_Contrast")
        self.actionInverse = QtWidgets.QAction(Main_App)
        self.actionInverse.setObjectName("actionInverse")
        self.actionSpice_Noise = QtWidgets.QAction(Main_App)
        self.actionSpice_Noise.setObjectName("actionSpice_Noise")
        self.actionGasuss_Noise = QtWidgets.QAction(Main_App)
        self.actionGasuss_Noise.setObjectName("actionGasuss_Noise")
        self.actionThresholding = QtWidgets.QAction(Main_App)
        self.actionThresholding.setObjectName("actionThresholding")
        self.actionBasic = QtWidgets.QAction(Main_App)
        self.actionBasic.setObjectName("actionBasic")
        self.actionAverage = QtWidgets.QAction(Main_App)
        self.actionAverage.setObjectName("actionAverage")
        self.actionMaximum = QtWidgets.QAction(Main_App)
        self.actionMaximum.setObjectName("actionMaximum")
        self.actionSpectrum = QtWidgets.QAction(Main_App)
        self.actionSpectrum.setObjectName("actionSpectrum")
        self.actionEmboss = QtWidgets.QAction(Main_App)
        self.actionEmboss.setObjectName("actionEmboss")
        self.actionClustering = QtWidgets.QAction(Main_App)
        self.actionClustering.setObjectName("actionClustering")
        self.actionGamma = QtWidgets.QAction(Main_App)
        self.actionGamma.setObjectName("actionGamma")
        self.actionLog = QtWidgets.QAction(Main_App)
        self.actionLog.setObjectName("actionLog")
        self.actionRetro = QtWidgets.QAction(Main_App)
        self.actionRetro.setObjectName("actionRetro")
        self.actionHorizontal = QtWidgets.QAction(Main_App)
        self.actionHorizontal.setObjectName("actionHorizontal")
        self.actionVertical = QtWidgets.QAction(Main_App)
        self.actionVertical.setObjectName("actionVertical")
        self.action2bit = QtWidgets.QAction(Main_App)
        self.action2bit.setObjectName("action2bit")
        self.action3bit = QtWidgets.QAction(Main_App)
        self.action3bit.setObjectName("action3bit")
        self.action4bit = QtWidgets.QAction(Main_App)
        self.action4bit.setObjectName("action4bit")
        self.action5bit = QtWidgets.QAction(Main_App)
        self.action5bit.setObjectName("action5bit")
        self.action6bit = QtWidgets.QAction(Main_App)
        self.action6bit.setObjectName("action6bit")
        self.action7bit = QtWidgets.QAction(Main_App)
        self.action7bit.setObjectName("action7bit")
        self.actionMorphology = QtWidgets.QAction(Main_App)
        self.actionMorphology.setObjectName("actionMorphology")
        self.actionRoberts = QtWidgets.QAction(Main_App)
        self.actionRoberts.setObjectName("actionRoberts")
        self.actionPrewitt = QtWidgets.QAction(Main_App)
        self.actionPrewitt.setObjectName("actionPrewitt")
        self.actionSobel = QtWidgets.QAction(Main_App)
        self.actionSobel.setObjectName("actionSobel")
        self.actionScharr = QtWidgets.QAction(Main_App)
        self.actionScharr.setObjectName("actionScharr")
        self.actionCanny = QtWidgets.QAction(Main_App)
        self.actionCanny.setObjectName("actionCanny")
        self.actionLOG = QtWidgets.QAction(Main_App)
        self.actionLOG.setObjectName("actionLOG")
        self.actionLaplacian = QtWidgets.QAction(Main_App)
        self.actionLaplacian.setObjectName("actionLaplacian")
        self.actionResize = QtWidgets.QAction(Main_App)
        self.actionResize.setObjectName("actionResize")
        self.actionPlot_Grey = QtWidgets.QAction(Main_App)
        self.actionPlot_Grey.setObjectName("actionPlot_Grey")
        self.actionPlot_RGB = QtWidgets.QAction(Main_App)
        self.actionPlot_RGB.setObjectName("actionPlot_RGB")
        self.actionEqualization = QtWidgets.QAction(Main_App)
        self.actionEqualization.setObjectName("actionEqualization")
        self.actionRedo = QtWidgets.QAction(Main_App)
        self.actionRedo.setObjectName("actionRedo")
        self.actionBack = QtWidgets.QAction(Main_App)
        self.actionBack.setObjectName("actionBack")
        self.actionBox_Blur = QtWidgets.QAction(Main_App)
        self.actionBox_Blur.setObjectName("actionBox_Blur")
        self.actionGaussian_Blur = QtWidgets.QAction(Main_App)
        self.actionGaussian_Blur.setObjectName("actionGaussian_Blur")
        self.actionMedian_Blur = QtWidgets.QAction(Main_App)
        self.actionMedian_Blur.setObjectName("actionMedian_Blur")
        self.actionLaplacian_Sharpen = QtWidgets.QAction(Main_App)
        self.actionLaplacian_Sharpen.setObjectName("actionLaplacian_Sharpen")
        self.actionUSM_Sharpen = QtWidgets.QAction(Main_App)
        self.actionUSM_Sharpen.setObjectName("actionUSM_Sharpen")
        self.actionIdeal_Filter = QtWidgets.QAction(Main_App)
        self.actionIdeal_Filter.setObjectName("actionIdeal_Filter")
        self.actionCarve = QtWidgets.QAction(Main_App)
        self.actionCarve.setObjectName("actionCarve")
        self.actionGround_Glass = QtWidgets.QAction(Main_App)
        self.actionGround_Glass.setObjectName("actionGround_Glass")
        self.actionSketch = QtWidgets.QAction(Main_App)
        self.actionSketch.setObjectName("actionSketch")
        self.actionOil_Painting = QtWidgets.QAction(Main_App)
        self.actionOil_Painting.setObjectName("actionOil_Painting")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionQuit)
        self.menuRotate.addAction(self.actionClockwise_90)
        self.menuRotate.addAction(self.actionAnticlockwise_90)
        self.menuRotate.addAction(self.action180)
        self.menuRotate.addSeparator()
        self.menuRotate.addAction(self.actionAny_Angle)
        self.menuScaling.addAction(self.actionPyramid_Up)
        self.menuScaling.addAction(self.actionPyramid_Down)
        self.menuScaling.addSeparator()
        self.menuScaling.addAction(self.actionResize)
        self.menuConvert_to_Gray.addAction(self.actionBasic)
        self.menuConvert_to_Gray.addAction(self.actionAverage)
        self.menuConvert_to_Gray.addAction(self.actionMaximum)
        self.menuAdjustment.addAction(self.menuConvert_to_Gray.menuAction())
        self.menuAdjustment.addAction(self.actionBrightness_Contrast)
        self.menuAdjustment.addAction(self.actionInverse)
        self.menuAdjustment.addSeparator()
        self.menuAdjustment.addAction(self.actionGamma)
        self.menuAdjustment.addAction(self.actionLog)
        self.menuFlip.addAction(self.actionHorizontal)
        self.menuFlip.addAction(self.actionVertical)
        self.menuQuantization.addAction(self.action2bit)
        self.menuQuantization.addAction(self.action3bit)
        self.menuQuantization.addAction(self.action4bit)
        self.menuQuantization.addAction(self.action5bit)
        self.menuQuantization.addAction(self.action6bit)
        self.menuQuantization.addAction(self.action7bit)
        self.menuHistogram.addAction(self.actionPlot_Grey)
        self.menuHistogram.addAction(self.actionPlot_RGB)
        self.menuHistogram.addSeparator()
        self.menuHistogram.addAction(self.actionEqualization)
        self.menuImage.addAction(self.menuScaling.menuAction())
        self.menuImage.addAction(self.actionTranslation)
        self.menuImage.addAction(self.menuRotate.menuAction())
        self.menuImage.addAction(self.menuFlip.menuAction())
        self.menuImage.addSeparator()
        self.menuImage.addAction(self.menuAdjustment.menuAction())
        self.menuImage.addAction(self.menuHistogram.menuAction())
        self.menuImage.addSeparator()
        self.menuImage.addAction(self.menuQuantization.menuAction())
        self.menuImage.addAction(self.actionThresholding)
        self.menuImage.addAction(self.actionClustering)
        self.menuNoise.addAction(self.actionSpice_Noise)
        self.menuNoise.addAction(self.actionGasuss_Noise)
        self.menuStylize.addAction(self.actionEmboss)
        self.menuStylize.addAction(self.actionCarve)
        self.menuBlur.addAction(self.actionBox_Blur)
        self.menuBlur.addAction(self.actionGaussian_Blur)
        self.menuBlur.addAction(self.actionMedian_Blur)
        self.menuFT.addAction(self.actionSpectrum)
        self.menuFT.addAction(self.actionIdeal_Filter)
        self.menuSharpen.addAction(self.actionLaplacian_Sharpen)
        self.menuSharpen.addAction(self.actionUSM_Sharpen)
        self.menuSpecial.addAction(self.actionRetro)
        self.menuSpecial.addAction(self.actionGround_Glass)
        self.menuSpecial.addAction(self.actionSketch)
        self.menuSpecial.addAction(self.actionOil_Painting)
        self.menuFind_Edges_2.addAction(self.actionRoberts)
        self.menuFind_Edges_2.addAction(self.actionPrewitt)
        self.menuFind_Edges_2.addAction(self.actionSobel)
        self.menuFind_Edges_2.addAction(self.actionScharr)
        self.menuFind_Edges_2.addAction(self.actionCanny)
        self.menuFind_Edges_2.addAction(self.actionLOG)
        self.menuFind_Edges_2.addAction(self.actionLaplacian)
        self.menuFilter.addAction(self.menuNoise.menuAction())
        self.menuFilter.addAction(self.menuBlur.menuAction())
        self.menuFilter.addAction(self.menuSharpen.menuAction())
        self.menuFilter.addAction(self.menuFT.menuAction())
        self.menuFilter.addAction(self.menuFind_Edges_2.menuAction())
        self.menuFilter.addAction(self.actionMorphology)
        self.menuFilter.addAction(self.menuStylize.menuAction())
        self.menuFilter.addAction(self.menuSpecial.menuAction())
        self.menuAbout.addAction(self.actionSource_Code)
        self.menuAbout.addAction(self.actionAbout_me)
        self.menuEdit.addAction(self.actionRedo)
        self.menuEdit.addAction(self.actionBack)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuImage.menuAction())
        self.menubar.addAction(self.menuFilter.menuAction())
        self.menubar.addAction(self.menuAbout.menuAction())

        self.retranslateUi(Main_App)
        QtCore.QMetaObject.connectSlotsByName(Main_App)

    def retranslateUi(self, Main_App):
        _translate = QtCore.QCoreApplication.translate
        Main_App.setWindowTitle(_translate("Main_App", "What Image"))
        self.menuFile.setTitle(_translate("Main_App", "File"))
        self.menuImage.setTitle(_translate("Main_App", "Image"))
        self.menuRotate.setTitle(_translate("Main_App", "Rotate"))
        self.menuScaling.setTitle(_translate("Main_App", "Scaling"))
        self.menuAdjustment.setTitle(_translate("Main_App", "Adjustment"))
        self.menuConvert_to_Gray.setTitle(_translate("Main_App", "Convert to Gray"))
        self.menuFlip.setTitle(_translate("Main_App", "Flip"))
        self.menuQuantization.setTitle(_translate("Main_App", "Quantization"))
        self.menuHistogram.setTitle(_translate("Main_App", "Histogram"))
        self.menuFilter.setTitle(_translate("Main_App", "Filter"))
        self.menuNoise.setTitle(_translate("Main_App", "Noise"))
        self.menuStylize.setTitle(_translate("Main_App", "Stylize"))
        self.menuBlur.setTitle(_translate("Main_App", "Blur"))
        self.menuFT.setTitle(_translate("Main_App", "FT"))
        self.menuSharpen.setTitle(_translate("Main_App", "Sharpen"))
        self.menuSpecial.setTitle(_translate("Main_App", "Special"))
        self.menuFind_Edges_2.setTitle(_translate("Main_App", "Find Edges"))
        self.menuAbout.setTitle(_translate("Main_App", "About"))
        self.menuEdit.setTitle(_translate("Main_App", "Edit"))
        self.actionOpen.setText(_translate("Main_App", "Open"))
        self.actionOpen.setShortcut(_translate("Main_App", "Ctrl+O"))
        self.actionSave.setText(_translate("Main_App", "Save"))
        self.actionSave.setShortcut(_translate("Main_App", "Ctrl+S"))
        self.actionSave_As.setText(_translate("Main_App", "Save As"))
        self.actionSave_As.setShortcut(_translate("Main_App", "Ctrl+Shift+S"))
        self.actionQuit.setText(_translate("Main_App", "Quit"))
        self.actionQuit.setToolTip(_translate("Main_App", "Quit"))
        self.actionQuit.setShortcut(_translate("Main_App", "Ctrl+Q"))
        self.actionSource_Code.setText(_translate("Main_App", "Source Code"))
        self.actionAbout_me.setText(_translate("Main_App", "About Me"))
        self.actionTranslation.setText(_translate("Main_App", "Translation"))
        self.actionClockwise_90.setText(_translate("Main_App", "Clockwise 90°"))
        self.actionAnticlockwise_90.setText(_translate("Main_App", "Anticlockwise 90°"))
        self.action180.setText(_translate("Main_App", "180°"))
        self.actionAny_Angle.setText(_translate("Main_App", "Any Angle"))
        self.actionPyramid_Up.setText(_translate("Main_App", "Pyramid Up"))
        self.actionPyramid_Up.setToolTip(_translate("Main_App", "Pyramid Up"))
        self.actionPyramid_Down.setText(_translate("Main_App", "Pyramid Down"))
        self.actionPyramid_Down.setToolTip(_translate("Main_App", "Pyramid Down"))
        self.actionBrightness_Contrast.setText(_translate("Main_App", "Brightness/Contrast"))
        self.actionBrightness_Contrast.setToolTip(_translate("Main_App", "Brightness/Contrast"))
        self.actionInverse.setText(_translate("Main_App", "Inverse"))
        self.actionSpice_Noise.setText(_translate("Main_App", "Spice Noise"))
        self.actionGasuss_Noise.setText(_translate("Main_App", "Gasuss Noise"))
        self.actionThresholding.setText(_translate("Main_App", "Thresholding"))
        self.actionBasic.setText(_translate("Main_App", "Basic"))
        self.actionAverage.setText(_translate("Main_App", "Average"))
        self.actionMaximum.setText(_translate("Main_App", "Maximum"))
        self.actionSpectrum.setText(_translate("Main_App", "Spectrum"))
        self.actionEmboss.setText(_translate("Main_App", "Emboss"))
        self.actionClustering.setText(_translate("Main_App", "Clustering"))
        self.actionGamma.setText(_translate("Main_App", "Gamma"))
        self.actionLog.setText(_translate("Main_App", "Log"))
        self.actionRetro.setText(_translate("Main_App", "Retro"))
        self.actionHorizontal.setText(_translate("Main_App", "Horizontal"))
        self.actionVertical.setText(_translate("Main_App", "Vertical"))
        self.action2bit.setText(_translate("Main_App", "2bit(4Levels)"))
        self.action2bit.setToolTip(_translate("Main_App", "2bit(4Levels)"))
        self.action3bit.setText(_translate("Main_App", "3bit(8Levels)"))
        self.action3bit.setToolTip(_translate("Main_App", "3bit(8Levels)"))
        self.action4bit.setText(_translate("Main_App", "4bit(16Levels)"))
        self.action4bit.setToolTip(_translate("Main_App", "4bit(16Levels)"))
        self.action5bit.setText(_translate("Main_App", "5bit(32Levels)"))
        self.action5bit.setToolTip(_translate("Main_App", "5bit(32Levels)"))
        self.action6bit.setText(_translate("Main_App", "6bit(64Levels)"))
        self.action6bit.setToolTip(_translate("Main_App", "6bit(64Levels)"))
        self.action7bit.setText(_translate("Main_App", "7bit(128Levels)"))
        self.action7bit.setToolTip(_translate("Main_App", "7bit(128Levels)"))
        self.actionMorphology.setText(_translate("Main_App", "Morphology"))
        self.actionRoberts.setText(_translate("Main_App", "Roberts"))
        self.actionPrewitt.setText(_translate("Main_App", "Prewitt"))
        self.actionSobel.setText(_translate("Main_App", "Sobel"))
        self.actionScharr.setText(_translate("Main_App", "Scharr"))
        self.actionCanny.setText(_translate("Main_App", "Canny"))
        self.actionLOG.setText(_translate("Main_App", "LOG"))
        self.actionLaplacian.setText(_translate("Main_App", "Laplacian"))
        self.actionResize.setText(_translate("Main_App", "Resize"))
        self.actionPlot_Grey.setText(_translate("Main_App", "Plot Grey"))
        self.actionPlot_RGB.setText(_translate("Main_App", "Plot RGB"))
        self.actionEqualization.setText(_translate("Main_App", "Equalization"))
        self.actionRedo.setText(_translate("Main_App", "Redo"))
        self.actionRedo.setShortcut(_translate("Main_App", "Ctrl+Y"))
        self.actionBack.setText(_translate("Main_App", "Back"))
        self.actionBack.setShortcut(_translate("Main_App", "Ctrl+Z"))
        self.actionBox_Blur.setText(_translate("Main_App", "Box Blur"))
        self.actionGaussian_Blur.setText(_translate("Main_App", "Gaussian Blur"))
        self.actionMedian_Blur.setText(_translate("Main_App", "Median Blur"))
        self.actionLaplacian_Sharpen.setText(_translate("Main_App", "Laplacian Sharpen"))
        self.actionUSM_Sharpen.setText(_translate("Main_App", "USM Sharpen"))
        self.actionIdeal_Filter.setText(_translate("Main_App", "Ideal Filter"))
        self.actionCarve.setText(_translate("Main_App", "Carve"))
        self.actionGround_Glass.setText(_translate("Main_App", "Ground Glass"))
        self.actionSketch.setText(_translate("Main_App", "Sketch"))
        self.actionOil_Painting.setText(_translate("Main_App", "Oil Painting"))

