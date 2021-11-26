# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'brightness.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Brightness_Contrast(object):
    def setupUi(self, Brightness_Contrast):
        Brightness_Contrast.setObjectName("Brightness_Contrast")
        Brightness_Contrast.resize(341, 213)
        self.centralwidget = QtWidgets.QWidget(Brightness_Contrast)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(30, 130, 281, 30))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(20)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_apply = QtWidgets.QPushButton(self.widget)
        self.btn_apply.setObjectName("btn_apply")
        self.horizontalLayout.addWidget(self.btn_apply)
        self.btn_cancel = QtWidgets.QPushButton(self.widget)
        self.btn_cancel.setObjectName("btn_cancel")
        self.horizontalLayout.addWidget(self.btn_cancel)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(30, 20, 281, 91))
        self.widget1.setObjectName("widget1")
        self.gridLayout = QtWidgets.QGridLayout(self.widget1)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_Brightness = QtWidgets.QLabel(self.widget1)
        self.label_Brightness.setObjectName("label_Brightness")
        self.gridLayout.addWidget(self.label_Brightness, 0, 0, 1, 1)
        self.sld_brightness = QtWidgets.QSlider(self.widget1)
        self.sld_brightness.setMinimum(-100)
        self.sld_brightness.setMaximum(100)
        self.sld_brightness.setOrientation(QtCore.Qt.Horizontal)
        self.sld_brightness.setInvertedAppearance(False)
        self.sld_brightness.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sld_brightness.setTickInterval(20)
        self.sld_brightness.setObjectName("sld_brightness")
        self.gridLayout.addWidget(self.sld_brightness, 0, 1, 1, 1)
        self.label_Contrast = QtWidgets.QLabel(self.widget1)
        self.label_Contrast.setObjectName("label_Contrast")
        self.gridLayout.addWidget(self.label_Contrast, 1, 0, 1, 1)
        self.sld_contrast = QtWidgets.QSlider(self.widget1)
        self.sld_contrast.setMinimum(-100)
        self.sld_contrast.setSliderPosition(0)
        self.sld_contrast.setOrientation(QtCore.Qt.Horizontal)
        self.sld_contrast.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.sld_contrast.setTickInterval(20)
        self.sld_contrast.setObjectName("sld_contrast")
        self.gridLayout.addWidget(self.sld_contrast, 1, 1, 1, 1)
        Brightness_Contrast.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Brightness_Contrast)
        self.statusbar.setObjectName("statusbar")
        Brightness_Contrast.setStatusBar(self.statusbar)

        self.retranslateUi(Brightness_Contrast)
        self.btn_cancel.clicked.connect(Brightness_Contrast.close)
        QtCore.QMetaObject.connectSlotsByName(Brightness_Contrast)

    def retranslateUi(self, Brightness_Contrast):
        _translate = QtCore.QCoreApplication.translate
        Brightness_Contrast.setWindowTitle(_translate("Brightness_Contrast", "Brightness/Contrast"))
        self.btn_apply.setText(_translate("Brightness_Contrast", "Apply"))
        self.btn_cancel.setText(_translate("Brightness_Contrast", "Cancel"))
        self.label_Brightness.setText(_translate("Brightness_Contrast", "Brightness"))
        self.label_Contrast.setText(_translate("Brightness_Contrast", "Contrast"))

