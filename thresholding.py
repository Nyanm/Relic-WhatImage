# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'thresholding.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Thresholding(object):
    def setupUi(self, Thresholding):
        Thresholding.setObjectName("Thresholding")
        Thresholding.resize(254, 186)
        self.centralwidget = QtWidgets.QWidget(Thresholding)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(30, 20, 201, 117))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_Type = QtWidgets.QLabel(self.widget)
        self.label_Type.setObjectName("label_Type")
        self.gridLayout.addWidget(self.label_Type, 0, 0, 1, 1)
        self.btn_apply = QtWidgets.QPushButton(self.widget)
        self.btn_apply.setObjectName("btn_apply")
        self.gridLayout.addWidget(self.btn_apply, 3, 0, 1, 2)
        self.btn_cancel = QtWidgets.QPushButton(self.widget)
        self.btn_cancel.setObjectName("btn_cancel")
        self.gridLayout.addWidget(self.btn_cancel, 3, 2, 1, 1)
        self.comboBox_type = QtWidgets.QComboBox(self.widget)
        self.comboBox_type.setObjectName("comboBox_type")
        self.comboBox_type.addItem("")
        self.comboBox_type.addItem("")
        self.comboBox_type.addItem("")
        self.comboBox_type.addItem("")
        self.comboBox_type.addItem("")
        self.gridLayout.addWidget(self.comboBox_type, 0, 1, 1, 2)
        self.num_threshold = QtWidgets.QSpinBox(self.widget)
        self.num_threshold.setMaximum(254)
        self.num_threshold.setObjectName("num_threshold")
        self.gridLayout.addWidget(self.num_threshold, 1, 2, 1, 1)
        self.num_maximum = QtWidgets.QSpinBox(self.widget)
        self.num_maximum.setMaximum(255)
        self.num_maximum.setObjectName("num_maximum")
        self.gridLayout.addWidget(self.num_maximum, 2, 2, 1, 1)
        self.label_Threhold = QtWidgets.QLabel(self.widget)
        self.label_Threhold.setObjectName("label_Threhold")
        self.gridLayout.addWidget(self.label_Threhold, 1, 0, 1, 2)
        self.label_Maximum = QtWidgets.QLabel(self.widget)
        self.label_Maximum.setObjectName("label_Maximum")
        self.gridLayout.addWidget(self.label_Maximum, 2, 0, 1, 2)
        Thresholding.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Thresholding)
        self.statusbar.setObjectName("statusbar")
        Thresholding.setStatusBar(self.statusbar)

        self.retranslateUi(Thresholding)
        self.btn_cancel.clicked.connect(Thresholding.close)
        QtCore.QMetaObject.connectSlotsByName(Thresholding)

    def retranslateUi(self, Thresholding):
        _translate = QtCore.QCoreApplication.translate
        Thresholding.setWindowTitle(_translate("Thresholding", "Thresholding"))
        self.label_Type.setText(_translate("Thresholding", "Type"))
        self.btn_apply.setText(_translate("Thresholding", "Apply"))
        self.btn_cancel.setText(_translate("Thresholding", "Cancel"))
        self.comboBox_type.setItemText(0, _translate("Thresholding", "Binary"))
        self.comboBox_type.setItemText(1, _translate("Thresholding", "Inverse Binary"))
        self.comboBox_type.setItemText(2, _translate("Thresholding", "Trunc"))
        self.comboBox_type.setItemText(3, _translate("Thresholding", "Tozero"))
        self.comboBox_type.setItemText(4, _translate("Thresholding", "Inverse Tozero"))
        self.label_Threhold.setText(_translate("Thresholding", "Threshold"))
        self.label_Maximum.setText(_translate("Thresholding", "Maximum"))

