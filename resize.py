# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'resize.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Resize(object):
    def setupUi(self, Resize):
        Resize.setObjectName("Resize")
        Resize.resize(257, 156)
        self.centralwidget = QtWidgets.QWidget(Resize)
        self.centralwidget.setObjectName("centralwidget")
        self.btn_apply_2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_apply_2.setGeometry(QtCore.QRect(30, 90, 93, 28))
        self.btn_apply_2.setObjectName("btn_apply_2")
        self.btn_cancel_2 = QtWidgets.QPushButton(self.centralwidget)
        self.btn_cancel_2.setGeometry(QtCore.QRect(130, 90, 93, 28))
        self.btn_cancel_2.setObjectName("btn_cancel_2")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 10, 201, 70))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_y_input = QtWidgets.QLabel(self.layoutWidget)
        self.label_y_input.setObjectName("label_y_input")
        self.gridLayout.addWidget(self.label_y_input, 1, 0, 1, 1)
        self.label_x_input = QtWidgets.QLabel(self.layoutWidget)
        self.label_x_input.setObjectName("label_x_input")
        self.gridLayout.addWidget(self.label_x_input, 0, 0, 1, 1)
        self.num_x = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.num_x.setMinimum(0.01)
        self.num_x.setSingleStep(0.01)
        self.num_x.setProperty("value", 1.0)
        self.num_x.setObjectName("num_x")
        self.gridLayout.addWidget(self.num_x, 0, 1, 1, 1)
        self.num_y = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.num_y.setMinimum(0.01)
        self.num_y.setSingleStep(0.01)
        self.num_y.setProperty("value", 1.0)
        self.num_y.setObjectName("num_y")
        self.gridLayout.addWidget(self.num_y, 1, 1, 1, 1)
        Resize.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Resize)
        self.statusbar.setObjectName("statusbar")
        Resize.setStatusBar(self.statusbar)

        self.retranslateUi(Resize)
        self.btn_cancel_2.clicked.connect(Resize.close)
        QtCore.QMetaObject.connectSlotsByName(Resize)

    def retranslateUi(self, Resize):
        _translate = QtCore.QCoreApplication.translate
        Resize.setWindowTitle(_translate("Resize", "MainWindow"))
        self.btn_apply_2.setText(_translate("Resize", "Apply"))
        self.btn_cancel_2.setText(_translate("Resize", "Cancel"))
        self.label_y_input.setText(_translate("Resize", "Resized Height"))
        self.label_x_input.setText(_translate("Resize", "Resized Width"))

