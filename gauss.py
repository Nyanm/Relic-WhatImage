# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gauss.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Gauss(object):
    def setupUi(self, Gauss):
        Gauss.setObjectName("Gauss")
        Gauss.resize(236, 105)
        self.centralwidget = QtWidgets.QWidget(Gauss)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 10, 209, 59))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_var = QtWidgets.QLabel(self.layoutWidget)
        self.label_var.setObjectName("label_var")
        self.gridLayout.addWidget(self.label_var, 0, 0, 1, 1)
        self.btn_apply = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_apply.setObjectName("btn_apply")
        self.gridLayout.addWidget(self.btn_apply, 1, 0, 1, 1)
        self.btn_cancel = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_cancel.setObjectName("btn_cancel")
        self.gridLayout.addWidget(self.btn_cancel, 1, 1, 1, 1)
        self.num_var = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.num_var.setDecimals(6)
        self.num_var.setMaximum(0.999999)
        self.num_var.setSingleStep(1e-06)
        self.num_var.setObjectName("num_var")
        self.gridLayout.addWidget(self.num_var, 0, 1, 1, 1)
        Gauss.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Gauss)
        self.statusbar.setObjectName("statusbar")
        Gauss.setStatusBar(self.statusbar)

        self.retranslateUi(Gauss)
        self.btn_cancel.clicked.connect(Gauss.close)
        QtCore.QMetaObject.connectSlotsByName(Gauss)

    def retranslateUi(self, Gauss):
        _translate = QtCore.QCoreApplication.translate
        Gauss.setWindowTitle(_translate("Gauss", "Gaussian Noise"))
        self.label_var.setText(_translate("Gauss", "Variance"))
        self.btn_apply.setText(_translate("Gauss", "Apply"))
        self.btn_cancel.setText(_translate("Gauss", "Cancel"))

