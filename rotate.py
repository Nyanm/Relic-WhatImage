# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'rotate.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Rotate(object):
    def setupUi(self, Rotate):
        Rotate.setObjectName("Rotate")
        Rotate.resize(271, 114)
        self.centralwidget = QtWidgets.QWidget(Rotate)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(40, 10, 195, 59))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_angle = QtWidgets.QLabel(self.layoutWidget)
        self.label_angle.setObjectName("label_angle")
        self.gridLayout.addWidget(self.label_angle, 0, 0, 1, 1)
        self.btn_Apply = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_Apply.setObjectName("btn_Apply")
        self.gridLayout.addWidget(self.btn_Apply, 1, 0, 1, 1)
        self.btn_cancel = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_cancel.setObjectName("btn_cancel")
        self.gridLayout.addWidget(self.btn_cancel, 1, 1, 1, 1)
        self.num_angle = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.num_angle.setMinimum(-360.0)
        self.num_angle.setMaximum(360.0)
        self.num_angle.setSingleStep(0.01)
        self.num_angle.setObjectName("num_angle")
        self.gridLayout.addWidget(self.num_angle, 0, 1, 1, 1)
        Rotate.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Rotate)
        self.statusbar.setObjectName("statusbar")
        Rotate.setStatusBar(self.statusbar)

        self.retranslateUi(Rotate)
        self.btn_cancel.clicked.connect(Rotate.close)
        QtCore.QMetaObject.connectSlotsByName(Rotate)

    def retranslateUi(self, Rotate):
        _translate = QtCore.QCoreApplication.translate
        Rotate.setWindowTitle(_translate("Rotate", "MainWindow"))
        self.label_angle.setText(_translate("Rotate", "Angle"))
        self.btn_Apply.setText(_translate("Rotate", "Apply"))
        self.btn_cancel.setText(_translate("Rotate", "Cancel"))

