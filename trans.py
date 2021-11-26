# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'trans.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Translation(object):
    def setupUi(self, Translation):
        Translation.setObjectName("Translation")
        Translation.resize(242, 151)
        self.centralwidget = QtWidgets.QWidget(Translation)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 20, 197, 92))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_x = QtWidgets.QLabel(self.layoutWidget)
        self.label_x.setObjectName("label_x")
        self.gridLayout.addWidget(self.label_x, 0, 0, 1, 1)
        self.label_y = QtWidgets.QLabel(self.layoutWidget)
        self.label_y.setObjectName("label_y")
        self.gridLayout.addWidget(self.label_y, 1, 0, 1, 1)
        self.num_x = QtWidgets.QSpinBox(self.layoutWidget)
        self.num_x.setMinimum(-10000)
        self.num_x.setMaximum(10000)
        self.num_x.setObjectName("num_x")
        self.gridLayout.addWidget(self.num_x, 0, 1, 1, 1)
        self.num_y = QtWidgets.QSpinBox(self.layoutWidget)
        self.num_y.setMinimum(-10000)
        self.num_y.setMaximum(10000)
        self.num_y.setObjectName("num_y")
        self.gridLayout.addWidget(self.num_y, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_apply = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_apply.setObjectName("btn_apply")
        self.horizontalLayout.addWidget(self.btn_apply)
        self.btn_cancel = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_cancel.setObjectName("btn_cancel")
        self.horizontalLayout.addWidget(self.btn_cancel)
        self.verticalLayout.addLayout(self.horizontalLayout)
        Translation.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Translation)
        self.statusbar.setObjectName("statusbar")
        Translation.setStatusBar(self.statusbar)

        self.retranslateUi(Translation)
        self.btn_cancel.clicked.connect(Translation.close)
        QtCore.QMetaObject.connectSlotsByName(Translation)

    def retranslateUi(self, Translation):
        _translate = QtCore.QCoreApplication.translate
        Translation.setWindowTitle(_translate("Translation", "Translation"))
        self.label_x.setText(_translate("Translation", "X Axis"))
        self.label_y.setText(_translate("Translation", "Y Axis"))
        self.btn_apply.setText(_translate("Translation", "Apply"))
        self.btn_cancel.setText(_translate("Translation", "Cancel"))

