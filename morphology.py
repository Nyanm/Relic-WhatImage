# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'morphology.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Morphology(object):
    def setupUi(self, Morphology):
        Morphology.setObjectName("Morphology")
        Morphology.resize(383, 159)
        self.centralwidget = QtWidgets.QWidget(Morphology)
        self.centralwidget.setObjectName("centralwidget")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(30, 30, 321, 47))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_type = QtWidgets.QLabel(self.widget)
        self.label_type.setObjectName("label_type")
        self.gridLayout.addWidget(self.label_type, 0, 0, 1, 1)
        self.label_kernel_type = QtWidgets.QLabel(self.widget)
        self.label_kernel_type.setObjectName("label_kernel_type")
        self.gridLayout.addWidget(self.label_kernel_type, 0, 1, 1, 1)
        self.label_kernel_size = QtWidgets.QLabel(self.widget)
        self.label_kernel_size.setObjectName("label_kernel_size")
        self.gridLayout.addWidget(self.label_kernel_size, 0, 2, 1, 1)
        self.comboBox_type = QtWidgets.QComboBox(self.widget)
        self.comboBox_type.setObjectName("comboBox_type")
        self.comboBox_type.addItem("")
        self.comboBox_type.addItem("")
        self.comboBox_type.addItem("")
        self.comboBox_type.addItem("")
        self.comboBox_type.addItem("")
        self.comboBox_type.addItem("")
        self.comboBox_type.addItem("")
        self.gridLayout.addWidget(self.comboBox_type, 1, 0, 1, 1)
        self.comboBox_2_kernel_type = QtWidgets.QComboBox(self.widget)
        self.comboBox_2_kernel_type.setObjectName("comboBox_2_kernel_type")
        self.comboBox_2_kernel_type.addItem("")
        self.comboBox_2_kernel_type.addItem("")
        self.comboBox_2_kernel_type.addItem("")
        self.gridLayout.addWidget(self.comboBox_2_kernel_type, 1, 1, 1, 1)
        self.comboBox_kernel_size = QtWidgets.QComboBox(self.widget)
        self.comboBox_kernel_size.setObjectName("comboBox_kernel_size")
        self.comboBox_kernel_size.addItem("")
        self.comboBox_kernel_size.addItem("")
        self.comboBox_kernel_size.addItem("")
        self.comboBox_kernel_size.addItem("")
        self.gridLayout.addWidget(self.comboBox_kernel_size, 1, 2, 1, 1)
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(90, 90, 195, 30))
        self.widget1.setObjectName("widget1")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.btn_apply = QtWidgets.QPushButton(self.widget1)
        self.btn_apply.setObjectName("btn_apply")
        self.horizontalLayout.addWidget(self.btn_apply)
        self.btn_cancel = QtWidgets.QPushButton(self.widget1)
        self.btn_cancel.setObjectName("btn_cancel")
        self.horizontalLayout.addWidget(self.btn_cancel)
        Morphology.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Morphology)
        self.statusbar.setObjectName("statusbar")
        Morphology.setStatusBar(self.statusbar)

        self.retranslateUi(Morphology)
        self.btn_cancel.clicked.connect(Morphology.close)
        QtCore.QMetaObject.connectSlotsByName(Morphology)

    def retranslateUi(self, Morphology):
        _translate = QtCore.QCoreApplication.translate
        Morphology.setWindowTitle(_translate("Morphology", "Morphology"))
        self.label_type.setText(_translate("Morphology", "Type"))
        self.label_kernel_type.setText(_translate("Morphology", "Kernel Type"))
        self.label_kernel_size.setText(_translate("Morphology", "Kernel Size"))
        self.comboBox_type.setItemText(0, _translate("Morphology", "Erosion"))
        self.comboBox_type.setItemText(1, _translate("Morphology", "Dilation"))
        self.comboBox_type.setItemText(2, _translate("Morphology", "Open"))
        self.comboBox_type.setItemText(3, _translate("Morphology", "Close"))
        self.comboBox_type.setItemText(4, _translate("Morphology", "Gradient"))
        self.comboBox_type.setItemText(5, _translate("Morphology", "TopHat"))
        self.comboBox_type.setItemText(6, _translate("Morphology", "BottomHat"))
        self.comboBox_2_kernel_type.setItemText(0, _translate("Morphology", "Square"))
        self.comboBox_2_kernel_type.setItemText(1, _translate("Morphology", "Circle"))
        self.comboBox_2_kernel_type.setItemText(2, _translate("Morphology", "Cross"))
        self.comboBox_kernel_size.setItemText(0, _translate("Morphology", "(3??3)"))
        self.comboBox_kernel_size.setItemText(1, _translate("Morphology", "(5??5)"))
        self.comboBox_kernel_size.setItemText(2, _translate("Morphology", "(7??7)"))
        self.comboBox_kernel_size.setItemText(3, _translate("Morphology", "(9??9)"))
        self.btn_apply.setText(_translate("Morphology", "Apply"))
        self.btn_cancel.setText(_translate("Morphology", "Cancel"))

