# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'filter.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Ideal_Filter(object):
    def setupUi(self, Ideal_Filter):
        Ideal_Filter.setObjectName("Ideal_Filter")
        Ideal_Filter.resize(262, 148)
        self.centralwidget = QtWidgets.QWidget(Ideal_Filter)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(30, 20, 201, 88))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_perc = QtWidgets.QLabel(self.layoutWidget)
        self.label_perc.setObjectName("label_perc")
        self.gridLayout.addWidget(self.label_perc, 1, 0, 1, 2)
        self.btn_apply = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_apply.setObjectName("btn_apply")
        self.gridLayout.addWidget(self.btn_apply, 2, 0, 1, 2)
        self.btn_cancel = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_cancel.setObjectName("btn_cancel")
        self.gridLayout.addWidget(self.btn_cancel, 2, 2, 1, 1)
        self.num_per = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.num_per.setMaximum(0.99)
        self.num_per.setSingleStep(0.01)
        self.num_per.setObjectName("num_per")
        self.gridLayout.addWidget(self.num_per, 1, 2, 1, 1)
        self.comboBox_type = QtWidgets.QComboBox(self.layoutWidget)
        self.comboBox_type.setObjectName("comboBox_type")
        self.comboBox_type.addItem("")
        self.comboBox_type.addItem("")
        self.gridLayout.addWidget(self.comboBox_type, 0, 2, 1, 1)
        self.label_type = QtWidgets.QLabel(self.layoutWidget)
        self.label_type.setObjectName("label_type")
        self.gridLayout.addWidget(self.label_type, 0, 0, 1, 2)
        Ideal_Filter.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Ideal_Filter)
        self.statusbar.setObjectName("statusbar")
        Ideal_Filter.setStatusBar(self.statusbar)

        self.retranslateUi(Ideal_Filter)
        QtCore.QMetaObject.connectSlotsByName(Ideal_Filter)

    def retranslateUi(self, Ideal_Filter):
        _translate = QtCore.QCoreApplication.translate
        Ideal_Filter.setWindowTitle(_translate("Ideal_Filter", "Ideal Filter"))
        self.label_perc.setText(_translate("Ideal_Filter", "Percentage"))
        self.btn_apply.setText(_translate("Ideal_Filter", "Apply"))
        self.btn_cancel.setText(_translate("Ideal_Filter", "Cancel"))
        self.comboBox_type.setItemText(0, _translate("Ideal_Filter", "HPF"))
        self.comboBox_type.setItemText(1, _translate("Ideal_Filter", "LPF"))
        self.label_type.setText(_translate("Ideal_Filter", "Filter Type"))

