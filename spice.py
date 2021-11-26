# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'spice.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Spice(object):
    def setupUi(self, Spice):
        Spice.setObjectName("Spice")
        Spice.resize(236, 105)
        self.centralwidget = QtWidgets.QWidget(Spice)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 10, 195, 59))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_prob = QtWidgets.QLabel(self.layoutWidget)
        self.label_prob.setObjectName("label_prob")
        self.gridLayout.addWidget(self.label_prob, 0, 0, 1, 1)
        self.btn_apply = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_apply.setObjectName("btn_apply")
        self.gridLayout.addWidget(self.btn_apply, 1, 0, 1, 1)
        self.btn_cancel = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_cancel.setObjectName("btn_cancel")
        self.gridLayout.addWidget(self.btn_cancel, 1, 1, 1, 1)
        self.num_prob = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.num_prob.setDecimals(6)
        self.num_prob.setMaximum(0.999999)
        self.num_prob.setSingleStep(1e-06)
        self.num_prob.setObjectName("num_prob")
        self.gridLayout.addWidget(self.num_prob, 0, 1, 1, 1)
        Spice.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Spice)
        self.statusbar.setObjectName("statusbar")
        Spice.setStatusBar(self.statusbar)

        self.retranslateUi(Spice)
        self.btn_cancel.clicked.connect(Spice.close)
        QtCore.QMetaObject.connectSlotsByName(Spice)

    def retranslateUi(self, Spice):
        _translate = QtCore.QCoreApplication.translate
        Spice.setWindowTitle(_translate("Spice", "Spice Noise"))
        self.label_prob.setText(_translate("Spice", "Probability"))
        self.btn_apply.setText(_translate("Spice", "Apply"))
        self.btn_cancel.setText(_translate("Spice", "Cancel"))

