# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'clustering.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Clustering(object):
    def setupUi(self, Clustering):
        Clustering.setObjectName("Clustering")
        Clustering.resize(236, 109)
        self.centralwidget = QtWidgets.QWidget(Clustering)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 10, 195, 59))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_cluster = QtWidgets.QLabel(self.layoutWidget)
        self.label_cluster.setObjectName("label_cluster")
        self.gridLayout.addWidget(self.label_cluster, 0, 0, 1, 1)
        self.btn_apply = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_apply.setObjectName("btn_apply")
        self.gridLayout.addWidget(self.btn_apply, 1, 0, 1, 1)
        self.btn_cancel = QtWidgets.QPushButton(self.layoutWidget)
        self.btn_cancel.setObjectName("btn_cancel")
        self.gridLayout.addWidget(self.btn_cancel, 1, 1, 1, 1)
        self.num_cluster = QtWidgets.QSpinBox(self.layoutWidget)
        self.num_cluster.setMinimum(2)
        self.num_cluster.setMaximum(32)
        self.num_cluster.setObjectName("num_cluster")
        self.gridLayout.addWidget(self.num_cluster, 0, 1, 1, 1)
        Clustering.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Clustering)
        self.statusbar.setObjectName("statusbar")
        Clustering.setStatusBar(self.statusbar)

        self.retranslateUi(Clustering)
        self.btn_cancel.clicked.connect(Clustering.close)
        QtCore.QMetaObject.connectSlotsByName(Clustering)

    def retranslateUi(self, Clustering):
        _translate = QtCore.QCoreApplication.translate
        Clustering.setWindowTitle(_translate("Clustering", "Clustering"))
        self.label_cluster.setText(_translate("Clustering", "Cluster"))
        self.btn_apply.setText(_translate("Clustering", "Apply"))
        self.btn_cancel.setText(_translate("Clustering", "Cancel"))

