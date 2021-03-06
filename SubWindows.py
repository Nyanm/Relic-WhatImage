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
