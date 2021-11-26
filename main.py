# Main Window
from app import Ui_Main_App
# Sub Windows
from SubWindows import *
# PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
# Other Ingredients
import sys
import webbrowser
import cv2
import numpy as np
import func

img = np.uint8(np.array([[0]]))
img_path = ''


class Backup(object):
    def __init__(self):
        self.stack = []
        self.img_redo = 'flag'

    def add_img(self):
        self.stack.append(img)

    def back(self):
        global img
        if len(self.stack) < 2:
            return
        img = self.stack[-2]
        self.img_redo = self.stack.pop()
        Win_Main.show_image(0)

    def redo(self):
        global img
        if isinstance(self.img_redo, str):
            return
        img = self.img_redo
        Win_Main.show_image()


class MainWindow(QMainWindow, Ui_Main_App):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

    def show_image(self, flag=1):
        global img_path, img
        # Using "img" as global image.
        if len(img.shape) == 3:
            y, x, chn = img.shape
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame = QImage(image.data, x, y, 3 * x, QImage.Format_RGB888)
            # Parameter "bytesPerLine=3 * x" can fix bug of Pyramid  down due to some unknown reasons.
        elif len(img.shape) == 2:
            y, x = img.shape
            image = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            frame = QImage(image.data, x, y, 3 * x, QImage.Format_RGB888)
        else:
            QMessageBox.warning(self, 'File Error', 'Illegal file.')
            return

        if flag:
            CtrlZ.add_img()

        if x <= 1890 and y <= 940:
            self.graphicsView.resize(x + 10, y + 10)
        else:
            self.graphicsView.resize(1890, 940)  # Only optimized for 1080P screen.

        pix = QPixmap.fromImage(frame)
        item = QGraphicsPixmapItem(pix)
        scene = QGraphicsScene()
        scene.addItem(item)
        self.graphicsView.setScene(scene)

    def open_image(self):
        global img_path, img
        img_path, img_type = QFileDialog.getOpenFileName(self, "Select Image", "./", "Image Files(*.jpg *.png)")
        if not img_path:   # If the user just closes the window.
            return
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        CtrlZ.stack = []
        func.plt.cla()
        self.show_image()

    def save_image(self):
        global img_path, img
        if img_path == '':
            return
        cv2.imwrite(img_path, img)

    def save_image_as(self):
        global img_path, img
        if img_path == '':
            return
        save_data = QFileDialog.getSaveFileName(self, "Save Image As", "./", "Image Files(*.jpg *.png)")
        cv2.imwrite(save_data[0], img)

    def pyramid_down(self):
        global img_path, img
        if img_path == '':
            return
        img = func.pyramid_zoom(img)
        self.show_image()

    def pyramid_up(self):
        global img_path, img
        if img_path == '':
            return
        img = func.pyramid_zoom(img, 1)
        self.show_image()

    def rotate_clockwise_90(self):
        global img_path, img
        if img_path == '':
            return
        img = func.central_rotate(img, 90)
        self.show_image()

    def rotate_anticlockwise_90(self):
        global img_path, img
        if img_path == '':
            return
        img = func.central_rotate(img, -90)
        self.show_image()

    def rotate_180(self):
        global img_path, img
        if img_path == '':
            return
        img = func.central_rotate(img, 180)
        self.show_image()

    def flip_horizontal(self):
        global img_path, img
        if img_path == '':
            return
        img = func.flip(img, 0)
        self.show_image()

    def flip_vertical(self):
        global img_path, img
        if img_path == '':
            return
        img = func.flip(img, 1)
        self.show_image()

    def rgb2gray_basic(self):
        global img_path, img
        if img_path == '':
            return
        if len(img.shape) == 2:
            QMessageBox.warning(self, 'Channel Error', 'Convert requires 3 channels. Only find 2 channel.')
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.show_image()

    def rgb2gray_average(self):
        global img_path, img
        if img_path == '':
            return
        if len(img.shape) == 2:
            QMessageBox.warning(self, 'Channel Error', 'Convert requires 3 channels. Only find 2 channel.')
            return
        img = func.convert_color(img, 'ave')
        self.show_image()

    def rgb2gray_maximum(self):
        global img_path, img
        if img_path == '':
            return
        if len(img.shape) == 2:
            QMessageBox.warning(self, 'Channel Error', 'Convert requires 3 channels. Only find 2 channel.')
            return
        img = func.convert_color(img, 'max')
        self.show_image()

    def color_inverse(self):
        global img_path, img
        if img_path == '':
            return
        img = func.liner_scale(img, 255, -1)
        self.show_image()

    def color_gamma(self):
        global img_path, img
        if img_path == '':
            return
        img = func.non_liner_scale(img, 'gamma', 0.00000005, 4.0)
        self.show_image()

    def color_log(self):
        global img_path, img
        if img_path == '':
            return
        img = func.non_liner_scale(img, 'gamma', 42)
        self.show_image()

    def histogram_plot_grey(self):
        global img_path, img
        if img_path == '':
            return
        func.plot_gray_histogram(img)

    def histogram_plot_rgb(self):
        global img_path, img
        if img_path == '':
            return
        if len(img.shape) == 2:
            QMessageBox.warning(self, 'Channel Error', 'Require RGB picture.')
            return
        func.plot_rgb_histogram(img)

    def histogram_eq(self):
        global img_path, img
        if img_path == '':
            return
        img = func.equalization_normal(img)
        self.show_image()

    def quantization_2(self):
        global img_path, img
        if img_path == '':
            return
        img = func.quantization(img, 2)
        self.show_image()

    def quantization_3(self):
        global img_path, img
        if img_path == '':
            return
        img = func.quantization(img, 3)
        self.show_image()

    def quantization_4(self):
        global img_path, img
        if img_path == '':
            return
        img = func.quantization(img, 4)
        self.show_image()

    def quantization_5(self):
        global img_path, img
        if img_path == '':
            return
        img = func.quantization(img, 5)
        self.show_image()

    def quantization_6(self):
        global img_path, img
        if img_path == '':
            return
        img = func.quantization(img, 6)
        self.show_image()

    def quantization_7(self):
        global img_path, img
        if img_path == '':
            return
        img = func.quantization(img, 7)
        self.show_image()

    def blur_box(self):
        global img_path, img
        if img_path == '':
            return
        img = func.blur(img, 'box')
        self.show_image()

    def blur_gaussian(self):
        global img_path, img
        if img_path == '':
            return
        img = func.blur(img, 'gauss')
        self.show_image()

    def blur_median(self):
        global img_path, img
        if img_path == '':
            return
        img = func.blur(img, 'median')
        self.show_image()

    def sharpen_laplacian(self):
        global img_path, img
        if img_path == '':
            return
        img = func.sharpen(img, 'lap')
        self.show_image()

    def sharpen_USM(self):
        global img_path, img
        if img_path == '':
            return
        img = func.sharpen(img, 'USM')
        self.show_image()

    def ft_spectrum(self):
        global img_path, img
        if img_path == '':
            return
        func.get_spectrum(img)

    def find_edges_roberts(self):
        global img_path, img
        if img_path == '':
            return
        img = func.find_edges(img, 'roberts')
        self.show_image()

    def find_edges_prewitt(self):
        global img_path, img
        if img_path == '':
            return
        img = func.find_edges(img, 'prewitt')
        self.show_image()

    def find_edges_sobel(self):
        global img_path, img
        if img_path == '':
            return
        img = func.find_edges(img, 'sobel')
        self.show_image()

    def find_edges_scharr(self):
        global img_path, img
        if img_path == '':
            return
        img = func.find_edges(img, 'scharr')
        self.show_image()

    def find_edges_canny(self):
        global img_path, img
        if img_path == '':
            return
        img = func.find_edges(img, 'canny')
        self.show_image()

    def find_edges_LOG(self):
        global img_path, img
        if img_path == '':
            return
        img = func.find_edges(img, 'log')
        self.show_image()

    def find_edges_laplacian(self):
        global img_path, img
        if img_path == '':
            return
        img = func.find_edges(img, 'laplacian')
        self.show_image()

    def stylize_emboss(self):
        global img_path, img
        if img_path == '':
            return
        img = func.stylize(img, 'emboss')
        self.show_image()

    def stylize_carve(self):
        global img_path, img
        if img_path == '':
            return
        img = func.stylize(img, 'carve')
        self.show_image()

    def special_retro(self):
        global img_path, img
        if img_path == '':
            return
        img = func.special(img, 'retro')
        self.show_image()

    def special_ground_glass(self):
        global img_path, img
        if img_path == '':
            return
        img = func.special(img, 'glass')
        self.show_image()

    def special_sketch(self):
        global img_path, img
        if img_path == '':
            return
        img = func.special(img, 'sketch')
        self.show_image()

    def special_oil_painting(self):
        global img_path, img
        if img_path == '':
            return
        img = func.special(img, 'oil')
        self.show_image()

    def source_code(self):
        webbrowser.open('https://github.com/Nyanm/Relic-WhatImage')

    def about_me(self):
        QMessageBox.information(self, 'About Me', 'Presented by Nyanm, UESTC. \nHope you enjoy it.')


class TransWindow(QMainWindow, Ui_Translation):
    def __init__(self):
        super(TransWindow, self).__init__()
        self.setupUi(self)
        self.events()

    def events(self):
        self.btn_apply.clicked.connect(self.transmission)

    def transmission(self):
        global img_path, img
        if img_path == '':
            return
        x = np.float32(self.num_x.value())
        y = np.float32(self.num_y.value())
        img = func.central_transition(img, x, y)
        MainWindow.show_image(Win_Main)
        self.close()


class Resize(QMainWindow, Ui_Resize):
    def __init__(self):
        super(Resize, self).__init__()
        self.setupUi(self)
        self.events()

    def events(self):
        self.btn_apply_2.clicked.connect(self.image_resize)

    def image_resize(self):
        global img_path, img
        if img_path == '':
            return
        x = np.float32(self.num_x.value())
        y = np.float32(self.num_y.value())
        img = func.image_resize(img, x, y)
        MainWindow.show_image(Win_Main)
        self.close()


class Rotate(QMainWindow, Ui_Rotate):
    def __init__(self):
        super(Rotate, self).__init__()
        self.setupUi(self)
        self.events()

    def events(self):
        self.btn_Apply.clicked.connect(self.any_angle)

    def any_angle(self):
        global img_path, img
        if img_path == '':
            return
        angle = np.float32(self.num_angle.value())
        img = func.central_rotate(img, angle)
        MainWindow.show_image(Win_Main)
        self.close()


class BrightnessContrast(QMainWindow, Ui_Brightness_Contrast):
    def __init__(self):
        super(BrightnessContrast, self).__init__()
        self.setupUi(self)
        self.events()

    def events(self):
        self.btn_apply.clicked.connect(self.adjust)

    def adjust(self):
        global img_path, img
        if img_path == '':
            return
        brightness = self.sld_brightness.value()
        contrast = self.sld_contrast.value()
        img = func.bri_con_adjustment(img, brightness, contrast)
        MainWindow.show_image(Win_Main)
        self.close()


class Thresholding(QMainWindow, Ui_Thresholding):
    def __init__(self):
        super(Thresholding, self).__init__()
        self.setupUi(self)
        self.events()

    def events(self):
        self.btn_apply.clicked.connect(self.threshold)

    def threshold(self):
        global img_path, img
        if img_path == '':
            return
        thr_type = self.comboBox_type.currentText()
        thr = self.num_threshold.value()
        mxm = self.num_maximum.value()
        if thr >= mxm:
            QMessageBox.warning(self, 'Value Error', 'Maximum value must greater than threshold value.')
            return
        img = func.thresholding(img, thr_type, thr, mxm)
        MainWindow.show_image(Win_Main)
        self.close()


class Clustering(QMainWindow, Ui_Clustering):
    def __init__(self):
        super(Clustering, self).__init__()
        self.setupUi(self)
        self.events()

    def events(self):
        self.btn_apply.clicked.connect(self.cluster)

    def cluster(self):
        global img_path, img
        if img_path == '':
            return
        num = self.num_cluster.value()
        if num > 64:
            QMessageBox.warning(self, 'Value Error', 'Too much cluster may break the program down.')
            return
        elif num > 32:
            QMessageBox.warning(self, 'Value Warning', 'Too much cluster may take a long time to calculate.')
            pass
        img = func.clustering(img, num)
        MainWindow.show_image(Win_Main)
        self.close()


class SpiceNoise(QMainWindow, Ui_Spice):
    def __init__(self):
        super(SpiceNoise, self).__init__()
        self.setupUi(self)
        self.events()

    def events(self):
        self.btn_apply.clicked.connect(self.noisy)

    def noisy(self):
        global img_path, img
        if img_path == '':
            return
        prob = np.float32(self.num_prob.value())
        if prob >= 1 or prob <= 0:
            QMessageBox.warning(self, 'Value Warning', 'Probability must greater than 0 and less than 1.')
            return
        img = func.spice_noise(img, prob)
        MainWindow.show_image(Win_Main)
        self.close()


class GaussianNoise(QMainWindow, Ui_Gauss):
    def __init__(self):
        super(GaussianNoise, self).__init__()
        self.setupUi(self)
        self.events()

    def events(self):
        self.btn_apply.clicked.connect(self.noisier)

    def noisier(self):
        global img_path, img
        if img_path == '':
            return
        var = np.float32(self.num_var.value())
        if var <= 0 or var >= 1:
            QMessageBox.warning(self, 'Value Warning', 'Variance must greater than 0 and less than 1.')
            return
        img = func.gasuss_noise(img, var)
        MainWindow.show_image(Win_Main)
        self.close()


class IdealFilter(QMainWindow, Ui_Ideal_Filter):
    def __init__(self):
        super(IdealFilter, self).__init__()
        self.setupUi(self)
        self.events()

    def events(self):
        self.btn_apply.clicked.connect(self.NoIdeal)

    def NoIdeal(self):
        global img_path, img
        if img_path == '':
            return
        filter_type = self.comboBox_type.currentText()
        cover = np.float32(self.num_per.value())
        img = func.ideal_filter(img, filter_type, cover)
        MainWindow.show_image(Win_Main)
        self.close()


class Morphology(QMainWindow, Ui_Morphology):
    def __init__(self):
        super(Morphology, self).__init__()
        self.setupUi(self)
        self.events()

    def events(self):
        self.btn_apply.clicked.connect(self.hat_not_found)

    def hat_not_found(self):
        global img_path, img
        if img_path == '':
            return
        morphology_type = self.comboBox_type.currentText()
        kernel_type = self.comboBox_2_kernel_type.currentText()
        kernel_size = self.comboBox_kernel_size.currentText()
        img = func.morphology(img, morphology_type, kernel_type, kernel_size)
        MainWindow.show_image(Win_Main)
        self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    CtrlZ = Backup()
    Win_Main = MainWindow()
    Win_Translation = TransWindow()
    Win_Resize = Resize()
    Win_Rotate = Rotate()
    Win_Brightness_Contrast = BrightnessContrast()
    Win_Thresholding = Thresholding()
    Win_Clustering = Clustering()
    Win_Spice = SpiceNoise()
    Win_Gauss = GaussianNoise()
    Win_Filter = IdealFilter()
    Win_Morphology = Morphology()

    #            File Execution
    Win_Main.actionOpen.triggered.connect(Win_Main.open_image)
    Win_Main.actionQuit.triggered.connect(qApp.quit)
    Win_Main.actionSave.triggered.connect(Win_Main.save_image)
    Win_Main.actionSave_As.triggered.connect(Win_Main.save_image_as)

    #            Edit Execution
    Win_Main.actionBack.triggered.connect(CtrlZ.back)
    Win_Main.actionRedo.triggered.connect(CtrlZ.redo)

    #            Image Execution
    #        ----Scaling Execution
    Win_Main.actionPyramid_Up.triggered.connect(Win_Main.pyramid_up)
    Win_Main.actionPyramid_Down.triggered.connect(Win_Main.pyramid_down)
    Win_Main.actionResize.triggered.connect(Win_Resize.show)

    Win_Main.actionTranslation.triggered.connect(Win_Translation.show)
    #        ----Rotate Execution
    Win_Main.actionClockwise_90.triggered.connect(Win_Main.rotate_clockwise_90)
    Win_Main.actionAnticlockwise_90.triggered.connect(Win_Main.rotate_anticlockwise_90)
    Win_Main.action180.triggered.connect(Win_Main.rotate_180)
    Win_Main.actionAny_Angle.triggered.connect(Win_Rotate.show)

    #        ----Flip Execution
    Win_Main.actionHorizontal.triggered.connect(Win_Main.flip_horizontal)
    Win_Main.actionVertical.triggered.connect(Win_Main.flip_vertical)

    #        ----Adjustment Execution
    #    --------Convert to Grey Execution
    Win_Main.actionBasic.triggered.connect(Win_Main.rgb2gray_basic)
    Win_Main.actionAverage.triggered.connect(Win_Main.rgb2gray_average)
    Win_Main.actionMaximum.triggered.connect(Win_Main.rgb2gray_maximum)

    Win_Main.actionInverse.triggered.connect(Win_Main.color_inverse)
    Win_Main.actionBrightness_Contrast.triggered.connect(Win_Brightness_Contrast.show)
    Win_Main.actionGamma.triggered.connect(Win_Main.color_gamma)
    Win_Main.actionLog.triggered.connect(Win_Main.color_log)
    #        ----Histogram Execution
    Win_Main.actionPlot_Grey.triggered.connect(Win_Main.histogram_plot_grey)
    Win_Main.actionPlot_RGB.triggered.connect(Win_Main.histogram_plot_rgb)
    Win_Main.actionEqualization.triggered.connect(Win_Main.histogram_eq)

    #        ----Quantization Execution
    Win_Main.action2bit.triggered.connect(Win_Main.quantization_2)
    Win_Main.action3bit.triggered.connect(Win_Main.quantization_3)
    Win_Main.action4bit.triggered.connect(Win_Main.quantization_4)
    Win_Main.action5bit.triggered.connect(Win_Main.quantization_5)
    Win_Main.action6bit.triggered.connect(Win_Main.quantization_6)
    Win_Main.action7bit.triggered.connect(Win_Main.quantization_7)

    Win_Main.actionThresholding.triggered.connect(Win_Thresholding.show)
    Win_Main.actionClustering.triggered.connect(Win_Clustering.show)

    #            Filter Execution
    #        ----Noise Execution
    Win_Main.actionSpice_Noise.triggered.connect(Win_Spice.show)
    Win_Main.actionGasuss_Noise.triggered.connect(Win_Gauss.show)

    #        ----Blur Execution
    Win_Main.actionBox_Blur.triggered.connect(Win_Main.blur_box)
    Win_Main.actionGaussian_Blur.triggered.connect(Win_Main.blur_gaussian)
    Win_Main.actionMedian_Blur.triggered.connect(Win_Main.blur_median)

    #        ----Sharpen Execution
    Win_Main.actionLaplacian_Sharpen.triggered.connect(Win_Main.sharpen_laplacian)
    Win_Main.actionUSM_Sharpen.triggered.connect(Win_Main.sharpen_USM)

    #        ----FT Execution
    Win_Main.actionSpectrum.triggered.connect(Win_Main.ft_spectrum)
    Win_Main.actionIdeal_Filter.triggered.connect(Win_Filter.show)

    #        ----Find Edges Execution
    Win_Main.actionRoberts.triggered.connect(Win_Main.find_edges_roberts)
    Win_Main.actionPrewitt.triggered.connect(Win_Main.find_edges_prewitt)
    Win_Main.actionSobel.triggered.connect(Win_Main.find_edges_sobel)
    Win_Main.actionScharr.triggered.connect(Win_Main.find_edges_scharr)
    Win_Main.actionCanny.triggered.connect(Win_Main.find_edges_canny)
    Win_Main.actionLOG.triggered.connect(Win_Main.find_edges_LOG)
    Win_Main.actionLaplacian.triggered.connect(Win_Main.find_edges_laplacian)

    Win_Main.actionMorphology.triggered.connect(Win_Morphology.show)
    #        ----Stylize Execution
    Win_Main.actionEmboss.triggered.connect(Win_Main.stylize_emboss)
    Win_Main.actionCarve.triggered.connect(Win_Main.stylize_carve)

    #        ----Special Execution
    Win_Main.actionRetro.triggered.connect(Win_Main.special_retro)
    Win_Main.actionGround_Glass.triggered.connect(Win_Main.special_ground_glass)
    Win_Main.actionSketch.triggered.connect(Win_Main.special_sketch)
    Win_Main.actionOil_Painting.triggered.connect(Win_Main.special_oil_painting)

    #            About Execution
    Win_Main.actionSource_Code.triggered.connect(Win_Main.source_code)
    Win_Main.actionAbout_me.triggered.connect(Win_Main.about_me)

    Win_Main.show()
    sys.exit(app.exec_())
