import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random as rd


def quick_image(img, img_name='Demo'):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def central_transition(img, x, y):
    M = np.float32([[1, 0, x], [0, 1, -y]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


def pyramid_zoom(img, up=0):
    if up:
        return cv2.pyrUp(img)
    else:
        return cv2.pyrDown(img)


def image_resize(img, per_x, per_y):
    new_x = int(img.shape[1] * per_x)
    new_y = int(img.shape[0] * per_y)
    return cv2.resize(img, (new_x, new_y))


def central_rotate(img, angle):
    y, x = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((y / 2, x / 2), angle, 1)
    return cv2.warpAffine(img, M, (y, x))


def flip(img, flag):
    if flag:
        return cv2.flip(img, 0)
    else:
        return cv2.flip(img, 1)


def convert_color(img, flag):
    y, x, chn = img.shape
    if flag == 'max':
        gray = np.zeros((y, x), np.uint8)
        for i in range(y):
            for j in range(x):
                val = max(img[i, j][0], img[i, j][1], img[i, j][2])
                gray[i, j] = np.uint8(val)
        return gray
    elif flag == 'ave':
        img = np.float32(img)
        b, g, r = cv2.split(img)
        return np.uint8((b + g + r) / 3)


def liner_scale(img, shift=0, times=1.0):
    return np.uint8(np.clip((np.float32(img) * times + shift), 0, 255))


def non_liner_scale(img, t_type, val, gamma=1):
    img = np.float32(img)
    if t_type == "log":
        return np.uint8(np.clip(val * np.log(1.0 + img), 0, 255))
    # val推荐值为10^-7
    elif t_type == 'gamma':
        return np.uint8(np.clip(val * img ** gamma, 0, 255))


def plot_gray_histogram(img):
    plt.hist(img.ravel(), 256)
    plt.savefig('histogram.png')
    histogram = cv2.imread('histogram.png')
    quick_image(histogram, 'Press Any Key to Quit')
    os.remove('histogram.png')


def plot_rgb_histogram(img):
    hist_b = cv2.calcHist(img, [0], None, [256], [0, 255])
    hist_g = cv2.calcHist(img, [1], None, [256], [0, 255])
    hist_r = cv2.calcHist(img, [2], None, [256], [0, 255])

    plt.plot(hist_b, color='b')
    plt.plot(hist_g, color='g')
    plt.plot(hist_r, color='r')

    plt.savefig('histogram.png')
    histogram = cv2.imread('histogram.png')
    quick_image(histogram, 'Press Any Key to Quit')
    os.remove('histogram.png')


def equalization_normal(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    elif len(img.shape) == 3:
        chn_b, chn_g, chn_r = cv2.split(img)
        bH = cv2.equalizeHist(chn_b)
        gH = cv2.equalizeHist(chn_g)
        rH = cv2.equalizeHist(chn_r)
        return cv2.merge((bH, gH, rH))


def quantization(img, k):
    return np.uint8(np.round(np.float32(img) / (255 / (k - 1))) * (255 / (k - 1)))


def blur(img, flag):
    if flag == 'box':
        return cv2.blur(img, (3, 3))
    elif flag == 'gauss':
        return cv2.GaussianBlur(img, (3, 3), 0)
    elif flag == 'median':
        return cv2.medianBlur(img, 3)


def sharpen(img, flag):
    if flag == 'lap':
        kernel_sharpen = np.array((
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]), dtype="float32")
        return cv2.filter2D(img, -1, kernel_sharpen)
    elif flag == 'USM':
        blur_img = cv2.GaussianBlur(img, (0, 0), 5)
        return cv2.addWeighted(img, 1.5, blur_img, -0.5, 0)


def fft_to_spectrum(img):
    spectrum = np.fft.fftshift(np.fft.fft2(img))
    spectrum = np.log(np.abs(spectrum))
    min_num, max_num = np.min(spectrum), np.max(spectrum)
    return np.uint8(255 / (max_num - min_num) * (spectrum - min_num))


def get_spectrum(img):
    if len(img.shape) == 2:
        spectrum = fft_to_spectrum(img)
        quick_image(spectrum, 'Spectrum:Press Any Key to Quit')
    elif len(img.shape) == 3:
        b_chn, g_chn, r_chn = cv2.split(img)
        b_spectrum = fft_to_spectrum(b_chn)
        g_spectrum = fft_to_spectrum(g_chn)
        r_spectrum = fft_to_spectrum(r_chn)
        cv2.imshow('Blue Channel Spectrum', b_spectrum)
        cv2.imshow('Green Channel Spectrum', g_spectrum)
        cv2.imshow('Red Channel Spectrum', r_spectrum)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def find_edges(img, flag):
    if flag == 'roberts':
        kernel_x = np.array([[-1, 0], [0, 1]], dtype=int)
        kernel_y = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv2.filter2D(img, cv2.CV_16S, kernel_x)
        y = cv2.filter2D(img, cv2.CV_16S, kernel_y)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    elif flag == 'prewitt':
        kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        x = cv2.filter2D(img, cv2.CV_16S, kernel_x)
        y = cv2.filter2D(img, cv2.CV_16S, kernel_y)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    elif flag == 'sobel':
        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    elif flag == 'scharr':
        x = cv2.Scharr(img, cv2.CV_32F, 1, 0)
        y = cv2.Scharr(img, cv2.CV_32F, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    elif flag == 'canny':
        return cv2.Canny(cv2.GaussianBlur(img, (3, 3), 0), 50, 150)
    elif flag == 'log':
        return cv2.convertScaleAbs(cv2.Laplacian(cv2.GaussianBlur(img, (3, 3), 0), cv2.CV_16S, ksize=3))
    elif flag == 'laplacian':
        return cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_16S, ksize=3))


def stylize(img, flag):
    if flag == 'emboss':
        kernel_emboss = np.array((
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]), dtype="float32")
        return cv2.filter2D(img, -1, kernel_emboss)
    elif flag == 'carve':
        kernel_carve = np.array((
            [2, 1, 0],
            [1, 1, -1],
            [0, -1, -2]), dtype="float32")
        return cv2.filter2D(img, -1, kernel_carve)


def special(img, flag):
    if flag == 'retro':
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = np.float32(img)
        b_chn, g_chn, r_chn = cv2.split(img)
        r_retro = np.uint8(np.clip((0.393 * r_chn + 0.769 * g_chn + 0.189 * b_chn), 0, 255))
        g_retro = np.uint8(np.clip((0.349 * r_chn + 0.686 * g_chn + 0.168 * b_chn), 0, 255))
        b_retro = np.uint8(np.clip((0.272 * r_chn + 0.534 * g_chn + 0.131 * b_chn), 0, 255))
        return cv2.merge([b_retro, g_retro, r_retro])
    elif flag == 'sketch':
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_inv = 255 - img
        img_blur = cv2.GaussianBlur(img_inv, (15, 15), 0)
        return cv2.divide(img, 255 - img_blur, scale=256)
    elif flag == 'oil':
        temp_size, bkt_size, step = 4, 8, 2
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_img = ((gray_img / 256) * bkt_size).astype(int)
        img_y, img_x, chn = img.shape
        op_img = np.zeros(img.shape, np.uint8)

        for y in range(0, img_y, step):
            top, btm = y - temp_size, y + temp_size + 1
            if top < 0:
                top = 0
            if btm >= img_y:
                btm = img_y - 1

            for x in range(0, img_x, step):
                lft, rit = x - temp_size, x + temp_size + 1
                if lft < 0:
                    lft = 0
                if rit >= img_x:
                    rit = img_x - 1

                bkt = np.zeros(bkt_size, np.uint8)
                bkt_mean = [0, 0, 0]
                for index_y in range(top, btm):
                    for index_x in range(lft, rit):
                        bkt[gray_img[index_y, index_x]] += 1

                max_bkt = np.max(bkt)
                if max_bkt == 0:
                    max_bkt += 1
                max_index = np.argmax(bkt)

                for index_y in range(top, btm):
                    for index_x in range(lft, rit):
                        if gray_img[index_y, index_x] == max_index:
                            bkt_mean += img[index_y, index_x]
                bkt_mean = (bkt_mean / max_bkt).astype(int)

                for index_y in range(step):
                    for index_x in range(step):
                        op_img[y + index_y, x + index_x] = bkt_mean

        return np.uint8(op_img)
    elif flag == 'glass':
        gg_img = img.copy()
        x, y = img.shape[1], img.shape[0]
        for index_y in range(y):
            for index_x in range(x):
                ran = np.random.randint(0, 3)
                try:
                    gg_img[index_y, index_x] = img[index_y + ran, index_x + ran]
                except IndexError:
                    pass
        return gg_img


def bri_con_adjustment(img, bri, con):
    return np.uint8(np.clip((np.float32(img) - 128) * (np.float32(con) / 100 + 1) + (np.float32(bri)) + 128, 0, 255))


def thresholding(img, flag, thr, mxm):
    if flag == 'Binary':
        return cv2.threshold(img, thr, mxm, cv2.THRESH_BINARY)[1]
    elif flag == 'Inverse Binary':
        return cv2.threshold(img, thr, mxm, cv2.THRESH_BINARY_INV)[1]
    elif flag == 'Trunc':
        return cv2.threshold(img, thr, mxm, cv2.THRESH_TRUNC)[1]
    elif flag == 'Tozero':
        return cv2.threshold(img, thr, mxm, cv2.THRESH_TOZERO)[1]
    elif flag == 'Inverse Tozero':
        return cv2.threshold(img, thr, mxm, cv2.THRESH_TOZERO_INV)[1]


def clustering(img, k):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    if len(img.shape) == 3:
        data = np.float32(img.reshape((-1, 3)))
    elif len(img.shape) == 2:
        data = np.float32(img.reshape((-1, 1)))
    else:
        return -1
    compactness, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    return np.uint8(centers)[labels.flatten()].reshape(img.shape)


def gasuss_noise(img, var=0.01):
    img = np.array(img / 255, dtype=float)
    noise = np.random.normal(0, var ** 0.5, img.shape)
    out = cv2.add(noise, img)
    out = np.clip(out, 0, 1)
    return np.uint8(out * 255)


def spice_noise(img, prob=0.001):
    if len(img.shape) == 3:
        white, black = np.uint8([255, 255, 255]), np.uint8([0, 0, 0])
    elif len(img.shape) == 2:
        white, black = np.uint8(255), np.uint8(0)
    else:
        return
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            rd_num = rd.random()
            if prob >= rd_num > 0.5 * prob:
                img[y, x] = white
            elif 0.5 * prob >= rd_num > 0:
                img[y, x] = black
            else:
                pass
    return img


def morphology(img, m_type, k_type, k_size):
    if k_size == '(3×3)':
        kernel_size = (3, 3)
    elif k_size == '(5×5)':
        kernel_size = (5, 5)
    elif k_size == '(7×7)':
        kernel_size = (7, 7)
    elif k_size == '(9×9)':
        kernel_size = (9, 9)
    else:
        return

    if k_type == 'Square':
        kernel = np.uint8(np.ones(kernel_size))
    elif k_type == 'Circle':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    elif k_type == 'Cross':
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    else:
        return

    if m_type == 'Erosion':
        return cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    elif m_type == 'Dilation':
        return cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel)
    elif m_type == 'Open':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif m_type == 'Close':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif m_type == 'Gradient':
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    elif m_type == 'TopHat':
        return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    elif m_type == 'BottomHat':
        return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)


def ideal_filter(img, flag, per):

    def LPF(spectrum):
        y, x = spectrum.shape
        mid_y, mid_x = int(0.5 * y), int(0.5 * x)
        size_y, size_x = int(per * mid_y), int(per * mid_x)
        spectrum[0: size_y, :] = 0.00001
        spectrum[y - size_y: y, :] = 0.00001
        spectrum[:, 0: size_x] = 0.00001
        spectrum[:, x - size_x: x] = 0.00001
        return spectrum

    def HPF(spectrum):
        y, x = spectrum.shape
        mid_y, mid_x = int(0.5 * y), int(0.5 * x)
        size_y, size_x = int(per * mid_y), int(per * mid_x)
        spectrum[mid_y - size_y: mid_y + size_y, mid_x - size_x: mid_x + size_x] = 0.00001
        return spectrum

    if len(img.shape) == 2:
        gray_spectrum = np.fft.fftshift(np.fft.fft2(img))
        if flag == 'LPF':
            return np.uint8(np.abs(np.fft.ifft2(np.fft.ifftshift(LPF(gray_spectrum)))))
        elif flag == 'HPF':
            return np.uint8(np.abs(np.fft.ifft2(np.fft.ifftshift(HPF(gray_spectrum)))))
    elif len(img.shape) == 3:
        b_chn, g_chn, r_chn = cv2.split(img)
        b_spectrum = np.fft.fftshift(np.fft.fft2(b_chn))
        g_spectrum = np.fft.fftshift(np.fft.fft2(g_chn))
        r_spectrum = np.fft.fftshift(np.fft.fft2(r_chn))
        if flag == 'LPF':
            b_chn_ = np.uint8(np.abs(np.fft.ifft2(np.fft.ifftshift(LPF(b_spectrum)))))
            g_chn_ = np.uint8(np.abs(np.fft.ifft2(np.fft.ifftshift(LPF(g_spectrum)))))
            r_chn_ = np.uint8(np.abs(np.fft.ifft2(np.fft.ifftshift(LPF(r_spectrum)))))
            return cv2.merge([b_chn_, g_chn_, r_chn_])
        elif flag == 'HPF':
            b_chn_ = np.uint8(np.abs(np.fft.ifft2(np.fft.ifftshift(HPF(b_spectrum)))))
            g_chn_ = np.uint8(np.abs(np.fft.ifft2(np.fft.ifftshift(HPF(g_spectrum)))))
            r_chn_ = np.uint8(np.abs(np.fft.ifft2(np.fft.ifftshift(HPF(r_spectrum)))))
            return cv2.merge([b_chn_, g_chn_, r_chn_])
