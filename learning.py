# Drop! Drop! Drop the BASS!
import numpy as np
import random as rd
import matplotlib.pyplot as plt
import copy
import cv2


# 显示图片，任意键退出
def img_print(img):
    cv2.imshow("Demo", img)
    print("Press any key to continue.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# RGB与灰度图互转
def color_convert(img):
    try:
        len(img[0, 0])
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    except TypeError:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


# 分离通道并返回指定通道
def get_split(img, c):
    if c == 'b':
        return cv2.split(img)[0]
    elif c == 'g':
        return cv2.split(img)[1]
    elif c == 'r':
        return cv2.split(img)[2]
    else:
        print("Invalid color index.")
        return -1


# 分离通道并返回指定通道颜色
def get_single_color(img, c):
    y, x, chn = img.shape
    blank = np.zeros((y, x), dtype=img.dtype)

    if c == 'b':
        return cv2.merge([cv2.split(img)[0], blank, blank])
    elif c == 'g':
        return cv2.merge([blank, cv2.split(img)[1], blank])
    elif c == 'r':
        return cv2.merge([blank, blank, cv2.split(img)[2]])
    else:
        print("Invalid color index.")
        return -1


# 对图片叠加高斯噪声，mean为均值，var为方差
def gasuss_noise(img, mean=0, var=0.01):
    img = np.array(img / 255, dtype=float)  # 图像内数据为float时RGB色彩映射由0-255变为0.0-1.0
    noise = np.random.normal(mean, var ** 0.5, img.shape)  # 生成噪声
    out = cv2.add(noise, img)
    out = np.clip(out, 0, 1)
    return np.uint8(out * 255)  # 转换回U8


# 对图片叠加椒盐噪声，prob为转换概率
def spice_noise(img, prob=0.001):
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            rd_num = rd.random()
            if prob >= rd_num > 0.5 * prob:
                img[y, x] = 255  # 涂白
            elif 0.5 * prob >= rd_num > 0:
                img[y, x] = 0  # 涂黑
            else:
                pass
    return img


# 对图片中心旋转，angle为角度（顺时针)
def central_rotate(img, angle):
    y, x, chn = img.shape
    M = cv2.getRotationMatrix2D((y / 2, x / 2), angle, 1)
    return cv2.warpAffine(img, M, (y, x))


# 对图片单方向平移，x、y分别为不同坐标轴上的位移，单位为像素，采用笛卡尔坐标系
def central_transmission(img, x, y):
    M = np.float32([[1, 0, x], [0, 1, -y]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))


# 图像阈值化处理，默认阈值127，最大值255
def img_threshold(img, t_type, val=127, max_v=255):
    if t_type == 'bin':
        return cv2.threshold(img, val, max_v, cv2.THRESH_BINARY)[1]
    elif t_type == '1bin':
        return cv2.threshold(img, val, max_v, cv2.THRESH_BINARY_INV)[1]
    elif t_type == 'tru':
        return cv2.threshold(img, val, max_v, cv2.THRESH_TRUNC)[1]
    elif t_type == 'toz':
        return cv2.threshold(img, val, max_v, cv2.THRESH_TOZERO)[1]
    elif t_type == '1toz':
        return cv2.threshold(img, val, max_v, cv2.THRESH_TOZERO_INV)[1]
    else:
        return -1


# 绘制灰度直方图并输出
def plot_grey_histogram(img):
    plt.hist(img.ravel(), 256)
    plt.show()


# 绘制三原色直方图并输出
def plot_rgb_histogram(img):
    hist_b = cv2.calcHist(img, [0], None, [256], [0, 255])
    hist_g = cv2.calcHist(img, [1], None, [256], [0, 255])
    hist_r = cv2.calcHist(img, [2], None, [256], [0, 255])

    plt.plot(hist_b, color='b')
    plt.plot(hist_g, color='g')
    plt.plot(hist_r, color='r')
    plt.show()


# 提供多种加权方法的灰度化处理
def rgb_2_gray(img, g_type):
    y, x, chn = img.shape
    gray = np.zeros((y, x), np.uint8)

    if g_type == 'max':
        for i in range(y):
            for j in range(x):
                val = max(img[i, j][0], img[i, j][1], img[i, j][2])
                gray[i, j] = np.uint8(val)
        return gray
    elif g_type == 'ave':
        img = np.float32(img)
        b, g, r = cv2.split(img)
        return np.uint8((b + g + r) / 3)
    else:
        return -1


# 实现线性灰度/饱和度操作
def liner_scale(img, shift=0, times=1.0):
    return np.uint8(np.clip((np.float32(img) * times + shift), 0, 255))


def bri_and_con(img, bri, con):
    return np.uint8(np.clip((np.float32(img) - 128) * (np.float32(con) / 100 + 1) + (np.float32(bri)) + 128, 0, 255))


# 实现对数、伽马灰度/饱和度操作
def non_liner_scale(img, t_type, val, gamma=1):
    img = np.float32(img)
    if t_type == "log":
        return np.uint8(np.clip(val * np.log(1.0 + img), 0, 255))
    # val推荐值为10^-7
    elif t_type == 'gamma':
        return np.uint8(np.clip(val * img ** gamma, 0, 255))
    else:
        return -1


# 根据形状与大小获得内容为uint8的卷积核
# 支持矩形、（椭）圆形、十字形，大小需为奇数
def get_kernel(kernel_type, kernel_size=(3, 3)):
    if kernel_type == 'sqr':
        return np.uint8(np.ones(kernel_size))
    elif kernel_type == 'cir':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    elif kernel_type == 'crs':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)
    else:
        return -1


# 基于卷积的查找边缘，输出为灰度图
# 支持Roberts、Prewitt、Sobel、Scharr、Canny、LOG、Laplacian算子
def convolution_based_find_edges(img, operator):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 2:
        gray = img
    else:
        return -1

    if operator == 'roberts':
        kernel_x = np.array([[-1, 0], [0, 1]], dtype=int)
        kernel_y = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv2.filter2D(gray, cv2.CV_16S, kernel_x)
        y = cv2.filter2D(gray, cv2.CV_16S, kernel_y)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    elif operator == 'prewitt':
        kernel_x = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernel_y = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        x = cv2.filter2D(gray, cv2.CV_16S, kernel_x)
        y = cv2.filter2D(gray, cv2.CV_16S, kernel_y)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    elif operator == 'sobel':
        x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    elif operator == 'scharr':
        x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        return cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    elif operator == 'canny':
        return cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    elif operator == 'log':
        return cv2.convertScaleAbs(cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_16S, ksize=3))
    elif operator == 'laplacian':
        return cv2.convertScaleAbs(cv2.Laplacian(gray, cv2.CV_16S, ksize=3))
    else:
        return -1


# 基于卷积的滤镜，输出为灰度图
# 支持Blur（模糊）、Sharpen（锐化）、Emboss（浮雕）
def convolution_based_filter(img, operator):
    if operator == 'blur':
        kernel = np.array((
            [0.0625, 0.125, 0.0625],
            [0.125, 0.25, 0.125],
            [0.0625, 0.125, 0.0625]), dtype="float32")
        return cv2.filter2D(img, -1, kernel)
    elif operator == 'sharpen':
        kernel = np.array((
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]), dtype="float32")
        return cv2.filter2D(img, -1, kernel)
    elif operator == 'emboss':
        kernel = np.array((
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]), dtype="float32")
        return cv2.filter2D(img, -1, kernel)
    elif operator == 'carve':
        kernel = np.array((
            [2, 1, 0],
            [1, 1, -1],
            [0, -1, -2]), dtype="float32")
        return cv2.filter2D(img, -1, kernel)
    else:
        return -1


# 基于K-Means的图像聚类分割，k为簇的个数，需为2的幂
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


# 对图像进行量化，k为量化等级，范围为2^1~2^7
def quantization(img, k):
    return np.uint8(np.round(np.float32(img) / (255 / (k - 1))) * (255 / (k - 1)))


# 计算并返回图像频谱
def fourier_transform(img):
    return np.fft.fftshift(np.fft.fft2(img))


# 绘制图像频谱图
def get_spectrum(spectrum):
    spectrum = np.log(np.abs(spectrum))
    min_num, max_num = np.min(spectrum), np.max(spectrum)
    return np.uint8(255 / (max_num - min_num) * (spectrum - min_num))


# 根据图像频谱还原图像
def inverse_fourier_transform(spectrum):
    return np.uint8(np.abs(np.fft.ifft2(np.fft.ifftshift(spectrum))))


# 对图像低通滤波，size为滤波范围，采用分数（滤波范围/边长，非面积）
def LPF(spectrum, size):
    new_spectrum = copy.deepcopy(spectrum)
    y, x = new_spectrum.shape
    mid_y, mid_x = int(0.5 * y), int(0.5 * x)
    size_y, size_x = int(size * mid_y), int(size * mid_x)
    new_spectrum[0: size_y, :] = 0.00001
    new_spectrum[y - size_y: y, :] = 0.00001
    new_spectrum[:, 0: size_x] = 0.00001
    new_spectrum[:, x - size_x: x] = 0.00001
    return new_spectrum


# 对图像高通滤波，size为滤波范围，采用分数（滤波范围/边长，非面积）
def HPF(spectrum, size):
    new_spectrum = copy.deepcopy(spectrum)
    y, x = new_spectrum.shape
    mid_y, mid_x = int(0.5 * y), int(0.5 * x)
    size_y, size_x = int(size * mid_y), int(size * mid_x)
    new_spectrum[mid_y - size_y: mid_y + size_y, mid_x - size_x: mid_x + size_x] = 0.00001
    return new_spectrum


def lighten_edges(img):
    b_chn, g_chn, r_chn = cv2.split(img)
    kernel = get_kernel('sqr')
    b_gra = cv2.morphologyEx(b_chn, cv2.MORPH_GRADIENT, kernel)
    g_gra = cv2.morphologyEx(g_chn, cv2.MORPH_GRADIENT, kernel)
    r_gra = cv2.morphologyEx(r_chn, cv2.MORPH_GRADIENT, kernel)
    return cv2.merge([b_gra, g_gra, r_gra])


def retro(img):
    img = np.float32(img)
    b_chn, g_chn, r_chn = cv2.split(img)
    r_retro = np.uint8(np.clip((0.393 * r_chn + 0.769 * g_chn + 0.189 * b_chn), 0, 255))
    g_retro = np.uint8(np.clip((0.349 * r_chn + 0.686 * g_chn + 0.168 * b_chn), 0, 255))
    b_retro = np.uint8(np.clip((0.272 * r_chn + 0.534 * g_chn + 0.131 * b_chn), 0, 255))
    return cv2.merge([b_retro, g_retro, r_retro])


def ground_glass(img, size):
    gg_img = img.copy()
    x, y = img.shape[1], img.shape[0]
    for index_y in range(y):
        for index_x in range(x):
            ran = np.random.randint(0, size)
            try:
                gg_img[index_y, index_x] = img[index_y + ran, index_x + ran]
            except IndexError:
                pass
    return gg_img


def sketch(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_inv = 255 - img
    img_blur = cv2.GaussianBlur(img_inv, (15, 15), 0)
    dodge = cv2.divide(img, 255 - img_blur, scale=256)
    return dodge


def oil_painting(img, temp_size, bkt_size, step):
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

            print(x, y)

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
                    try :
                        op_img[y + index_y, x + index_x] = bkt_mean
                    except IndexError:
                        pass

    return np.uint8(op_img)


if __name__ == '__main__':
    uso = cv2.imread('uso.jpg', cv2.IMREAD_UNCHANGED)
    titania = cv2.imread('titania.jpg', cv2.IMREAD_UNCHANGED)
    kanata = cv2.imread("kanata.jpg", cv2.IMREAD_UNCHANGED)
    oil = oil_painting(kanata, 4, 8, 2)
    cv2.imwrite("oil_kanata.jpg", oil)

