import copy
import os
import numpy as np
import cv2
import math
from copy import deepcopy
from PIL import Image, ImageOps


def Dataset():
    sourceDir = "./data1"
    class_label = []
    samples_data = []
    # 获取data目录下文件名
    dirInx = 0
    for root, dir, files in os.walk(sourceDir):
        dirs = dir
        break

    # 添加标签
    for dir in dirs:
        subPath = os.path.join(sourceDir, dirs[dirInx])
        # 获取每个label下的样本
        for file1 in os.listdir(subPath):
            class_label.append(dir)
            # 图片路径
            imgPath = os.path.join(subPath, file1)
            im = Image.open(os.path.expanduser(imgPath)).convert("L")  # 将RGB转化为灰度图像，L代表灰度图像，像素值在0~255之间
            # im = ImageOps.invert(im)  # 因为画图软件是白底黑字，与MNIST相反，所以要反转一下
            im = im.resize((20, 20), Image.ANTIALIAS)  # resize image with high-quality 图像大小为20*20
            im = np.array(im).reshape(400, 1).astype(np.float32)  # 返回新形状的数组,把它变成一个 numpy 数组以匹配数据馈送格式
            # 添加样本
            samples_data.append(im)
        dirInx += 1
    return samples_data, class_label


# 求每类样本均值向量
def Get_Junzhi(samples_data, class_label):
    TO = len(samples_data)
    # num存放每类样本的个数
    num = {}
    # 图像size
    w = 20
    h = 20
    b = [0] * w * h
    b_all = [0] * w * h
    # 每类样本的像素和
    pix_b = {}
    # 除b类外样本的像素和
    b_res = [0] * w * h
    pix_res = {}
    # 全部样本的像素和
    pix_all = []
    for i in range(TO):
        if class_label[i] in num:
            # 同类样本个数计数
            num[class_label[i]] += 1
            images = samples_data[i]
            for j in range(w):  # 获得灰度照片的像素值
                for k in range(h):
                    L = images[j * h + k]
                    # 同类样本同一位置的像素点累加
                    b[j * h + k] += L
                    # 全部样本同一位置的像素点累加
                    b_all[j * h + k] += L
        else:
            num[class_label[i]] = 1
            images = samples_data[i]
            if len(num) > 1:
                # idx = len(num) - 2
                idx = i - 1
                # 保存同类样本同一位置的像素点累加结果
                pix_b[class_label[idx]] = b
                # b重置
                b = [0] * w * h
            for j in range(w):  # 获得灰度照片的像素值
                for k in range(h):
                    L = images[j * h + k]
                    # 同类样本同一位置的像素点累加
                    b[j * h + k] += L
                    # 全部样本同一位置的像素点累加
                    b_all[j * h + k] += L
    # 最后一类的像素点需要另外保存
    pix_b[class_label[TO - 1]] = b
    pix_all = copy.deepcopy(b_all)
    nu = 0
    for pkey in pix_b.keys():
        pix = pix_b.get(pkey)
        for j in range(w):
            for k in range(h):
                # 除b类外样本的像素和
                b_res[j * h + k] = int((b_all[j * h + k] - pix[j * h + k]) / (TO - num.get(pkey)))
                # 同类样本同一位置的像素点除以该类样本个数得均值
                pix[j * h + k] = int(pix[j * h + k] / num.get(pkey))
                if nu == 0:
                    pix_all[j * h + k] = int(pix_all[j * h + k] / TO)
        pix_res[pkey] = deepcopy(b_res)
        b_res = [0]*w*h
        nu += 1
    # pix_b 每类样本20*20均值向量,pix_res除b类外样本均值向量, pix_all全部样本20*20均值向量
    return pix_b, pix_res, pix_all


# 求Sb 类间离散度矩阵
def Get_Sb(mean_vector1, mean_res):
    # 图像size:20*20
    w, h = 20, 20
    t1, m1 = {}, {}
    tem_t1, tem_m1, Sb = [0] * w * h, [0] * w * h, [0] * w * h
    for i in mean_vector1.keys():
        for j in range(w):
            for k in range(h):
                tem_in = mean_vector1.get(i)
                tem_resin = mean_res.get(i)
                # 求u-ui
                tem_m1[j * h + k] = tem_in[j * h + k] - tem_resin[j * h + k]
                # #求u-ui的转置
                tem_t1[k * w + j] = tem_m1[j * h + k]
        # 保存每类样本的u-ui及其转置
        m1[i] = copy.deepcopy(tem_m1)
        t1[i] = copy.deepcopy(tem_t1)
        tem_m1 = [0] * w * h
        tem_t1 = [0] * w * h
    for i in m1.keys():
        tem_in = m1.get(i)
        for j in range(w):
            for k in range(h):
                Sb[j * h + k] += tem_in[j * h + k]
    # m1每个分类与其他分类总体之间的离散度,Sb 总类间离散度
    return m1, Sb


# 求Sw
def Get_Cov(samples_data, mean_vector1, mean_mres, samples_label):
    # 图像size:20*20
    w, h = 20, 20
    cov, rescov , tem_cov, tem_rescov= [0] * w * h, [0] * w * h, [0] * w * h, [0] * w * h
    covsum, rescovsum = {}, {}
    Sw = [0] * w * h
    m1 = {}
    for vec_key in mean_vector1.keys():
        tem_in = mean_vector1.get(vec_key)
        tem_resin = mean_mres.get(vec_key)
        i = 0
        for img in samples_data:
            # 取该样本的类别
            imgLabel = samples_label[i]
            if imgLabel == vec_key:
                for j in range(w*h):
                    # 求样本与均值的差值(X- Mi）
                    tem_cov[j] = img[j][0] - tem_in[j]
                cov += np.dot(np.array(tem_cov).reshape(400, 1), np.array(tem_cov).reshape(1, 400))
            else:
                for j in range(w * h):
                    # 求样本res与均值的差值
                    tem_rescov[j] = img[j][0] - tem_resin[j]
                    # 样本res与均值的差值的转置
                rescov += np.dot(np.array(tem_rescov).reshape(400, 1), np.array(tem_rescov).reshape(1, 400))
            i += 1
        covsum[vec_key] = copy.deepcopy(cov)
        rescovsum[vec_key] = copy.deepcopy(rescov)
        cov = [0] * w * h
        tem_cov = [0] * w * h
        rescov = [0] * w * h
        tem_rescov = [0] * w * h
    for x in covsum.keys():
        tem_m1 = covsum.get(x)
        tem_m2 = rescovsum.get(x)
        #Sw = S1 + S2
        m1[x] = tem_m1 + tem_m2
    # 求Sw 总的类内离散度
    for i in m1.keys():
        tem_in = m1.get(i)
        for j in range(w*h):
            Sw[j] += tem_in[j]
    # 求类内离散度
    # m1每个分类类内离散度,Sw 总类内离散度
    return m1, Sw


def trans_mar(Sb, Sw):
    # Sb，Sw转换为:w*h矩阵
    w = 20
    h = 20
    tmp_Sb = [0] * w
    tmp_Sw = [0] * w
    t_Sb = []
    t_Sw = []
    for j in range(w):
        for k in range(h):
            tmp_Sb[k] = Sb[j * h + k]
            tmp_Sw[k] = Sw[j * h + k]
        t_Sw.append(tmp_Sw)
        t_Sb.append(tmp_Sb)
        tmp_Sb = [0] * w
        tmp_Sw = [0] * w
    return t_Sb, t_Sw

# 求Sw的逆和Sb的乘积的特征值和 特征向量 W以及 W的转置
# 返回 W_T
def Get_tezhengzhi(Sbi, Swi):
    W_T = {}
    # 某类样本与其他样本的特征值
    for inx in Sbi.keys():
        Sb = Sbi.get(inx)
        Sw = Swi.get(inx)
        #tSb, tSw = trans_mar(Sb, Sw)
        Inv_Sw = np.linalg.pinv(Sw)
        Sw_Sb = np.dot(Inv_Sw, Sb)
        #a, W = np.linalg.eig(Sw_Sb)  # a--特征值  W--特征向量
        # 取前最大的(投影向量的个数)个特征向量组成W矩阵即可
        #W = Sw_Sb[1:]
        W_T[inx] = Sw_Sb
    return W_T


# 计算类样本中心的位置
def Get_Center_XY(mean_vector1, mean_res, W_T):
    XY_center = {}
    XY_centers = {}
    for key in mean_vector1.keys():
        val = mean_vector1.get(key)
        val2 = mean_res.get(key)
        # tval1, tval2 = trans_mar(val, val)
        #tval1 = np.array(val).reshape(20, 20)
        #tval2 = np.array(val2).reshape(20, 20)
        valwt = W_T.get(key)
        XY_1 = np.dot(valwt, val)
        XY_2 = np.dot(valwt, val2)
        XY_center[0] = copy.deepcopy(XY_1)
        XY_center[1] = copy.deepcopy(XY_2)
        XY_centers[key] = copy.deepcopy(XY_center)
    return XY_centers


# 求f(x),得到判断向量
# Fisher判断类别
def Get_F(data, W_TS, XY_centers):
    F = 'false'
    i, min = 0, 0
    for W_Tkey in W_TS:
        W_Tvalues = W_TS.get(W_Tkey)
        XY_data = abs(np.dot(W_Tvalues, data[0]))
        XY_centerval = XY_centers.get(W_Tkey)
        X1 = abs(XY_data - abs(XY_centerval[0]))
        X2 = abs(XY_data - abs(XY_centerval[1]))
        X = (abs(XY_centerval[0]) + abs(XY_centerval[1]))/2
        if i == 0:
            if X1 < X2:
                F = W_Tkey
                min = X
                i += 1
        else:
            if X1 < X2 and X2 < min:
                F = W_Tkey
                min = X

    return F


# RGB 变换
# 改变图像RGB存储形式
def Get_Img(testDir):
    # 测试本地图片
    images = []
    # 循环对目录里的图片进行预测，并得出结果
    for file in os.listdir(testDir):
        # 图片路径
        imgPath = os.path.join(testDir, file)
        im = Image.open(os.path.expanduser(imgPath)).convert("L")  # 将RGB转化为灰度图像，L代表灰度图像，像素值在0~255之间
        # im = ImageOps.invert(im)  # 因为画图软件是白底黑字，与MNIST相反，所以要反转一下
        im = im.resize((20, 20), Image.ANTIALIAS)  # resize image with high-quality 图像大小为28*28
        im = np.array(im).reshape(1, 400).astype(np.float32)  # 返回新形状的数组,把它变成一个 numpy 数组以匹配数据馈送格式。
        images.append(im)
    return images


# 实现费歇尔分类
def Get_Cla_Image(test_data, W_T, XY_center):
    result = []
    for img in test_data:
        F_X = Get_F(img, W_T, XY_center)
        result.append(F_X)
    return result


if __name__ == '__main__':
    test_data, test_label = Dataset()
    m1, mres, mall = Get_Junzhi(test_data, test_label)
    Sbi, Sb = Get_Sb(m1, mres)
    Swi, Sw = Get_Cov(test_data, m1, mres, test_label)
    W_T = Get_tezhengzhi(Sbi, Swi)
    XY_center = Get_Center_XY(m1, mres, W_T)

    testDir = "./test1"
    test_data = Get_Img(testDir)
    result = Get_Cla_Image(test_data, W_T, XY_center)
    print(result)
