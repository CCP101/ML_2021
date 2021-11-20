import os
import numpy as np
from PIL import Image


# 图片转换为txt
def ChangeData():
    source_dir = 'C:\\Users\\HP\\Desktop\\1\\'
    target_dir = 'C:\\Users\\HP\\Desktop\\2\\'
    for name in os.listdir(source_dir):
        file_dir = source_dir + '{}/'.format(name)
        for img_name in os.listdir(file_dir):
            # 读取图片
            img_path = file_dir + img_name
            im = Image.open(os.path.expanduser(img_path)).convert("L")  # 将RGB转化为灰度图像，L代表灰度图像，像素值在0~255之间
            im = im.resize((64, 64), Image.ANTIALIAS) 
            size = im.size
            print(size)

            # 计算图像im裁剪比例
            im_nu = np.array(im).reshape(64, 64).astype(np.int)  # 返回新形状的数组,把它变成一个 numpy 数组以匹配数据馈送格式
            raw_top, raw_down, col_top, col_down = corp_margin(im_nu)

            # 准备将图片切割
            weight, height = size[0], size[1]
            # left：与左边界的距离 up：与上边界的距离 right：还是与右边界的距离 below：还是与下边界的距离
            box = (int(weight * col_top), int(height * raw_top), int(weight * col_down), int(height * raw_down))
            im = im.crop(box)
            im = im.resize((64, 64), Image.ANTIALIAS)
            im = np.array(im).reshape(64, 64).astype(np.int)
            # 0/1修改
            for i in range(0, 64):
                for j in range(0, 64):
                    if im[i][j] > 50:
                        im[i][j] = 0
                    else:
                        im[i][j] = 1
            # 保存txt
            s1 = img_name.split('.')
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            txt_path = target_dir + name
            txtFile_path = target_dir + name + '/' + s1[0] + '.txt'
            if not os.path.exists(txt_path):
                os.mkdir(txt_path)
            np.savetxt(txtFile_path, im, fmt='%s', delimiter='', newline='\n')
    return 0


def corp_margin(img):
    (row, col) = img.shape
    bias = 50
    raw_top, raw_down, col_top, col_down = 0, 0, 0, 0
    for r in range(0, row):
        if img[r].sum() < 255 * col - bias:
            raw_top = r
            break

    for r in range(row - 1, 0, -1):
        if img[r].sum() < 255 * col - bias:
            raw_down = r
            break

    for c in range(0, col):
        if img[:, c].sum() < 255 * col - bias:
            col_top = c
            break

    for c in range(col - 1, 0, -1):
        if img[:, c].sum() < 255 * col - bias:
            col_down = c
            break

    # 保持图像方形
    ch = (col_down - col_top) - (raw_down - raw_top)
    if ch > 0:
        raw_top -= int(ch / 2) + 1
        if raw_top < 0:
            raw_top = 0
        raw_down += int(ch / 2)
        if raw_down > 64:
            raw_down = 64
    elif ch < 0:
        ch = abs(ch)
        col_top -= int(ch / 2) + 1
        if col_top < 0:
            col_top = 0
        col_down += int(ch / 2)
        if col_down > 64:
            col_down = 64

    return raw_top/64, raw_down/64, col_top/64, col_down/64


if __name__ == '__main__':
    re = ChangeData()
