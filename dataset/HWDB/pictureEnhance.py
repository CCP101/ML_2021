import os
import shutil
from cv2 import cv2
import imgaug as ia
import time
import re
from imgaug import augmenters as iaa

if __name__ == '__main__':
    g = os.walk(r"D:\\Dataset\\HWDB\\HWDB1.1F\\test_cut\\")

    # for path, dir_list, file_list in g:
    #     for i in dir_list:
    #         new_path = target_path + i
    #         print(new_path)
    #         folder = os.path.exists(new_path)
    #         if not folder:
    #             os.makedirs(new_path)

    for path, dir_list, file_list in g:
        count = 0
        count_z = 0
        label = 0
        for file_name in file_list:
            print(path + "\\" + file_name)
            img = cv2.imread(path + "\\" + file_name)
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
            label_path = path.replace("D:\\\\Dataset\\\\HWDB\\\\HWDB1.1F\\\\test_cut\\\\test", "")
            label_path = int(label_path[0:5])
            if label_path == label:
                count_z += 1
                if count_z <= 250:
                    img1 = img
                    new_path = "D:\\Dataset\\HWDB\\HWDB1.1F\\test\\"
                    store_path = new_path + str(label_path) + "_" + str(count) + ".png"
                    print(store_path)
                    cv2.imwrite(store_path, img1)
                    count += 1

                    seq1 = iaa.Sequential([
                        iaa.GaussianBlur(sigma=(0, 3.0))])  # 使用0到3.0的sigma模糊图像
                    new_image1 = seq1.augment_image(img)
                    store_path = new_path + str(label_path) + "_" + str(count) + ".png"
                    print(store_path)
                    cv2.imwrite(store_path, new_image1)
                    count += 1

                    seq2 = iaa.Sequential([
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.05 * 1000), per_channel=0.5)
                    ])
                    print(path + "\\" + file_name)
                    new_image2 = seq2.augment_image(img)
                    store_path = new_path + str(label_path) + "_" + str(count) + ".png"
                    print(store_path)
                    cv2.imwrite(store_path, new_image2)
                    count += 1

                    seq3 = iaa.Sequential([
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
                    ])
                    print(path + "\\" + file_name)
                    new_image3 = seq3.augment_image(img)
                    store_path = new_path + str(label_path) + "_" + str(count) + ".png"
                    print(store_path)
                    cv2.imwrite(store_path, new_image3)
                    count += 1
                else:
                    continue
            else:
                label = label_path
                count_z = 0
