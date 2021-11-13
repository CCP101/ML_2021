import os
import shutil
from cv2 import cv2
import imgaug as ia
import time
from imgaug import augmenters as iaa

if __name__ == '__main__':
    dataset_path = "D:\\Dataset\\HWDB\\HWDB1.1F\\train1\\"
    target_path = "D:\\Dataset\\HWDB\\HWDB1.1F\\train\\"
    g = os.walk(r"D:\\Dataset\\HWDB\\HWDB1.1F\\train1\\")

    for path, dir_list, file_list in g:
        for i in dir_list:
            new_path = target_path + i
            print(new_path)
            folder = os.path.exists(new_path)
            if not folder:
                os.makedirs(new_path)

    for path, dir_list, file_list in g:
        count = 0
        for file_name in file_list:
            if count < 160:
                print(path + "\\" + file_name)
                img = cv2.imread(path + "\\" + file_name)
                height, width = img.shape[:2]
                if height * width >= 400:
                    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                    new_path = path.replace("train1", "train")
                    store_path = new_path + "\\" + str(count)+".png"
                    # print(store_path)
                    cv2.imwrite(store_path, img)
                count += 1
            elif 160 <= count < 180:
                seq = iaa.Sequential([
                        iaa.GaussianBlur(sigma=(0, 3.0))])  # 使用0到3.0的sigma模糊图像
                print(path + "\\" + file_name)
                img = cv2.imread(path + "\\" + file_name)
                height, width = img.shape[:2]
                if height * width >= 400:
                    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                    new_image = seq.augment_image(img)
                    new_path = path.replace("train1", "train")
                    store_path = new_path + "\\" + str(count)+".png"
                    # print(store_path)
                    cv2.imwrite(store_path, new_image)
                count += 1
            elif 180 <= count < 200:
                seq = iaa.Sequential([
                    iaa.AdditiveGaussianNoise(
                        loc=0, scale=(0.0, 0.05 * 1000), per_channel=0.5)
                ])
                print(path + "\\" + file_name)
                img = cv2.imread(path + "\\" + file_name)
                height, width = img.shape[:2]
                if height * width >= 400:
                    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                    new_image = seq.augment_image(img)
                    new_path = path.replace("train1", "train")
                    store_path = new_path + "\\" + str(count)+".png"
                    # print(store_path)
                    cv2.imwrite(store_path, new_image)
                count += 1
            elif 200 <= count < 220:
                seq = iaa.Sequential([
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5))
                ])
                print(path + "\\" + file_name)
                img = cv2.imread(path + "\\" + file_name)
                height, width = img.shape[:2]
                if height * width >= 400:
                    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                    new_image = seq.augment_image(img)
                    new_path = path.replace("train1", "train")
                    store_path = new_path + "\\" + str(count)+".png"
                    # print(store_path)
                    cv2.imwrite(store_path, new_image)
                count += 1
