import os
import shutil
from cv2 import cv2 
import time

# MNIST数据集原始图片大小为28*28，为与HWDB数据集统一，缩放为64*64像素
if __name__ == '__main__':
    dataset_path = "D:\\Dataset\\HWDB\\HWDB1.1F\\test1\\"
    target_path = "D:\\Dataset\\HWDB\\HWDB1.1F\\test\\"
    g = os.walk(r"D:\\Dataset\\HWDB\\HWDB1.1F\\test1\\")

    # 批量创建文件夹
    # for path, dir_list, file_list in g:
    #     for i in dir_list:
    #         new_path = target_path + i
    #         print(new_path)
    #         folder = os.path.exists(new_path)
    #         if not folder:
    #             os.makedirs(new_path)

    # 游走到目标目录，并对所有文件处理
    for path, dir_list, file_list in g:
        count = 0
        for file_name in file_list:
            img = cv2.imread(path + "\\" + file_name)
            # 获取图片宽高属性
            height, width = img.shape[:2]
            # 判断图片是否过小
            if height * width >= 400:
                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                new_path = path.replace("test1", "test")
                # 新存储路径
                store_path = new_path + "\\" + str(count)+".png"
                print(store_path)
                # 存储新图片
                cv2.imwrite(store_path, img)
            count += 1
