import os
import random
from cv2 import cv2

g = os.walk(r"D:\\Dataset\\HWDB\\HWDB1.1F\\train_full\\")
target_dir = "D:\\Dataset\\HWDB\\HWDB1.1F\\train_final_shuffle\\"
count = 0
for path, dir_list, file_list in g:
    # 暴力shuffle 获取所有图片 每次随机拿出一张图片并重新命名
    file = file_list
    print(len(file))
    while len(file) != 0:
        # 当前剩余图片数量
        num_pic = len(file)
        # 随机抽取一张图片
        random_num = random.randint(0, num_pic-1)
        shuffle_file = file[random_num]
        # 获得对应标签
        tag = shuffle_file.split("_")[0]
        img = cv2.imread("D:\\Dataset\\HWDB\\HWDB1.1F\\train_full\\" + shuffle_file)
        # 新存储目录以及文件名
        store_path = target_dir + str(count) + "_" + tag + ".png"
        print(store_path)
        cv2.imwrite(store_path, img)
        # 从列表中移除该张图片
        file.pop(random_num)
        count += 1
