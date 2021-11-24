import os
import random
from cv2 import cv2

g = os.walk(r"D:\\Dataset\\HWDB\\HWDB1.1F\\train_full\\")
target_dir = "D:\\Dataset\\HWDB\\HWDB1.1F\\train_final_shuffle\\"
count = 0
for path, dir_list, file_list in g:
    file = file_list
    print(len(file))
    while len(file) != 0:
        # 当前剩余图片数量
        num_pic = len(file)
        random_num = random.randint(0, num_pic-1)
        shuffle_file = file[random_num]
        tag = shuffle_file.split("_")[0]
        img = cv2.imread("D:\\Dataset\\HWDB\\HWDB1.1F\\train_full\\" + shuffle_file)
        store_path = target_dir + str(count) + "_" + tag + ".png"
        print(store_path)
        cv2.imwrite(store_path, img)
        file.pop(random_num)
        count += 1
