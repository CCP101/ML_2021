import os
import shutil
import cv2
import imgaug as ia
import time
from imgaug import augmenters as iaa

if __name__ == '__main__':
    dataset_path = "D:\\Dataset\\HWDB\\HWDB1.1F\\test1\\"
    target_path = "D:\\Dataset\\HWDB\\HWDB1.1F\\test\\"
    g = os.walk(r"D:\\Dataset\\HWDB\\HWDB1.1F\\test1\\")

    # for path, dir_list, file_list in g:
    #     for i in dir_list:
    #         new_path = target_path + i
    #         print(new_path)
    #         folder = os.path.exists(new_path)
    #         if not folder:
    #             os.makedirs(new_path)

    for path, dir_list, file_list in g:
        count = 0
        for file_name in file_list:
            img = cv2.imread(path + "\\" + file_name)
            height, width = img.shape[:2]
            if height * width >= 400:
                img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                new_path = path.replace("test1", "test")
                store_path = new_path + "\\" + str(count)+".png"
                print(store_path)
                cv2.imwrite(store_path, img)
            count += 1
