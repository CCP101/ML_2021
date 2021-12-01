import os
from cv2 import cv2

if __name__ == '__main__':
    img_path = "D:\\Dataset\\MNIST\\test_images\\"
    g = os.walk(r"D:\\Dataset\\MNIST\\test_images\\")
    for path, dir_list, file_list in g:
        # 对所有文件执行resize操作
        for file_name in file_list:
            img = cv2.imread(path + "\\" + file_name)
            print(img.shape)
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(path + "\\0_" + file_name, img)
