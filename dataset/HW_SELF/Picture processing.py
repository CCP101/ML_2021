from PIL import Image
import os

def produceImage(file_in, width, height, file_out):
    image = Image.open(file_in,"r")
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    resized_image.save(file_out)


if __name__ == '__main__':

    for name in os.listdir('./0/'):
        file_dir = './0/'  # 原始图片路径
         # 保存路径

        for img_name in os.listdir(file_dir):
            save_path = './new_data/{}'.format(img_name)
            img_path = file_dir + img_name  # 批量读取图片
            # print(img_path)
            img = Image.open(img_path)
            width = 64
            height = 64
            produceImage(img_path, width, height, save_path)