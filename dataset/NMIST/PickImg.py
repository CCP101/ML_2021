import os
import shutil

if __name__ == '__main__':
    dataset_path = "D:\\Dataset\\NMIST\\test1_images\\"
    target_path = "D:\\Dataset\\NMIST\\test_images\\"
    g = os.walk(r"D:\\Dataset\\NMIST\\test1_images\\")
    for path, dir_list, file_list in g:
        count = 0
        for file_name in file_list:
            target_path = path.replace("test1", "test")
            shutil.copyfile(path+"\\"+file_name, target_path+"\\"+str(count)+".jpg")
            print(target_path+"\\"+str(count)+".jpg")
            count += 1
            if count >= 34:
                break
