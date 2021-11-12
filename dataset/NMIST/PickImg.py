import os

if __name__ == '__main__':
    dataset_path = "D:\\Dataset\\NMIST\\train1_images\\"
    g = os.walk(r"D:\\Dataset\\NMIST\\train1_images\\")
    for path, dir_list, file_list in g:
        print(dir_list)
        count = 0
        for file_name in file_list:

            count += 1
            if count >= 1:
                break

