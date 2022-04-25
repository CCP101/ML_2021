# -*- coding: utf-8 -*-
import math
import random
import numpy as np
import matplotlib.pyplot as plt

dataset = np.random.rand(5000, 2)


def euclideanDistance(vec1, vec2):
    return math.sqrt(((vec1 - vec2) ** 2).sum())


class Canopy:
    def __init__(self, dataset):
        self.dataset = dataset
        self.t1 = 0
        self.t2 = 0
        # 设置初始阈值

    def setThreshold(self, t1, t2):
        if t1 > t2:
            self.t1 = t1
            self.t2 = t2
        else:
            print('t1 needs to be larger than t2!')

    # 使用欧式距离进行距离的计算
    def euclideanDistance(self, vec1, vec2):
        return math.sqrt(((vec1 - vec2) ** 2).sum())

    # 根据当前dataset的长度随机选择一个下标
    def getRandIndex(self):
        return random.randint(0, len(self.dataset) - 1)

    def clustering(self):
        if self.t1 == 0:
            print('Please set the threshold.')
        else:
            canopies = []  # 用于存放最终归类结果
            while len(self.dataset) != 0:
                rand_index = self.getRandIndex()
                current_center = self.dataset[rand_index]  # 随机获取一个中心点，定为P点
                current_center_list = []  # 初始化P点的canopy类容器
                delete_list = []  # 初始化P点的删除容器
                self.dataset = np.delete(
                    self.dataset, rand_index, 0)  # 删除随机选择的中心点P
                for datum_j in range(len(self.dataset)):
                    datum = self.dataset[datum_j]
                    distance = self.euclideanDistance(
                        current_center, datum)  # 计算选取的中心点P到每个点之间的距离
                    if distance < self.t1:
                        # 若距离小于t1，则将点归入P点的canopy类
                        current_center_list.append(datum)
                    if distance < self.t2:
                        delete_list.append(datum_j)  # 若小于t2则归入删除容器
                # 根据删除容器的下标，将元素从数据集中删除
                self.dataset = np.delete(self.dataset, delete_list, 0)
                canopies.append((current_center, current_center_list))
        return canopies


# showCanopy(canopies, dataset, t1, t2)
def showCanopy(canopies, dataset, t1, t2):
    fig = plt.figure()
    sc = fig.add_subplot(111)
    colors = ['brown', 'green', 'blue', 'y', 'r', 'tan', 'dodgerblue', 'deeppink', 'orangered', 'peru', 'blue', 'y']
    markers = ['*', 'h', 'H', '+', 'o', '1', '2', '3', ',', 'v', 'H', '+']
    for i in range(len(canopies)):
        canopy = canopies[i]
        center = canopy[0]
        components = canopy[1]
        sc.plot(center[0], center[1], marker=markers[i % 12],
                color=colors[i % 12], markersize=10)
        t1_circle = plt.Circle(
            xy=(center[0], center[1]), radius=t1, color='dodgerblue', fill=False)
        t2_circle = plt.Circle(
            xy=(center[0], center[1]), radius=t2, color='skyblue', alpha=0.2)
        sc.add_artist(t1_circle)
        sc.add_artist(t2_circle)
        for component in components:
            sc.plot(component[0], component[1],
                    marker=markers[i % 12], color=colors[i % 12], markersize=1.5)
    maxvalue = np.amax(dataset)
    minvalue = np.amin(dataset)
    plt.xlim(minvalue - t1, maxvalue + t1)
    plt.ylim(minvalue - t1, maxvalue + t1)
    plt.show()


if __name__ == '__main__':
    # showCanopy(canopies, dataset, t1, t2)
    t1 = 0.6
    t2 = 0.4
    T = 1000
    Tmin = 10
    distance = 100000
    while T >= Tmin:
        gc = Canopy(dataset)
        gc.setThreshold(t1, t2)
        canopies = gc.clustering()
        current_sum_distance = 0
        for i in range(len(canopies)):
            canopy = canopies[i]
            center = canopy[0]
            current_sum_distance += euclideanDistance(center, i)
        print("聚类结果距离总和为：" + str(current_sum_distance))
        print('得到 %s 初始聚类中心.' % len(canopies))
        print("目前温度为" + str(T))
        print("t1:" + str(t1))
        if current_sum_distance < distance:
            distance = current_sum_distance
            T -= random.randint(0, 9)
            t1 += random.randint(-10, 10) / 1000
        else:
            if random.randint(0, 9) < 4:
                T -= random.randint(0, 9)
            else:
                T += random.randint(0, 4)
