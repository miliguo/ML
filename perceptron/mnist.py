from numpy import *
import operator
import os
import numpy as np
import time
from scipy.special import expit
import matplotlib.pyplot as plt
from matplotlib import cm
from os import listdir
from mpl_toolkits.mplot3d import Axes3D
import struct
import math
#读取图片
def read_image(file_name):
    #先用二进制方式把文件都读进来
    file_handle=open(file_name,"rb")  #以二进制打开文档
    file_content=file_handle.read()   #读取到缓冲区中
    offset=0
    head = struct.unpack_from('>IIII', file_content, offset)  # 取前4个整数，返回一个元组
    offset += struct.calcsize('>IIII')
    imgNum = head[1]  #图片数
    rows = head[2]   #宽度
    cols = head[3]  #高度

    images=np.empty((imgNum , 784))#empty，是它所常见的数组内的所有元素均为空，没有实际意义，它是创建数组最快的方法
    image_size=rows*cols#单个图片的大小
    fmt='>' + str(image_size) + 'B'#单个图片的format

    for i in range(imgNum):
        images[i] = np.array(struct.unpack_from(fmt, file_content, offset))
        # images[i] = np.array(struct.unpack_from(fmt, file_content, offset)).reshape((rows, cols))
        offset += struct.calcsize(fmt)
    return images

#读取标签
def read_label(file_name):
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中

    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')

    labelNum = head[1]  # label数
    # print(labelNum)
    bitsString = '>' + str(labelNum) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)

def loadDataSet():
    train_x_filename="train-images.idx3-ubyte"
    train_y_filename="train-labels.idx1-ubyte"
    test_x_filename="t10k-images.idx3-ubyte"
    test_y_filename="t10k-labels.idx1-ubyte"
    train_x=read_image(train_x_filename)
    train_y=read_label(train_y_filename)
    test_x=read_image(test_x_filename)
    test_y=read_label(test_y_filename)
    for i in range(len(train_y)):
        if train_y[i]!=0:
            train_y[i]=1
    for i in range(len(test_y)):
        if test_y[i]!=0:
            test_y[i]=1

    #调试的时候让速度快点，就先减少数据集大小
    train_x=train_x[0:1000,:]
    train_y=train_y[0:1000]
    test_x=test_x[0:500,:]
    test_y=test_y[0:500]


    return train_x, test_x, train_y, test_y

