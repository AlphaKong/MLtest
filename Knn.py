# -*- coding: utf-8 -*-
'''近邻算法 
优点：精度高、对异常值不敏感、无数据输入假定。
缺点：计算复杂度高、空间复杂度高。
适用数据范围：数值型和标称型。
k-近邻算法的一般流程 
(1)收集数据：可以使用任何方法。
(2)准备数据：距离计算所需要的数值，最好是结构化的数据格式。
(3)分析数据：可以使用任何方法。
(4)训练算法：此步驟不适用于1 近邻算法。
(5)测试算法：计算错误率。
(6)使用算法：首先需要输入样本数据和结构化的输出结果，然后运行女-近邻算法判定输
入数据分别属于哪个分类，最后应用对计算出的分类执行后续的处理。

--曼哈顿距离和欧式距离
'''
import kreadmnist 
import numpy as np
import matplotlib.pyplot as plt

def knn_classify(inputs,dataset,label,k=3):
    datasize=dataset.shape[0]
    #欧式距离
    #将inputs与数据集中每一个数据相减
    diff=np.tile(inputs,(datasize,1))-dataset
    #结果进行平方
    sqdiff=diff**2
    #然后将每个对应的列相加起来，求得总和
    squareDist=np.sum(sqdiff,axis=1)
    #对每一个总和求平方根
    dist=squareDist**0.5
    
    #对距离进行排序
    #np.argsort 根据元素的值从大到小队元素进行排序，返回下标
    sortedDistIndex=np.argsort(dist)
    
    #print(len(sortedDistIndex))
    
    #声明一个字典
    classCount={}
    for i in range(k):
        #对选取的k个样本所属的类别个数进行统计
        voteLabel=label[sortedDistIndex[i]]
        classCount[voteLabel]=classCount.get(voteLabel,0)+1
    
    #选取出现的类别次数最多的类别
    maxCount=0
    for key,value in classCount.items():
        if value >maxCount:
            maxCount=value
            classes=key
    return classes

def plotimage(data):
    image=data.reshape(28,28)
    plt.figure()
    plt.imshow(image, cmap="gray_r") # 在MNIST官网中有说道 “Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).”
#    plt.savefig("") #保存图片
    plt.show()
    
if __name__=='__main__':
    #数据的准备                 
    train_x, train_y, test_x, test_y = kreadmnist.read_data()
    
#    targetlabel=knn_classify(tdata[0],data,labels,k=3)
#    print(targetlabel)
#    plotimage(tdata[0])
    
    MrRight=0
    for i in range(test_x.shape[0]):
        tlabel=knn_classify(test_x[i],train_x,train_y,k=3)
        if tlabel==test_y[i]:
            MrRight=MrRight+1
            print('第{}个正确'.format(i))
        else:
            print('第{}个错误'.format(i))
    
    accuracy=float(MrRight)/float(test_x.shape[0])
    print('正确率为：{}'.format(accuracy))
    
    
    

