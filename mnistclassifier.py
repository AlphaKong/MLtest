# -*- coding: utf-8 -*-

#import sys
#import os
import time
from sklearn import metrics
import numpy as np
import pickle
import kreadmnist 



'''
    classifiers = {'NB':naive_bayes_classifier, #快
                  'KNN':knn_classifier, #慢
                   'LR':logistic_regression_classifier, #慢
                   'RF':random_forest_classifier, #快
                   'DT':decision_tree_classifier, #快
                  'SVM':svm_classifier,#慢
                 'GBDT':gradient_boosting_classifier, #慢
                 'AB':AdaBoost_classifier,
                  'NN':Neural_Network_classifier
                  }
'''
 
# Multinomial Naive Bayes Classifier朴素贝叶斯
def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=0.01)
    model.fit(train_x, train_y)
    return model
 
 
# KNN Classifier K近邻
#class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, 
#weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, 
#metric=’minkowski’, metric_params=None, n_jobs=1, **kwargs)
#优点
#1.简单，易于理解，易于实现，无需估计参数，无需训练；
#2. 适合对稀有事件进行分类；
#3.特别适合于多分类问题(multi-modal,对象具有多个类别标签)， kNN比SVM的表现要好。
#缺点
#数据量大时不可行，计算量大等等
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model
 
 
# Logistic Regression Classifier  逻辑回归
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model
 
 
# Random Forest Classifier  随机森林
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_x, train_y)
    return model
 
 
# Decision Tree Classifier  决策树
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model
 
 
# GBDT(Gradient Boosting Decision Tree) Classifier 迭代决策树
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model



# AdaBoost 
def AdaBoost_classifier(train_x,train_y):
    from sklearn.ensemble import AdaBoostClassifier
    model=AdaBoostClassifier(n_estimators=200)
    model.fit(train_x,train_y)
    return model
 
# SVM Classifier   SVM，支持向量机
#class sklearn.svm.SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’,
#coef0=0.0, shrinking=True, probability=False,tol=0.001, 
#cache_size=200, class_weight=None, verbose=False, max_iter=-1, 
#decision_function_shape=’ovr’, random_state=None)[source]
#优点
#二分类效果相当好
#缺点
#数据量大时不可行，计算量大，算法复杂，多分类表现一般等等
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model

# Neural_Network MLP
def Neural_Network_classifier(train_x,train_y):
    from sklearn.neural_network import MLPClassifier
    model=MLPClassifier()
    model.fit(train_x,train_y)
    return model

     
if __name__ == '__main__':
    thresh = 0.5
    model_save_file = None #如果想要保存训练好的模型，就加个名字上去
    model_save = {}
    '''
    全部一起计算会花销大量的时间，所以建议修改后，单个计算，单个记录
    如下
    '''
    test_classifiers = ['RF']
    classifiers = {
                  'RF':random_forest_classifier
    }
    #计算时间
#    test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT','AB']
#    classifiers = {'NB':naive_bayes_classifier, #快
#                  'KNN':knn_classifier, #慢
#                   'LR':logistic_regression_classifier, #慢
#                   'RF':random_forest_classifier, #快
#                   'DT':decision_tree_classifier, #快
#                  'SVM':svm_classifier,#慢
#                 'GBDT':gradient_boosting_classifier, #慢
#                 'AB':AdaBoost_classifier,
#                  'NN':Neural_Network_classifier
#    }
    print('正在读取训练数据集和测试数据集......')
    train_x, train_y, test_x, test_y = kreadmnist.read_data()
    
#    #PCA
#    from sklearn.decomposition import PCA
#    estimator=PCA(n_components=700)
#    
#    train_x=estimator.fit_transform(train_x)
#    test_x=estimator.fit_transform(test_x)
    
    #以下是截取一小部分的数据进行测试，检测算法是否运行
#    train_x= train_x[0:1000]
#    train_y= train_y[0:1000]
#    test_x= test_x[0:50]
#    test_y =test_y[0:50]
#    
    print(train_x.shape)
    #训练集合测试集的数量，数据向量的维度
    num_train, num_feat = train_x.shape
    num_test, num_feat = test_x.shape
    is_binary_class = (len(np.unique(train_y)) == 2)
    print('******************** 数据的信息 *********************')
    #training data: %d, #testing_data: %d, dimension:
    print('#训练数据集: %d, #测试数据集: %d, 数据的维度: %d' % (num_train, num_test, num_feat))
     
    for classifier in test_classifiers:
        print('******************* %s ********************' % classifier)
        start_time = time.time()
        model = classifiers[classifier](train_x, train_y)
        print('training took %fs!' % (time.time() - start_time))
        
        predict = model.predict(test_x)
        if model_save_file != None:
            model_save[classifier] = model
        if is_binary_class:
            precision = metrics.precision_score(test_y, predict)
            recall = metrics.recall_score(test_y, predict)
            print('precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall))
        
        '''----------------------------------------'''
        '''下面的循环主要作用是看第几个有错误，同时看进行检验的过程'''
#        for i in range(len(predict)):
#              if predict[i]==test_y[i]:
#                    print("第{}个正确".format(i))
#              else:
#                    print("第{}个错误".format(i))
        '''---------------------------------------'''
        
        ''' accuracy是sklearn自带的方法  '''
        accuracy = metrics.accuracy_score(test_y, predict)
        print('{} accuracy: {}%'.format(classifier,100 * accuracy))
    #如果model_save_file没有设置，下面语句将不会执行，即训练出来的模型不保存
    if model_save_file != None:
        pickle.dump(model_save, open(model_save_file, 'wb'))


