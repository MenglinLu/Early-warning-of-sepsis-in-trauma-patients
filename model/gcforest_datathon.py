#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 09:59:49 2018

@author: DJ-TJ
"""

import numpy
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from sklearn.metrics import roc_curve
#data=pd.read_csv("D:/DATAFinal.csv")
# -*- coding:utf-8 -*-
__author__ = 'fms'

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from GCForest import gcForest
# import gcforest as gcForest
from sklearn import cross_validation,metrics
#from python_dailyworking.Get_KS import get_ks
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


input_data = pd.read_csv('inputs/data_last.csv')
data = input_data.drop('hospital_expire_flag', axis=1)

# nt("222")
X_df = data.iloc[:,data.columns != 'Sepsis']
y_df = data.iloc[:,data.columns == 'Sepsis']
X_df1 =X_df.fillna(X_df.mean())
X = np.array(X_df1)
Y = np.array(y_df)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
gcf = gcForest(shape_1X=5, window = 3, tolerance=0.001)
gcf.fit(X_train, Y_train)

#predict 方法预测的是每一条样本的分类类别，结果 pred_X 是一个 [0,1,2...]的 array
pred_X = gcf.predict(X_test)
accuracy = accuracy_score(y_true=Y_test, y_pred=pred_X) #用 test 数据的真实类别和预测类别算准确率
print ('gcForest accuracy:{}'.format(accuracy))

#  predict_proba 方法预测的是每一条样本为 0，1,...类别的概率，结果是这样的：
# [[ 概率 1，概率 2，...],[ 概率 1，概率 2，...],...]的 DataFrame
# [:,1]表示取出序号为 1 的列，也就是预测类别为 1 的概率值,结果是一个数组
Y_predict_prod_test = gcf.predict_proba(X_test)[:, 1]
test_auc = metrics.roc_auc_score(Y_test,Y_predict_prod_test)#验证集上的auc值

print ("预测的AUC是： %s" %test_auc)
# print(Y_predict_prod_test)
y_test_df = pd.DataFrame(Y_test)
y_test_df['y_predict'] = Y_predict_prod_test
y_test_df.to_csv('gcforest_sepsis.csv')

fpr,tpr,thresh=roc_curve(Y_test, Y_predict_prod_test)
fig=plt.figure()
plt.plot(fpr,tpr)
plt.title('gcForest(0.809)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

#Y_predict_prod_train = gcf.predict_proba(X_train)[:, 1]
#print(Y_predict_prod_test.shape)
#print(Y_predict_prod_train.shape)
#下面是算 KS 值，不理解的可以置之不理
#print ('model Y_test ks: ',get_ks(Y_test, Y_predict_prod_test))
#print ('model Y_train ks: ',get_ks(Y_train, Y_predict_prod_train))

#y=input_data["is_sepsis"]
