# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 08:41:14 2018

@author: Jolin
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from mlens.ensemble import SuperLearner
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import neighbors
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LassoLarsIC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score,recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE

seed=42
data_all=pd.read_csv('C:\\Users\\u\\Desktop\\datathon\\data-sofain.csv',header=0)
data_all.fillna(data_all.mean(),inplace=True)
data_all.to_csv('C:\\Users\\u\\Desktop\\datathon\\data_nomiss.csv')

data_x=data_all.iloc[:,1:-2]
data_y=data_all['hospital_expire_flag']
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.25, random_state=42)

ensemble = SuperLearner(scorer=roc_auc_score,random_state=seed,folds=10,backend="multiprocessing")
ensemble.add([RandomForestClassifier(random_state=seed,n_estimators=250), SVC(),LassoLarsIC(criterion='bic'),ElasticNet(random_state=0),BayesianRidge(),MLPClassifier(),BaggingClassifier(),neighbors.KNeighborsClassifier(),tree.DecisionTreeClassifier(),GradientBoostingClassifier(n_estimators=200)])

    # Attach the final meta estimator
ensemble.add_meta(LogisticRegression())
    
ensemble.fit(x_train, y_train)
preds = ensemble.predict(x_test)
    
ensemble_data=pd.DataFrame(ensemble.data)
auroc=roc_auc_score(preds, y_test)
acc=accuracy_score(preds,y_test)

p=precision_score(preds, y_test)
r=recall_score(preds, y_test)



frp,tpr,threshholds=roc_curve(preds,y_test)

fig=plt.figure()
plt.plot(frp,tpr)
plt.show()


ensemble_data.to_csv('E:\\sepsis_index\\backup\\bbb\\ensembledata_index_numis'+str(9)+'_'+str(round(roc_auc_score(preds, y_test),4))+'_'+str(index)+'.csv')
    
