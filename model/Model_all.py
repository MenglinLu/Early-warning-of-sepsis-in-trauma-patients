import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
CardiacData1=pd.read_csv('G:/Cardiac/CardiacData_rf.csv',sep=',',encoding='gbk')
a= CardiacData1.iloc[:,:59]
b= CardiacData1.iloc[:,-1]
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.3, random_state=42)
train= pd.concat([a_train,b_train],axis=1)
train=train.reset_index(drop=True)
test=pd.concat([a_test,b_test],axis=1)
test=test.reset_index(drop=True)
test_X=test.iloc[:,:-1]
test_y=test.iloc[:,-1]
X=train.iloc[:,:-1]
y=train['术后并发症']
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X_resampled, y_resampled = ros.fit_sample(X, y)
SEED = 148
np.random.seed(SEED)

#SVM
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
param_dicts={
'C':np.logspace(-1,3,100),
'gamma':np.linspace(0.0001,10,100)
}
clf_svm=SVC(kernel='rbf',random_state=SEED,class_weight={1: 3})
searchsvm=RandomizedSearchCV(estimator=clf_svm,param_distributions=param_dicts,cv=5,scoring='precision',n_jobs = -1,n_iter=1000)
searchsvm.fit(X,y)
searchsvm.grid_scores_, searchsvm.best_params_, searchsvm.best_score_
print("Best parameter values:", searchsvm.best_params_)
print("CV Score with best parameter values:", searchsvm.best_score_)
df_svm = pd.DataFrame(searchsvm.grid_scores_)
best_svm = searchsvm.best_estimator_
best_svm.fit(X, y)
prediction_svm=best_svm.predict(test_X)
metrics.accuracy_score(test_y, prediction_svm)
metrics.recall_score(test_y, prediction_svm, average='micro')
metrics.confusion_matrix(test_y, prediction_svm)
metrics.precision_score(test_y, prediction_svm, average='macro')
metrics.f1_score(test_y, prediction_svm, average='weighted')
metrics.roc_auc_score(prediction_svm, test_y)  
svm=SVC(C=0.486, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

#Ridge#
from sklearn.linear_model import RidgeClassifier
clf_ridge = RidgeClassifier(random_state=SEED,class_weight={1: 3})
param_dicts={
'alpha':np.logspace(-4, 2, 1000)
}
searchridge=GridSearchCV(estimator=clf_ridge,param_grid=param_dicts,cv=5,scoring='accuracy',n_jobs = -1)
searchridge.fit(X,y)
searchridge.grid_scores_, searchridge.best_params_, searchridge.best_score_
df_ridge = pd.DataFrame(searchridge.grid_scores_)
best_ridge = searchridge.best_estimator_
best_ridge.fit(X, y)
prediction_ridge=best_ridge.predict(test_X)
metrics.confusion_matrix(test_y, prediction_ridge)

#xgboost#
import xgboost as xgb
import matplotlib.pyplot as plt
clf_xgb=xgb.XGBClassifier(objective="binary:logistic",random_state=SEED,scale_pos_weight=3,seed=148)
param_dicts={'n_estimators': range(80, 300,4),
             'learning_rate': np.linspace(0.01,0.2,20),
             'subsample': np.linspace(0.5,1,21),
             'max_depth': range(3,10,1),
             'colsample_bytree': np.linspace(0.5,1,11),
             'max_delta_step':[0,1,2],
             'min_child_weight':range(1,6,1),
             'reg_alpha':[1e-5,1e-2,0.02,0.1,0.3,1]

    }
searchxgb=RandomizedSearchCV(estimator=clf_xgb,param_distributions=param_dicts,cv=5,scoring='recall',n_jobs = -1,n_iter=200)
searchxgb.fit(X,y)
searchxgb.grid_scores_, searchxgb.best_params_, searchxgb.best_score_
print("Best parameter values:", searchxgb.best_params_)
print("CV Score with best parameter values:", searchxgb.best_score_)
df_xgb = pd.DataFrame(searchxgb.grid_scores_)
best_xgb = searchxgb.best_estimator_
best_xgb.fit(X, y)
prediction_xgb=best_xgb.predict(test_X)
prediction_xgb1=best_xgb.predict_proba(test_X)
metrics.accuracy_score(test_y, prediction_xgb)
metrics.recall_score(test_y, prediction_xgb, average='micro')
metrics.confusion_matrix(test_y, prediction_xgb)
metrics.precision_score(test_y, prediction_xgb, average='macro')
metrics.f1_score(test_y, prediction_xgb, average='weighted')
prediction_xgb1=best_xgb.predict_proba(test_X)
metrics.roc_auc_score(test_y, prediction_xgb1[:,1])
metrics.f1_score(prediction_xgb, test_y) 
fig, ax = plt.subplots(figsize=(50,50))
ax.spines['left'].set_position(('outward', 0.2))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position(('outward', 0.2))
ax.tick_params(labelsize=14)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
xgb.plot_importance(clf_xgb,ax,importance_type='gain',height=0.4,max_num_features=40)
fig.savefig('G:/Cardiac/featureimportance.png')
score_weight = clf_xgb.get_booster().get_score(importance_type='weight')
score_gain = clf_xgb.get_booster().get_score(importance_type='gain')
score_cover = clf_xgb.get_booster().get_score(importance_type='cover')
#LightGBM
import lightgbm as lgbm
clf_lgbm=lgbm.LGBMClassifier(random_state=SEED)
param_dicts={'learning_rate': np.linspace(0.001,0.1,30),
     'colsample_bytree': np.linspace(0.5,1,11),
     'num_leaves':range(100,300,4),
     'max_depth': range(10,100,4),
     'n_estimators': range(80, 300,4),
     'min_data_in_leaf':range(150, 300,4)
     }
searchlgbm=RandomizedSearchCV(estimator=clf_lgbm,param_distributions=param_dicts,cv=5,scoring='roc_auc',n_jobs = -1,n_iter=1000)
searchlgbm.fit(X,y)
searchlgbm.grid_scores_, searchlgbm.best_params_, searchlgbm.best_score_
print("Best parameter values:", searchlgbm.best_params_)
print("CV Score with best parameter values:", searchlgbm.best_score_)
df_lgbm = pd.DataFrame(searchlgbm.grid_scores_)
best_lgbm = searchlgbm.best_estimator_
best_lgbm.fit(X, y)
prediction_lgbm=best_lgbm.predict(test_X)
metrics.confusion_matrix(test_y, prediction_lgbm)
LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.55,
        learning_rate=0.048793103448275865, max_depth=94,
        min_child_samples=20, min_child_weight=0.001, min_data_in_leaf=198,
        min_split_gain=0.0, n_estimators=252, n_jobs=-1, num_leaves=220,
        objective=None, random_state=148, reg_alpha=0.0, reg_lambda=0.0,
        silent=True, subsample=1.0, subsample_for_bin=200000,
        subsample_freq=1)

#Logistic Regression
roc_auc_list = []
from sklearn import (metrics, linear_model, preprocessing)
mean_auc = 0.0
n = 5
for i in range(n):
    X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(
    X, y, test_size=0.2, random_state=i*SEED)
    clf_lr.fit(X_train, y_train) 
    preds = clf_lr.predict_proba(X_cv)[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_cv, preds)
    roc_auc = metrics.roc_auc_score(y_cv, preds)
    print("AUC: %0.2f" % (roc_auc))
    roc_auc_list.append(roc_auc)
    mean_auc += roc_auc
print ("Mean AUC: %0.2f" % (mean_auc/n))
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import KFold
param_dicts={
'C':np.logspace(-2,2,100),
'penalty':['l1','l2']
}
clf_lr=LogisticRegression(random_state=SEED)
searchlr=GridSearchCV(estimator=clf_lr,param_grid=param_dicts,cv=5,scoring='roc_auc',n_jobs = -1)
searchlr.fit(X,y)
searchlr.grid_scores_, searchlr.best_params_, searchlr.best_score_
print("Best parameter values:", searchlr.best_params_)
print("CV Score with best parameter values:", searchlr.best_score_)
df_lr = pd.DataFrame(searchlr.grid_scores_)
from sklearn import (metrics, cross_validation, linear_model, preprocessing)
mean_auc = 0.0
n = 5
clf_lr=LogisticRegression(random_state=SEED,n_jobs=-1,class_weight={1: 3.25})
clf_lr.fit(X,y)
prediction_lr=clf_lr.predict(test_X)
metrics.confusion_matrix(test_y, prediction_lr)
prediction_lr=clf_lr.predict_proba(test_X)
metrics.roc_auc_score(test_y,prediction_lr[:,1])

#AdaBoost
weight = np.array([3.25 if i == 1 else 1 for i in y])
from sklearn.ensemble import AdaBoostClassifier
clf_adab=AdaBoostClassifier(random_state=SEED)
param_dicts={'n_estimators': range(80,300,4),
             'learning_rate': np.linspace(0.01,1,50)
}
searchadab=RandomizedSearchCV(estimator=clf_adab,param_distributions=param_dicts,cv=5,scoring='accuracy',n_jobs = -1,n_iter=1000,random_state=SEED)
searchadab.fit(X,y)
searchadab.grid_scores_, searchadab.best_params_, searchadab.best_score_
print("Best parameter values:", searchadab.best_params_)
print("CV Score with best parameter values:", searchadab.best_score_)
df_adab = pd.DataFrame(searchadab.grid_scores_)
best_adab = searchadab.best_estimator_
best_adab.fit(X, y,sample_weight=weight)
prediction_adab=best_adab.predict(test_X)
metrics.confusion_matrix(test_y, prediction_adab)

#RandomForest
from sklearn.ensemble import RandomForestClassifier
clf_rf=RandomForestClassifier(random_state=SEED,class_weight={1: 3.25})
param_dicts = {'n_estimators': np.arange(30,300,5),
               'max_depth':np.arange(3,14,1), 
               'min_samples_split':np.arange(50,200,10),
               'min_samples_leaf':np.arange(10,60,10)
}
searchrf = RandomizedSearchCV(estimator = clf_rf ,param_distributions = param_dicts, scoring='accuracy',cv=5,n_jobs=-1,n_iter=1000)
searchrf.fit(X,y)
searchrf.grid_scores_, searchrf.best_params_, searchrf.best_score_
print("Best parameter values:", searchrf.best_params_)
print("CV Score with best parameter values:", searchrf.best_score_)
df_rf = pd.DataFrame(searchrf.grid_scores_)
best_rf = searchrf.best_estimator_
best_rf.fit(X, y)
prediction_rf=best_rf.predict(test_X)
metrics.confusion_matrix(test_y, prediction_rf)

#ExtraTrees
from sklearn.ensemble import ExtraTreesClassifier
clf_et=ExtraTreesClassifier(random_state=SEED,class_weight={1: 3.25})
param_dicts={'n_estimators': np.arange(30,300,5),
             'max_depth':np.arange(3,14,1), 
             'min_samples_split':np.arange(50,200,10),
             'min_samples_leaf':np.arange(10,60,10)
}
searchet = RandomizedSearchCV(estimator = clf_et ,param_distributions = param_dicts, scoring='accuracy',cv=5,n_jobs=-1,n_iter=1000)
searchet.fit(X,y)
searchet.grid_scores_, searchet.best_params_, searchet.best_score_
print("Best parameter values:", searchet.best_params_)
print("CV Score with best parameter values:", searchet.best_score_)
df_et = pd.DataFrame(searchet.grid_scores_)
best_et = searchet.best_estimator_
best_et.fit(X, y)
prediction_et=best_et.predict(test_X)
metrics.confusion_matrix(test_y, prediction_et)

#GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier
clf_gbdt=GradientBoostingClassifier(random_state=SEED)
param_dicts={'n_estimators':range(20,100,5),
             'max_depth':range(3,14,1), 
             'min_samples_split':range(100,1000,50),
             'min_samples_leaf':range(60,100,5),
             'learning_rate': np.linspace(0.01,0.3,50)
}
searchgbdt = RandomizedSearchCV(estimator = clf_gbdt ,param_distributions = param_dicts, scoring='accuracy',cv=5,n_jobs=-1,n_iter=1000)
searchgbdt.fit(X,y)
searchgbdt.grid_scores_, searchgbdt.best_params_, searchgbdt.best_score_
print("Best parameter values:", searchgbdt.best_params_)
print("CV Score with best parameter values:", searchgbdt.best_score_)
df_gbdt = pd.DataFrame(searchgbdt.grid_scores_)
best_gbdt = searchgbdt.best_estimator_
best_gbdt.fit(X, y,sample_weight=weight)
prediction_gbdt=best_gbdt.predict(test_X)
metrics.confusion_matrix(test_y, prediction_gbdt)

#KNeighbors
from sklearn.neighbors import KNeighborsClassifier
clf_knn=KNeighborsClassifier()
param_dicts={'n_neighbors':np.arange(2,20,1)
}
searchknn=GridSearchCV(estimator=clf_knn,param_grid=param_dicts,cv=5,scoring='accuracy',n_jobs = -1)
searchknn.fit(X,y)
searchknn.grid_scores_, searchknn.best_params_, searchknn.best_score_
df_knn = pd.DataFrame(searchknn.grid_scores_)
best_knn = searchknn.best_estimator_
best_knn.fit(X_resampled,y_resampled)
prediction_knn=best_knn.predict(test_X)
metrics.confusion_matrix(test_y, prediction_knn)

#NaiveBayes
from sklearn.naive_bayes import GaussianNB
clf_gnb = GaussianNB()
clf_gnb.fit(X_resampled,y_resampled)
prediction_gnb=clf_gnb.predict(test_X)
metrics.confusion_matrix(test_y, prediction_gnb)

#Compare different model 
from sklearn.kernel_approximation import Nystroem
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import make_pipeline


def get_models():
    clf_nb = GaussianNB()
    clf_svc = SVC(C=0.7742636826811272,random_state=SEED,probability=True)
    clf_knn = KNeighborsClassifier(n_neighbors=15)
    clf_lr = LogisticRegression(random_state=SEED)
    clf_gb = GradientBoostingClassifier(learning_rate=0.07510204081632653,max_depth=9,min_samples_leaf=95, min_samples_split=700,n_estimators=55,random_state=148)
    clf_rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=SEED)
    clf_et = ExtraTreesClassifier(max_depth=6,min_samples_leaf=10,min_samples_split=80,n_estimators=45,random_state=148)
    clf_xgb = xgb.XGBClassifier(colsample_bytree=1.0, gamma=0, learning_rate=0.01, max_delta_step=1,max_depth=2, min_child_weight=5, missing=None, n_estimators=80,objective='binary:logistic',random_state=148,seed=148, silent=True, subsample=0.975)
    clf_lgbm= lgbm.LGBMClassifier(colsample_bytree=0.6,learning_rate=0.09317241379310345, max_depth=38,min_child_samples=20, min_child_weight=0.001, min_data_in_leaf=210,n_estimators=264, n_jobs=-1, num_leaves=124,random_state=148)
    clf_adb = AdaBoostClassifier(learning_rate=0.07061224489795918, n_estimators=80,random_state=148)

    models = {'svm': clf_svc,
              'knn': clf_knn,
              'naive bayes': clf_nb,
              'random forest': clf_rf,
              'gbm': clf_gb,
              'logistic': clf_lr,
              'xgboost':clf_xgb,
              'adaboost':clf_adb,
              'lightgbm':clf_lgbm,
              'extratree':clf_et,
              }

    return models


def train_predict(model_list):
    P = np.zeros((test_y.shape[0], len(model_list)))
    P = pd.DataFrame(P)
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(X_resampled,y_resampled)
        P.iloc[:, i] = m.predict_proba(test_X)[:, 1]
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        score = metrics.roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")

models = get_models()
P = train_predict(models)
score_models(P, test_y)

from mlens.visualization import corrmat
f, ax = plt.subplots(figsize = (25, 25))
corrmat(P.corr(), inflate=False,ax=ax)
plt.show()
f.savefig('G:/Cardiac/ModelCorrmat.jpg')
    
from sklearn.metrics import roc_curve

def plot_roc_curve(test_y, P_base_learners, P_ensemble, labels, ens_label):
    """Plot the roc curve for base learners and ensemble."""
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--')
    
    cm = [plt.cm.rainbow(i)
      for i in np.linspace(0, 1.0, P_base_learners.shape[1] + 1)]
    
    for i in range(P_base_learners.shape[1]):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(test_y, p)
        plt.plot(fpr, tpr, label=labels[i], c=cm[i + 1])

    fpr, tpr, _ = roc_curve(test_y, P_ensemble)
    plt.plot(fpr, tpr, label=ens_label, c=cm[0])   
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(frameon=False)
    plt.savefig('G:/Cardiac/Roc_curve.jpg')
    plt.show()

plot_roc_curve(test_y, P.values, P.mean(axis=1), list(P.columns), "ensemble")

def get_models():
    clf_lr = LogisticRegression(random_state=148) 
    clf_lr1 = LogisticRegression(random_state=148,class_weight={1: 2.5})   
    clf_xgb=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.95, gamma=0, learning_rate=0.05,
       max_delta_step=2, max_depth=3, min_child_weight=5, missing=None,
       n_estimators=124, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=1,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=0.85)
    clf_xgb1=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.95, gamma=0, learning_rate=0.05,
       max_delta_step=2, max_depth=3, min_child_weight=5, missing=None,
       n_estimators=124, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=1,
       reg_lambda=1, scale_pos_weight=3, seed=None, silent=True,
       subsample=0.85)
    models = {
              'logistic': clf_lr,
              'logistic+weight': clf_lr1,
              'xgboost':clf_xgb,
              'xgboost+weight':clf_xgb1,
              }

    return models


def train_predict(model_list):
    P = np.zeros((test_y.shape[0], len(model_list)))
    P = pd.DataFrame(P)
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(X, y)
        P.iloc[:, i] = m.predict_proba(test_X)[:, 1]
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        score = metrics.roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")

models = get_models()
P = train_predict(models)
score_models(P, test_y)

def plot_roc_curve(test_y, P_base_learners, labels):
    """Plot the roc curve for base learners and ensemble."""
    plt.figure(figsize=(12, 11))
    plt.plot([0, 1], [0, 1], 'k--',lw=lw)

    colors = cycle(['#8A2BE2', '#008000', '#FFFF00','#FF4500'])
    for i,color in zip(range(P_base_learners.shape[1]),colors):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(test_y, p)
        plt.plot(fpr, tpr, label=labels[i], color=color,lw=lw)
    clf_lr2 = LogisticRegression(penalty='l1')
    clf_lr2.fit(X_resampled, y_resampled)
    prediction_lr=clf_lr2.predict_proba(test_X)
    fpr, tpr, _ = roc_curve(test_y, prediction_lr[:,1])
    plt.plot(fpr, tpr, label='logistic+SMOTE', color='#87CEEB',lw=lw)
    clf_xgb2=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
   colsample_bytree=0.95, gamma=0, learning_rate=0.05,
   max_delta_step=2, max_depth=3, min_child_weight=5, missing=None,
   n_estimators=124, n_jobs=1, nthread=None,
   objective='binary:logistic', random_state=1,
   reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
   subsample=0.85)   
    clf_xgb2.fit(X_resampled, y_resampled)
    prediction_xgb=clf_xgb2.predict_proba(test_X1)
    fpr, tpr, _ = roc_curve(test_y, prediction_xgb[:,1])
    plt.plot(fpr, tpr, label='xgboost+SMOTE', color='#808080',lw=lw)
    font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 28,
}
    plt.xlabel('False positive rate',font2)
    plt.ylabel('True positive rate',font2)
    plt.tick_params(labelsize=18)
    plt.legend(loc="lower right",prop={'size':22})
    plt.savefig('G:/Cardiac/Roc_curve.jpg')
    plt.show()

plot_roc_curve(test_y, P.values, list(P.columns))

CardiacData1=pd.read_csv('G:/Cardiac/CardiacData_LOS.csv',sep=',',encoding='gbk')
CardiacData1['出院时状态']=True
droplist=['术后住院时间','出院时状态']
a= CardiacData1.drop(droplist,axis=1)
b= CardiacData1[droplist]
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.3, random_state=42)
train= pd.concat([a_train,b_train],axis=1)
train=train.reset_index(drop=True)
test=pd.concat([a_test,b_test],axis=1)
test=test.reset_index(drop=True)
test_X=test.drop(droplist,axis=1)
test_y=test['术后住院时间']
from lifelines import CoxPHFitter
cph = CoxPHFitter()
train=train.drop('神经系统-膈肌麻痹（可能膈神经损伤）',axis=1)
test_X=test_X.drop('神经系统-膈肌麻痹（可能膈神经损伤）',axis=1)
train=train.drop('Id',axis=1)
test_X=test_X.drop('Id',axis=1)
cph.fit(train, duration_col='术后住院时间', event_col='出院时状态', show_progress=True)
cph.predict_partial_hazard(test_X)
survival_result=cph.predict_survival_function(test_X)
survival_result=survival_result[survival_result<=0.5]
LOSResult=pd.DataFrame(np.arange(1354).reshape((677,2)),columns=['Id','LOS'])
i=0
for c in survival_result.columns:
    item=survival_result[c].idxmax()
    LOSResult.iloc[i,0]=i
    LOSResult.iloc[i,1]=item
    i=i+1
test['Id']=test.index
test1=pd.merge(test,LOSResult,on='Id')
fig, ax = plt.subplots(figsize = (12, 12))
from lifelines import KaplanMeierFitter
kmf_control = KaplanMeierFitter()
ax = kmf_control.fit(test1['术后住院时间'],label='Real').plot(ax=ax,color='#C32B4A')
kmf_exp = KaplanMeierFitter()
ax = kmf_exp.fit(test1['LOS'],label='Predicted').plot(ax=ax,color='#3F76B4')
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 28,
}
plt.xlabel('Postoperative hospital stay (days)',font2)
plt.ylabel('Percent hospitalized',font2)
ax.spines['left'].set_position(('outward', 0.2))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_position(('outward', 0.2))
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
plt.tick_params(labelsize=18)
labels = ax.get_xticklabels() + ax.get_yticklabels()
plt.legend(loc='upper center',prop={'size':22})
fig.savefig('G:/Cardiac/Figure/Hospital stay.tiff')

def get_models():
    clf_lr = LogisticRegression(random_state=148)  
    clf_lr1 = LogisticRegression(random_state=148,class_weight={1: 2.5})   
    models = {
              'logistic': clf_lr,
              'logistic+weight': clf_lr1,
              }

    return models


def train_predict(model_list):
    P = np.zeros((test_y.shape[0], len(model_list)))
    P = pd.DataFrame(P)
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(X, y)
        P.iloc[:, i] = m.predict_proba(test_X)[:, 1]
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P


def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        score = metrics.roc_auc_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")

models = get_models()
P = train_predict(models)
score_models(P, test_y)
lw = 2
from itertools import cycle
from sklearn.metrics import roc_curve, auc
def plot_roc_curve(test_y, P_base_learners, labels):
    """Plot the roc curve for base learners and ensemble."""
    plt.figure(figsize=(12, 11))
    plt.plot([0, 1], [0, 1], 'k--',lw=lw)

    colors = cycle(['#8A2BE2', '#008000'])
    for i,color in zip(range(P_base_learners.shape[1]),colors):
        p = P_base_learners[:, i]
        fpr, tpr, _ = roc_curve(test_y, p)
        plt.plot(fpr, tpr, label=labels[i], color=color,lw=lw)
    clf_lr2 = LogisticRegression(penalty='l1')
    clf_lr2.fit(X_resampled, y_resampled)
    prediction_lr=clf_lr2.predict_proba(test_X)
    fpr, tpr, _ = roc_curve(test_y, prediction_lr[:,1])
    plt.plot(fpr, tpr, label='logistic+SMOTE', color='#87CEEB',lw=lw)
    clf_xgb2=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
   colsample_bytree=0.95, gamma=0, learning_rate=0.05,
   max_delta_step=2, max_depth=3, min_child_weight=5, missing=None,
   n_estimators=124, n_jobs=1, nthread=None,
   objective='binary:logistic', random_state=1,
   reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
   subsample=0.85)   
    clf_xgb2.fit(X_resampled, y_resampled)
    prediction_xgb=clf_xgb2.predict_proba(test_X1)
    fpr, tpr, _ = roc_curve(test_y, prediction_xgb[:,1])
    plt.plot(fpr, tpr, label='xgboost+SMOTE', color='#808080',lw=lw)
    font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 28,
}
    plt.xlabel('False positive rate',font2)
    plt.ylabel('True positive rate',font2)
    plt.tick_params(labelsize=18)
    plt.legend(loc="lower right",prop={'size':22})
    plt.savefig('G:/Cardiac/Roc_curve.jpg')
    plt.show()

plot_roc_curve(test_y, P.values, list(P.columns))

plt.figure(figsize=(12, 11))
clf_lr0 = LogisticRegression(penalty='l2')
clf_lr0.fit(X, y)
prediction_lr=clf_lr0.predict_proba(test_X)
prediction_lr1=clf_lr0.predict(test_X)
fpr, tpr, _ = roc_curve(test_y, prediction_lr[:,1])
plt.plot(fpr, tpr, label='logistic', color='aqua',lw=lw)
table=metrics.confusion_matrix(test_y, prediction_lr1)
x1=table[1,1]/(table[1,1]+table[1,0])
y1=1-(table[0,0]/(table[0,0]+table[0,1]))
plt.plot(y1, x1,'*',color='aqua',markersize=14)
clf_lr1 = LogisticRegression(random_state=148,class_weight={1: 2.5}) 
clf_lr1.fit(X, y)
prediction_lr=clf_lr1.predict_proba(test_X)
prediction_lr1=clf_lr1.predict(test_X)
fpr, tpr, _ = roc_curve(test_y, prediction_lr[:,1])
plt.plot(fpr, tpr, label='logistic+weight', color='darkorange',lw=lw)
table=metrics.confusion_matrix(test_y, prediction_lr1)
x1=table[1,1]/(table[1,1]+table[1,0])
y1=1-(table[0,0]/(table[0,0]+table[0,1]))
plt.plot(y1, x1,'*',color='darkorange',markersize=14)
clf_lr2 = LogisticRegression(penalty='l1')
clf_lr2.fit(X_resampled, y_resampled)
prediction_lr=clf_lr2.predict_proba(test_X)
prediction_lr1=clf_lr2.predict(test_X)
fpr, tpr, _ = roc_curve(test_y, prediction_lr[:,1])
plt.plot(fpr, tpr, label='logistic+SMOTE', color='cornflowerblue',lw=lw)
table=metrics.confusion_matrix(test_y, prediction_lr1)
x1=table[1,1]/(table[1,1]+table[1,0])
y1=1-(table[0,0]/(table[0,0]+table[0,1]))
plt.plot(y1, x1,'*',color='cornflowerblue',markersize=14)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 28,
}
plt.xlabel('False positive rate',font2)
plt.ylabel('True positive rate',font2)
plt.tick_params(labelsize=18)
plt.legend(loc="lower right",prop={'size':22})
plt.savefig('G:/Cardiac/Roc_curve.tiff')
plt.show()

plt.figure(figsize=(12, 11))
clf_xgb0=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.95, gamma=0, learning_rate=0.05,
       max_delta_step=2, max_depth=3, min_child_weight=5, missing=None,
       n_estimators=124, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=1,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=0.85)
clf_xgb0.fit(X, y)
prediction_lr=clf_xgb0.predict_proba(test_X)
prediction_lr1=clf_xgb0.predict(test_X)
fpr, tpr, _ = roc_curve(test_y, prediction_lr[:,1])
plt.plot(fpr, tpr, label='xgboost', color='aqua',lw=lw)
table=metrics.confusion_matrix(test_y, prediction_lr1)
x1=table[1,1]/(table[1,1]+table[1,0])
y1=1-(table[0,0]/(table[0,0]+table[0,1]))
plt.plot(y1, x1,'*',color='aqua',markersize=14)
clf_xgb1=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.95, gamma=0, learning_rate=0.05,
       max_delta_step=2, max_depth=3, min_child_weight=5, missing=None,
       n_estimators=124, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=1,
       reg_lambda=1, scale_pos_weight=3, seed=None, silent=True,
       subsample=0.85)
clf_xgb1.fit(X, y)
prediction_lr=clf_xgb1.predict_proba(test_X)
prediction_lr1=clf_xgb1.predict(test_X)
fpr, tpr, _ = roc_curve(test_y, prediction_lr[:,1])
plt.plot(fpr, tpr, label='xgboost+weight', color='darkorange',lw=lw)
table=metrics.confusion_matrix(test_y, prediction_lr1)
x1=table[1,1]/(table[1,1]+table[1,0])
y1=1-(table[0,0]/(table[0,0]+table[0,1]))
plt.plot(y1, x1,'*',color='darkorange',markersize=14)
clf_xgb2=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
   colsample_bytree=0.95, gamma=0, learning_rate=0.05,
   max_delta_step=2, max_depth=3, min_child_weight=5, missing=None,
   n_estimators=124, n_jobs=1, nthread=None,
   objective='binary:logistic', random_state=1,
   reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
   subsample=0.85)
clf_xgb2.fit(X_resampled, y_resampled)
prediction_lr=clf_xgb2.predict_proba(test_X)
prediction_lr1=clf_xgb2.predict(test_X)
fpr, tpr, _ = roc_curve(test_y, prediction_lr[:,1])
plt.plot(fpr, tpr, label='xgboost+SMOTE', color='cornflowerblue',lw=lw)
table=metrics.confusion_matrix(test_y, prediction_lr1)
x1=table[1,1]/(table[1,1]+table[1,0])
y1=1-(table[0,0]/(table[0,0]+table[0,1]))
plt.plot(y1, x1,'*',color='cornflowerblue',markersize=14)
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 28,
}
plt.xlabel('False positive rate',font2)
plt.ylabel('True positive rate',font2)
plt.tick_params(labelsize=18)
plt.legend(loc="lower right",prop={'size':22})
plt.savefig('G:/Cardiac/Roc_curve.tiff')
plt.show()




