# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:17:14 2018

@author: xuzh
"""

import imp
import set_user_data_path
imp.reload(set_user_data_path)
set_user_data_path.set_data_path()
import logging
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2
import pandas as pd
import getpass
import imp
import set_user_data_path
imp.reload(set_user_data_path)
set_user_data_path.set_data_path()

from sklearn import cross_validation, metrics
from sklearn.cross_validation import train_test_split
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False   
from sklearn.metrics import confusion_matrix
from sklearn.metrics  import roc_curve, auc, accuracy_score
from sklearn.decomposition import PCA
import collections
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import collections
import jieba
import jieba.posseg as pseg
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import json
import time
from sklearn.externals import joblib
import pdb
import sys
stdi, stdo, stde = sys.stdin, sys.stdout, sys.stderr
sys.stdin, sys.stdout, sys.stderr = stdi, stdo, stde
import lightgbm as lgb
from sklearn.externals import joblib
import transfusion_data_clear as transfusion
from collections import OrderedDict
from sklearn.feature_selection import SelectFromModel
import Filter_data_by_condition as my_filter
from sklearn.metrics import roc_auc_score, mean_squared_error
import logging
logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %y %H:%M:%S',
                filename='blood_train_test_binary.log',
                filemode='w')
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
import re
from StyleFrame import StyleFrame
from sklearn.preprocessing import LabelEncoder
import online_training.train as trn
imp.reload(trn)
#%%
# Create a database connection
user = 'postgres'
passwd = 'postgres'
host = '192.168.1.104'
dbname = 'mimiciii'
schema = 'mimiciii'

a=pd.read_csv('d://301datathon//a_injury.csv')
c1 = a['code']

# Connect to the database
con = psycopg2.connect(dbname=dbname, user=user, host=host, password=passwd)
cur = con.cursor()
rows_all = []
for code in c1:
    cur.execute("select * from mimiciii.diagnoses_icd where icd9_code like '%s'"%(code))
    rows=cur.fetchall()
    rows_all.append(rows)
cur.close()


import pdb
rows_total = []
for i in rows_all:
    if len(i)>0:
        for j in i:
            rows_total.append(j)
        

truma_db = pd.DataFrame(rows_total, columns=['row_id', 'subject_id', 'hadm_id', 'seq_num', 'icd9_code'])
truma_db2 = truma_db.drop_duplicates('hadm_id')

idx = pd.isnull(truma_db2['hadm_id'])==False

truma_db3 = truma_db2[idx].reset_index(drop=True)
truma_db3.drop(['row_id'], axis=1, inplace=True)

#%%


#%%
#加入icu记录，与icu连接
con = psycopg2.connect(dbname=dbname, user=user, host=host, password=passwd)
cur = con.cursor()
cur.execute("select * from mimiciii.icustay_detail")
column_names = [desc[0] for desc in cur.description]
icustays=pd.DataFrame(cur.fetchall(),columns=column_names)
cur.close()

truma_with_icu = truma_db3.merge(icustays, on=['subject_id', 'hadm_id'], how='inner')
#删除'row_id'
if 'row_id' in truma_with_icu.columns:
    truma_with_icu.drop(['row_id'], axis=1, inplace=True)

#剔除18岁以下的
truma_with_icu = truma_with_icu[truma_with_icu['admission_age']>18].reset_index(drop=True)
#%%
print('subject id: %d'%(truma_with_icu['subject_id'].nunique()))
print('hadm id: %d'%(truma_with_icu['hadm_id'].nunique()))
print('icustays id: %d'%(truma_with_icu['icustay_id'].nunique()))
    


#%%
#读入每一次ICU的sofa
sofa = pd.read_csv('d://301datathon//sofa.csv')
truma_icu_sofa = truma_with_icu.merge(sofa, on=['subject_id', 'hadm_id', 'icustay_id'], how='inner')

#plot
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 
plt.hist(truma_icu_sofa['sofa'],bins=30)
#%%
#按照每一次入院的诊断中是否sepsis来聚合。
sepsis_cohort = pd.read_csv('d://301datathon//sepsis_cohort.csv')
sepsis_cohort = sepsis_cohort[['icustay_id', 'suspected_of_infection_poe']]

truma_icu_sofa_cohort = truma_icu_sofa.merge(sepsis_cohort, on=['icustay_id'], how='inner')
#%%
truma_icu_sofa_cohort['is_sepsis'] = truma_icu_sofa_cohort.apply(lambda x: x['sofa']>2 and x['suspected_of_infection_poe']==1, axis=1)
print('total %d sepsis in %d truma patients icu stays'%(np.sum(truma_icu_sofa_cohort['is_sepsis']), len(truma_icu_sofa_cohort)))

#%%
#身高、体重
con = psycopg2.connect(dbname=dbname, user=user, host=host, password=passwd)
cur = con.cursor()

cur.execute("select * from mimiciii.weightfirstday")
column_names = [desc[0] for desc in cur.description]
weight = pd.DataFrame(cur.fetchall(),columns=column_names)

cur.execute("select * from mimiciii.heightfirstday")
column_names = [desc[0] for desc in cur.description]
height = pd.DataFrame(cur.fetchall(),columns=column_names)

cur.execute("select * from mimiciii.rrtfirstday")
column_names = [desc[0] for desc in cur.description]
rrt = pd.DataFrame(cur.fetchall(),columns=column_names)

cur.execute("select * from mimiciii.uofirstday")
column_names = [desc[0] for desc in cur.description]
uo = pd.DataFrame(cur.fetchall(),columns=column_names)

cur.execute("select * from mimiciii.ventfirstday ")
column_names = [desc[0] for desc in cur.description]
vent = pd.DataFrame(cur.fetchall(),columns=column_names)

cur.execute("select * from mimiciii.gcsfirstday ")
column_names = [desc[0] for desc in cur.description]
gcs = pd.DataFrame(cur.fetchall(),columns=column_names)


cur.close()


#%%
#生化&血气
labs = pd.read_csv('d://301datathon//labsfirstday.csv')
bldgas = pd.read_csv('d://301datathon//bloodgasfirstdayarterial.csv')
#生命体征
vital = pd.read_csv('d://301datathon//vital.csv')

#读入用药
drug = pd.read_csv('d://301datathon//cohort_drug.csv')
#%%
#读入血气
gas = pd.read_csv('d://301datathon//bloodgas.csv')

ptrn = re.compile('.*subject.*')
ff = [f for f in gas.columns if ptrn.match(f)]
gas = gas.drop(ff, axis=1)


#%%
#关联生化
new_truma1 = truma_icu_sofa_cohort.merge(labs, on=['subject_id', 'hadm_id','icustay_id'], how='left')
#关联通气
new_truma2 = new_truma1.merge(vent, on=['subject_id', 'hadm_id','icustay_id'], how='left')
#关联生命体征
new_truma3 = new_truma2.merge(vital, on=['subject_id', 'hadm_id','icustay_id'], how='left')
#关联透析
new_truma4 = new_truma3.merge(rrt, on=['subject_id', 'hadm_id','icustay_id'], how='left')
#关联尿常规
new_truma5 = new_truma4.merge(uo, on=['subject_id', 'hadm_id','icustay_id'], how='left')
#关联gcs
new_truma6 = new_truma5.merge(gcs, on=['subject_id', 'hadm_id','icustay_id'], how='left')
#关联身高
new_truma7 = new_truma6.merge(height, on=['icustay_id'], how='left')
#关联体重
new_truma8 = new_truma7.merge(weight, on=['icustay_id'], how='left')

#关联用药
new_truma9 = new_truma8.merge(drug, on=['subject_id', 'hadm_id','icustay_id'], how='left')


#%%
#关联用血
#读入入ICU前手术是否输血
procedure_rbc = pd.read_csv('d://301datathon//transfusion_in_procudure.csv')
procedure_rbc = procedure_rbc[['icustay_id']]
procedure_rbc['procedure_rbc'] = 1
new_truma10 = new_truma9.merge(procedure_rbc, on=['icustay_id'], how='left')
new_truma10['procedure_rbc']=0

#读入ICU中是否输血
ICU_rbc = pd.read_csv('d://301datathon//inputu_rbc_intake.csv')
ICU_rbc = ICU_rbc[['icustay_id']]
ICU_rbc['ICU_rbc']=1
new_truma11 = new_truma10.merge(ICU_rbc, on=['icustay_id'], how='left')
new_truma11['ICU_rbc'] = 0


#%%
#关联血气
new_truma12 = new_truma11.merge(gas, on=['icustay_id'], how='left')
new_truma12.to_csv('d://301datathon//创伤患者关联特征后数据.csv')
joblib.dump(new_truma12, 'datathon.pkl')








