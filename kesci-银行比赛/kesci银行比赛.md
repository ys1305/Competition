## 导包读数据



```python
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from tqdm import tqdm
# %matplotlib inline
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")

train = pd.read_csv('./train_set.csv')
test = pd.read_csv('./test_set.csv')
data = pd.concat([train, test])
```



## 特征工程

```python
feature = data.columns.tolist()
feature.remove('ID')
feature.remove('y')
sparse_feature = ['campaign', 'contact', 'default', 'education',
                  'housing', 'job', 'loan', 'marital', 'month', 'poutcome']
dense_feature = list(set(feature) - set(sparse_feature))
# 区分出连续数据和离散数据（小窍门：直接根据数据的nunique来区分，

```



时间特征 

月份属于第几个季度，day在月份中的第几周

```python
data["month"] = data["month"].map({"jan": 1, "feb": 2, 'mar': 3, 'apr': 4, 
                                   'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                                   'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})
data['quater_of_year'] = data['month'].apply(lambda x : x//4)  # 季度
data['quater_of_month'] = data['day'].apply(lambda x : x//7)  # 周
```





### 特征处理1

```python
# 统计分组之后的频次
def feature_count(data, features):
    feature_name = 'count'
    for i in features:
        feature_name += '_' + i
    temp = data.groupby(features).size().reset_index().rename(
        columns={0: feature_name})
    data = data.merge(temp, 'left', on=features)
    return data, feature_name
    
    
ll = []
for f in['campaign', 'contact', 'default', 'education', 'housing', 'job', 'loan', 'marital', 'poutcome']:
    data, _ = feature_count(data, ['day', 'month', f])
    ll.append(_)
```



### 特征处理2

```python
def get_new_columns(name, aggs):
    l = []
    for k in aggs.keys():
        for agg in aggs[k]:
            if str(type(agg)) == "<class 'function'>":
                l.append(name + '_' + k + '_' + 'other')
            else:
                l.append(name + '_' + k + '_' + agg)
    return l


for d in tqdm(sparse_feature):
    aggs = {}
    for s in sparse_feature:
        aggs[s] = ['count', 'nunique']
    for den in dense_feature:
        aggs[den] = ['mean', 'max', 'min', 'std']
    aggs.pop(d)
    temp = data.groupby(d).agg(aggs).reset_index()
    # 修改列名，不修改前的列名为MultiIndex
    temp.columns = [d] + get_new_columns(d, aggs)
    data = pd.merge(data, temp, on=d, how='left')
```

#### agg测试

```python
data = pd.read_csv('./seaborn-data-master/tips.csv')
data['tip_pct'] = data['tip'] / data['total_bill']
grouped = data.groupby('day')

functions = ['count', 'mean', 'max']
# 取出来两列进行操作
result = grouped['tip_pct', 'total_bill'].agg(functions)
print(type(result))
print(result)

# 使用字典可以对不同的列进行特定的函数操作 使用列表则是对所有的列进行相同的函数操作
grouped.agg({'tip_pct' : ['min', 'max', 'mean', 'std'],'size' : 'sum'})
```





### 特征处理3-onehot编码

```python
for s in ['campaign', 'contact', 'default', 'education', 'housing', 'job', 'loan', 'marital', 'month', 'poutcome']:
    data = pd.concat([data, pd.get_dummies(data[s], prefix=s + '_')], axis=1)
    data.drop(s, axis=1, inplace=True)
```



### 其他处理





## 拆分数据集

```python
df_train = data[data['y'].notnull()]
df_test = data[data['y'].isnull()]
# 贴程序是加了一行test = -1,导致这里拆分数据集时错误。导致了使用Kfold时出现不匹配的问题
target = df_train['y']
df_train_columns = df_train.columns.tolist()
df_train_columns.remove('ID')
df_train_columns.remove('y')
```



## 数据标准化

加上数据标准化后效果反而会有所下降

应该是将onehot类型的变量进行归一化后变成了稠密矩阵，改变了原来的含义

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
mms = StandardScaler()
df_train[df_train_columns] = mms.fit_transform(df_train[df_train_columns])
df_train[df_train_columns].describe()
df_test[df_train_columns]= mms.transform(df_test[df_train_columns])

```







## 特征选择

#### 注意

```python
df_train[df_train[df_train_columns].isnull().values==True]
# 会有nan的出现，而特征选择的时候不能出现nan 
# lgb可以处理缺失值，但特征选择不能处理

# 缺失值填补
df_train[df_train_columns] = df_train[df_train_columns].fillna(df_train[df_train_columns].mean())
```



```python
import lightgbm as lgb
from sklearn.feature_selection import RFE

model = lgb.LGBMClassifier(
        boosting_type="gbdt", num_leaves=30, reg_alpha=0, reg_lambda=0.,
    max_depth=-1, n_estimators=600, objective='binary',metric= 'auc',
    subsample=0.85, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.02, random_state=2019
    )
from sklearn.model_selection import cross_val_score
score = []
for i in range(1,500,50):
    X_wrapper = RFE(model,n_features_to_select=i, step=50).fit_transform(df_train[df_train_columns], target)
    once = cross_val_score(model,X_wrapper,target,cv=5,scoring='roc_auc').mean()
    print('特征数量:',i,',auc:',once)
    score.append(once)

```



```
未进行标准化
特征数量: 1 ,auc: 0.7935873254768542
特征数量: 51 ,auc: 0.9410002321749318
特征数量: 101 ,auc: 0.9405527394912129
特征数量: 151 ,auc: 0.9405328823623723
特征数量: 201 ,auc: 0.9408284102248168
特征数量: 251 ,auc: 0.9405677873055179
特征数量: 301 ,auc: 0.9407322636247646
特征数量: 351 ,auc: 0.940533122705326
特征数量: 401 ,auc: 0.9406939289039478
特征数量: 451 ,auc: 0.9404385124993364

进行标准化后结果反而变差
特征数量: 1 ,auc: 0.7944185772558552
特征数量: 51 ,auc: 0.9404873787210775
特征数量: 101 ,auc: 0.9401977211266516
特征数量: 151 ,auc: 0.9403600475119557
特征数量: 201 ,auc: 0.9401212847240394
特征数量: 251 ,auc: 0.9402594711308122
特征数量: 301 ,auc: 0.9401619530151825
```



### 精细选择

```python
score = []
for i in range(30,76,5):
    X_wrapper = RFE(model,n_features_to_select=i, step=20).fit_transform(df_train[df_train_columns], target)
    once = cross_val_score(model,X_wrapper,target,cv=5,scoring='roc_auc').mean()
    print('特征数量:',i,',auc:',once)
    score.append(once)
# 未标准化
# 特征数量: 25 ,auc: 0.9392600590373865
# 特征数量: 30 ,auc: 0.9409825802863147
# 特征数量: 35 ,auc: 0.9412280205803961
# 特征数量: 40 ,auc: 0.9410969648019115
# 特征数量: 45 ,auc: 0.9409206267843725
# 特征数量: 50 ,auc: 0.9410097738663016
# 特征数量: 55 ,auc: 0.9408808260329253
# 特征数量: 60 ,auc: 0.9407293019878438
# 特征数量: 65 ,auc: 0.9405286498745273
# 特征数量: 70 ,auc: 0.940852444429453
```



### 具体选择

```python
rfe = RFE(model,40,step=10)
# step 默认为1，每次只剔除1个特征。速度将会非常慢
rfe = rfe.fit(df_train[df_train_columns], target)


print(rfe.support_)
print(rfe.ranking_)

df_train[df_train_columns].iloc[:,rfe.support_].head()
X_wrapper = rfe.transform(df_train[df_train_columns])
X_wrapper
# 可以比较二者其实是相同的  
# X_wrapper and df_train[df_train_columns].iloc[:,rfe.support_].head()

num_features = df_train[df_train_columns].iloc[:,rfe.support_].columns.tolist()
```



## xgboost

```python
df_train = pd.read_csv('./datalab/27890/train_std.csv')
df_test = pd.read_csv('./datalab/27890/test_std.csv')

df_train_columns=num_features
# num_features为经过选择后的特征的名称

import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import XGBClassifier as XGBC



folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=666)
predictions = np.zeros(len(test_std))

param={'booster':'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth':5,
    'gamma': 0.1,
    'lambda':1,
    'subsample':0.75,
    'colsample_bytree':0.75,
    'min_child_weight':2,
    'eta': 0.025,
    'seed':888,
    'nthread':8,
    'silent':1,
    "scale_pos_weight":1
}
# split() missing 1 required positional argument: 'y'
for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train, df_train['y'].values)):
    print("fold {}".format(fold_))
    trn_data = xgb.DMatrix(
        df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])
    val_data = xgb.DMatrix(
        df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])

    num_round = 10000
    watchlist = [(trn_data,'train'),(val_data,'val')]
    clf = xgb.train(param, trn_data, num_round, evals=watchlist, verbose_eval=100, early_stopping_rounds=100)
    # verbose_eval=100 每100轮打印一次auc

    predictions += clf.predict(xgb.DMatrix(df_test[df_train_columns])) / folds.n_splits

    
sub = df_test[['ID']]
sub['pred'] = predictions
sub.to_csv(path+'/xgb-10-std-Result.csv', index=False)
```



## lgb



```python
train_x = df_train[num_features]
test_x = df_test[num_features]

res = test[['ID']]


res['pred'] = 0
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
for train_idx, val_idx in kfold.split(train_x):
    clf.random_state = clf.random_state + 1
    train_x1 = train_x.iloc[train_idx]
    train_y1 = target.iloc[train_idx]
    test_x1 = train_x.iloc[val_idx]
    test_y1 = target.iloc[val_idx]
    #,(vali_x,vali_y)
    clf.fit(train_x1, train_y1, eval_set=[
            (train_x1, train_y1), (test_x1, test_y1)], eval_metric='auc', early_stopping_rounds=100, verbose=False)
    res['pred'] += clf.predict_proba(test_x)[:, 1]

# StratifiedKFold用法类似Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
res['pred'] = 0
for train_idx, val_idx in kfold.split(train_x, df_train['y'].values):
    clf.random_state = clf.random_state + 1
    train_x1 = train_x.iloc[train_idx]
    train_y1 = target.iloc[train_idx]
    test_x1 = train_x.iloc[val_idx]
    test_y1 = target.iloc[val_idx]
    clf.fit(train_x1, train_y1,
            eval_set=[(train_x1, train_y1), (test_x1, test_y1)],
            eval_metric='auc', early_stopping_rounds=100, verbose=False)
    res['pred'] += clf.predict_proba(test_x)[:, 1]

res['pred'] = res['pred'] / 10
res.to_csv('./res/lgb-50-10-SK.csv', index=False)
```



### 原生lgb

```python
param = {'num_leaves': 31,
         'min_data_in_leaf': 30, 
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.01,
         "min_child_samples": 20,
         "boosting": "gbdt",
         "feature_fraction": 0.9,
         "bagging_freq": 1,
         "bagging_fraction": 0.9 ,
         "bagging_seed": 11,
         "metric": 'auc',
         "lambda_l1": 0.1,
         "verbosity": -1,
         "nthread": 4,
         "random_state": 666}
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=666)
oof = np.zeros(len(df_train))
predictions = np.zeros(len(df_test))
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(df_train,df_train['y'].values)):
    print("fold {}".format(fold_))
    trn_data = lgb.Dataset(df_train.iloc[trn_idx][df_train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)
    val_data = lgb.Dataset(df_train.iloc[val_idx][df_train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)

    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)
    oof[val_idx] = clf.predict(df_train.iloc[val_idx][df_train_columns], num_iteration=clf.best_iteration)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = df_train_columns
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    predictions += clf.predict(df_test[df_train_columns], num_iteration=clf.best_iteration) / folds.n_splits
    

from sklearn.metrics import roc_auc_score
roc_auc_score(target,oof)

sub=df_test[['ID']]
sub['pred']=predictions
sub.to_csv('./Result.csv',index=False)
```



## 词频统计

加上对于auc的效果其实没啥提升

```python
# 获取向量化特征
data['new_con'] = data['job'].astype(str)

for i in ['marital', 'education', 'contact','month','poutcome']:
    data['new_con'] = data['new_con'].astype(str) + '_' + data[i].astype(str)
data['new_con']
#0          management_married_tertiary_unknown_may_unknown
#1           technician_divorced_primary_cellular_apr_other
#2            admin._married_secondary_cellular_jul_unknown
#3         management_single_secondary_cellular_jul_unknown
#4        technician_divorced_secondary_unknown_may_unknown
#5          services_divorced_secondary_unknown_jun_unknown
data['new_con'] = data['new_con'].apply(lambda x: ' '.join(x.split('_')))
data['new_con']
# 空格 替换 _

#0          management married tertiary unknown may unknown
#1           technician divorced primary cellular apr other
#2            admin. married secondary cellular jul unknown
#3         management single secondary cellular jul unknown
#4        technician divorced secondary unknown may unknown
#5          services divorced secondary unknown jun unknown

train_x=df_train[num_features]
test_x=df_test[num_features]
# 此时特征数量为50

vector_feature = ['new_con']
cv=CountVectorizer()
for feature in vector_feature:
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    test_a = cv.transform(test[feature])


    
train_a
<25317x87 sparse matrix of type '<class 'numpy.float64'>'
	with 1392880 stored elements in Compressed Sparse Row format>

cv.vocabulary_
# 词汇表；字典型
cv.get_feature_names()
len(cv.get_feature_names())
# 37
train_a.toarray()
# 一共37个词，生成一个样本中每个词出现的次数

df_a = pd.DataFrame(train_a.toarray())
# 生成列名
fis = cv.get_feature_names()
sec = []
for i in fis:
    i = 'count_'+i
    sec.append(i)
df_a.columns = sec

df_b = pd.DataFrame(test_a.toarray(),columns=sec)
# 这里会遇到一个坑,df_b的index与test_b的index不同，不能直接拼接
# 需要先将df_b与test_b的index变得一致
df_b.index = test_x.index
# 拼接数据
train_x = pd.concat([train_x,df_a],axis=1)
test_x = pd.concat([test_x,df_b],axis=1)

# 试了下，结果也没有什么提升

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=666)

res['pred'] = 0
clf = lgb.LGBMClassifier(
    boosting_type="gbdt", num_leaves=30, reg_alpha=0, reg_lambda=0.,
    max_depth=-1, n_estimators=1000, objective='binary', metric='auc',
    subsample=0.85, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.02, random_state=2019
)

for train_idx, val_idx in kfold.split(train_x, df_train['y'].values):
    clf.random_state = clf.random_state + 1
    train_x1 = train_x.iloc[train_idx]
    train_y1 = target.iloc[train_idx]
    test_x1 = train_x.iloc[val_idx]
    test_y1 = target.iloc[val_idx]
    clf.fit(train_x1, train_y1,
            eval_set=[(train_x1, train_y1), (test_x1, test_y1)],
            eval_metric='auc', early_stopping_rounds=100, verbose=100)
    # 这里verbose =100 是为了每过100次打印一下指标
    res['pred'] += clf.predict_proba(test_x)[:, 1]
```













```python
# 获取向量化特征
data['new_con'] = data['job'].astype(str)

for i in ['marital', 'education', 'contact','month','poutcome']:
    data['new_con'] = data['new_con'].astype(str) + '_' + data[i].astype(str)
data['new_con']
#0          management_married_tertiary_unknown_may_unknown
#1           technician_divorced_primary_cellular_apr_other
#2            admin._married_secondary_cellular_jul_unknown
#3         management_single_secondary_cellular_jul_unknown
#4        technician_divorced_secondary_unknown_may_unknown
#5          services_divorced_secondary_unknown_jun_unknown
data['new_con'] = data['new_con'].apply(lambda x: ' '.join(x.split('_')))
data['new_con']
# 空格 替换 _

#0          management married tertiary unknown may unknown
#1           technician divorced primary cellular apr other
#2            admin. married secondary cellular jul unknown
#3         management single secondary cellular jul unknown
#4        technician divorced secondary unknown may unknown
#5          services divorced secondary unknown jun unknown

train_x=df_train[num_features]
test_x=df_test[num_features]
# 此时特征数量为50

vector_feature = ['new_con']
cv=CountVectorizer()
for feature in vector_feature:
    cv.fit(data[feature])
    train_a = cv.transform(train[feature])
    test_a = cv.transform(test[feature])
    train_x = sparse.hstack((train_x, train_a), 'csr')
    # 拼接成稀疏矩阵
    test_x = sparse.hstack((test_x, test_a), 'csr')

train_a
<25317x87 sparse matrix of type '<class 'numpy.float64'>'
	with 1392880 stored elements in Compressed Sparse Row format>
    
cv.vocabulary_
# 词汇表；字典型

cv.get_feature_names()
len(cv.get_feature_names())
# 37

train_a.toarray()
# 一共37个词，生成一个样本中每个词出现的次数
```



```python
import lightgbm as lgb
n_splits=10
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
res=test[['ID']]
res['pred'] = 0
from time import time
from sklearn.metrics import roc_auc_score
start = time()
pred_valid = pd.DataFrame()
# 这里的train_x不是dataframe类型，不能进行.iloc
for train_idx, val_idx in kfold.split(train_x,df_train['y'].values):
    clf.random_state = clf.random_state + 1
    train_x1 = train_x[train_idx]
    train_y1 = target.iloc[train_idx]
    valid_x1 = train_x[val_idx]
    valid_y1 = target.iloc[val_idx]
    #,(vali_x,vali_y)
    clf.fit(train_x1, train_y1,eval_set=[(train_x1, train_y1),(valid_x1, valid_y1)],eval_metric='auc',early_stopping_rounds=100, verbose=False)
    res['pred'] += clf.predict_proba(test_x)[:,1]
    pred_valid1 = pd.DataFrame({'Id':valid_y1.index, 'pred':clf.predict_proba(valid_x1)[:,1], 'y':valid_y1.values})
    pred_valid = pred_valid.append(pred_valid1).reset_index(drop=True)
    
end = time()

auc = roc_auc_score(pred_valid['y'], pred_valid['pred'])
print('验证集得分为：{}'.format(auc))
print('训练时间为：{} min'.format((end-start)/60))
```


