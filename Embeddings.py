# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random as rd # generating random numbers
import datetime # manipulating date formats
import matplotlib.pyplot as plt # basic plotting
import seaborn as sns # for prettier plots
import os
pd.set_option('display.max_columns', None)
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
from itertools import product
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from xgboost import XGBRegressor
from xgboost import plot_importance
import time
import sys
import gc
import pickle
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Activation
from keras.optimizers import SGD, Adam
from sklearn.preprocessing import StandardScaler


items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')

train = train[train.item_price<100000]
train = train[train.item_cnt_day<1001]

median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
train.loc[train.item_price<0, 'item_price'] = median


train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57

train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58

train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11

shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops = shops[['shop_id','city_code']]

cats['split'] = cats['item_category_name'].str.split('-')
cats['type'] = cats['split'].map(lambda x: x[0].strip())
cats['type_code'] = LabelEncoder().fit_transform(cats['type'])
# if subtype is nan then type
cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])
cats = cats[['item_category_id','type_code', 'subtype_code']]

items.drop(['item_name'], axis=1, inplace=True)

cats.head()

shops.head()

ts = time.time()
matrix = []
cols = ['date_block_num','shop_id','item_id']
for i in range(34):
    sales = train[train.date_block_num==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
    #her month için bütün shop_id ve item_id combinasyonu oluşturulması

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)
matrix.sort_values(cols,inplace=True)
time.time() - ts

train['revenue'] = train['item_price'] *  train['item_cnt_day']

ts = time.time()
group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0,20)
                                .astype(np.float16))
time.time() - ts

matrix.head()

group.head()

test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)

test.head()

ts = time.time()
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) # 34 month
time.time() - ts

matrix.tail()

ts = time.time()
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
matrix['city_code'] = matrix['city_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['type_code'] = matrix['type_code'].astype(np.int8)
matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)
time.time() - ts

matrix.head()

#bir sonraki aylar

def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df

ts = time.time()
matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')
time.time() - ts

matrix[matrix['date_block_num']==1].head()

#Ürünün ilgili ayda shop başına ortalama satışı

ts = time.time()
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_item_avg_item_cnt')
matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

#Lokasyonun ilgili ayda ürün başına ortalama satışı

ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_shop_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_shop_avg_item_cnt')
matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

#Kategorinin ilgili ayda toplam satışı

ts = time.time()
group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['sum']})
group.columns = [ 'date_cat_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_category_id'], how='left')
matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_cat_avg_item_cnt')
matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

#Kategorinin ilgili ayda lokasyon bazında ortalama satışı
ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_cat_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
matrix['date_shop_cat_avg_item_cnt'] = matrix['date_shop_cat_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_cat_avg_item_cnt')
matrix.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts

#Kategorinin ilgili ayda lokasyon bazında toplam satışı
ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['sum']})
group.columns = ['date_shop_cat_tot_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
matrix['date_shop_cat_tot_item_cnt'] = matrix['date_shop_cat_tot_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_cat_tot_item_cnt')
matrix.drop(['date_shop_cat_tot_item_cnt'], axis=1, inplace=True)
time.time() - ts

#Aylık toplam satış

ts = time.time()
group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['sum']})
group.columns = [ 'date_tot_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
matrix['date_tot_item_cnt'] = matrix['date_tot_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_tot_item_cnt')
matrix.drop(['date_tot_item_cnt'], axis=1, inplace=True)
time.time() - ts

matrix.head()

matrix['month'] = matrix['date_block_num'] % 12

days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)

matrix['month'] = matrix['month']+1


saturday= pd.read_csv('../input/saturday/saturdays.csv')

matrix = pd.merge(matrix, saturday, on= ['date_block_num'], how = 'left')

matrix.tail()

ts = time.time()
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)
    return df

matrix = fill_na(matrix)
time.time() - ts

matrix.columns





matrix = matrix[matrix.date_block_num > 11]

X_train = matrix[matrix.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = matrix[matrix.date_block_num < 33]['item_cnt_month']
X_valid = matrix[matrix.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = matrix[matrix.date_block_num == 33]['item_cnt_month']
X_test = matrix[matrix.date_block_num == 34].drop(['item_cnt_month'], axis=1)

#del matrix
del group
del items
del shops
del cats
del train
# leave test for submission
gc.collect();

embed_cols=['month','date_block_num', 'shop_id', 'item_id', 'city_code', 'item_category_id', 'type_code', 'subtype_code']



numeric_columns = [each for each in matrix.columns if each not in embed_cols ]
del numeric_columns[0]


scaler = StandardScaler()
#X_train.drop('date_tot_item_cnt_lag_1', inplace=True, axis=1)
matrix[numeric_columns] = scaler.fit_transform(matrix[numeric_columns])

for each in numeric_columns:
    matrix[each] = matrix[each].astype(np.float16)

for each in embed_cols:

    group = matrix.groupby([each]).agg({'item_cnt_month': ['mean']})
    group.columns = [ each + 'y']
    group.reset_index(inplace=True)
    group['C'] = np.arange(len(group))

    no_of_unique_cat  = matrix[each].nunique()
    embedding_size = int(min(np.ceil((no_of_unique_cat)/2), 15 ))

    model = Sequential()
    model.reset_states( )
    model.add(Embedding(input_dim=no_of_unique_cat, output_dim=embedding_size, input_shape=(1,)))
    model.add(Flatten())
    model.add(Dense(units=1))
    learning_rate = 0.1
    if each =='item_id': learning_rate = 0.01
    sgd = Adam(lr=learning_rate)
    model.compile(optimizer=sgd, loss="mean_squared_error", metrics=["accuracy"])

    model.fit(group['C'], group[each + 'y'], epochs=25);
    df = pd.DataFrame(model.layers[0].get_weights()[0],columns=[each + str(a) for a in range(model.layers[0].get_weights()[0].shape[1])])
    temp = pd.concat([df,group], axis=1)
    temp.drop(['C', each +'y'], inplace=True, axis=1)
    temp[temp.columns] = temp[temp.columns].astype(np.float16)
    matrix = pd.merge(matrix, temp, on=each, how='left')
    if each!= 'date_block_num' : matrix.drop(each, inplace=True, axis=1)



#  import sys
# # These are the usual ipython objects, including this one you are creating
# ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# # Get a sorted list of the objects and their sizes
# sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)



lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_valid, Y_valid, reference=lgb_train)

params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                valid_sets=lgb_eval,
                early_stopping_rounds=10)

y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

y_pred= y_pred.clip(0,20)

submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": y_pred
})
submission.to_csv('lightgbm_nat_submission.csv', index=False)

submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": np.floor(y_pred)
})
submission.to_csv('lightgbm_floor_submission.csv', index=False)

submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": np.ceil(y_pred)
})
submission.to_csv('lightgbm_ceil_submission.csv', index=False)

submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": np.round(y_pred)
})
submission.to_csv('lightgbm_round_submission.csv', index=False)
