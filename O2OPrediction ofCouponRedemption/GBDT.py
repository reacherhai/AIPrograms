import pandas as pd
from sklearn import ensemble
import xgboost as xgb
from sklearn import preprocessing
import  numpy
import joblib

from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('data/dataset1.csv')

dataset1.label.replace(-1, 0, inplace=True)

dataset2 = pd.read_csv('data/dataset2.csv')

dataset2.label.replace(-1, 0, inplace=True)

dataset3 = pd.read_csv('data/dataset3.csv')

dataset1.drop_duplicates(inplace=True)

dataset2.drop_duplicates(inplace=True)

dataset3.drop_duplicates(inplace=True)

dataset12 = pd.concat([dataset1, dataset2], axis=0)

dataset1_y = dataset1.label

dataset1_x = dataset1.drop(['user_id', 'label', 'day_gap_before', 'day_gap_after'],
                           axis=1)  # 'day_gap_before','day_gap_after' cause overfitting, 0.77

dataset2_y = dataset2.label

dataset2_x = dataset2.drop(['user_id', 'label', 'day_gap_before', 'day_gap_after'], axis=1)

dataset12_y = dataset12.label

dataset12_x = dataset12.drop(['user_id', 'label', 'day_gap_before', 'day_gap_after'], axis=1)

dataset3_preds = dataset3[['user_id', 'coupon_id', 'date_received']]

dataset3_x = dataset3.drop(['user_id', 'coupon_id', 'date_received', 'day_gap_before', 'day_gap_after'], axis=1)

print( dataset1_x.shape, dataset2_x.shape, dataset3_x.shape , dataset12_x.shape)

scalar = preprocessing.StandardScaler()
features = scalar.fit_transform(dataset12_x)
features = numpy.nan_to_num(features)


clf = ensemble.GradientBoostingClassifier(max_depth =6, learning_rate =0.05,n_estimators=200 )
clf.fit(features, dataset12_y)  # Training model
dataset3_x = scalar.fit_transform(dataset3_x)
dataset3_x = numpy.nan_to_num(dataset3_x)
temp = clf.predict_proba(dataset3_x)[:,1]
print(str(temp))
joblib.dump(clf,'gbdt.pkl')
clf = joblib.load('gbdt.pkl')
#dataset3_preds = clf.predict(dataset3_x)[:,1]  # predict
# 包含的参数和上面一致。

dataset3_preds['label'] = temp
#print(str(temp))

dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label.values.reshape(-1, 1))

dataset3_preds.sort_values(by=['coupon_id', 'label'], inplace=True)

dataset3_preds.to_csv("GBDT_preds.csv", index=None, header=None)

'''
dataset1 = xgb.DMatrix(dataset1_x, label=dataset1_y)

dataset2 = xgb.DMatrix(dataset2_x, label=dataset2_y)

dataset12 = xgb.DMatrix(dataset12_x, label=dataset12_y)

dataset3 = xgb.DMatrix(dataset3_x)

params = {'booster': 'gbtree',

          'objective': 'rank:pairwise',

          'eval_metric': 'auc',

          'gamma': 0.1,

          'min_child_weight': 1.1,

          'max_depth': 5,

          'lambda': 10,

          'subsample': 0.7,

          'colsample_bytree': 0.7,

          'colsample_bylevel': 0.7,

          'eta': 0.01,

          'tree_method': 'exact',

          'seed': 0,

          'nthread': 12

          }

# train on dataset1, evaluate on dataset2

# watchlist = [(dataset1,'train'),(dataset2,'val')]

# model = xgb.train(params,dataset1,num_boost_round=3000,evals=watchlist,early_stopping_rounds=300)


watchlist = [(dataset12, 'train')]

model = xgb.train(params, dataset12, num_boost_round=3500, evals=watchlist)

# predict test set

dataset3_preds['label'] = model.predict(dataset3)

dataset3_preds.label = MinMaxScaler().fit_transform(dataset3_preds.label.values.reshape(-1, 1))

dataset3_preds.sort_values(by=['coupon_id', 'label'], inplace=True)

dataset3_preds.to_csv("xgb_preds.csv", index=None, header=None)

'''