# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 18:54:25 2018

@author: shashank
"""

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import xgboost as xgb

path = r''

trainData = pd.read_csv(path+r'\train.csv')

testData = pd.read_csv(path+r'\test.csv')

'''Feature Processing'''

def FeatureProcessing(df):
    
    #gender to binary
    df['Gender'][df['Gender']=='M'] = 1
    df['Gender'][df['Gender']=='F'] = 0
    df['Gender'] = df['Gender'].map(int)
    
    #age to numeric
    df['Age'][df['Age']=="0-17"] = 15
    df['Age'][df['Age']=="18-25"] = 21
    df['Age'][df['Age']=="26-35"] = 30
    df['Age'][df['Age']=="36-45"] = 40
    df['Age'][df['Age']=="46-50"] = 48
    df['Age'][df['Age']=="51-55"] = 53
    df['Age'][df['Age']=="55+"] = 60
    df['Age'] = df['Age'].map(int)
    
    #Stay_In_Current_City_Years to numeric
    df['Stay_In_Current_City_Years'][df['Stay_In_Current_City_Years']=='4+'] = 4
    df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].map(int)
    
    #city_category to one-hot
    df = getDummies('City_Category',df)
    
    return df


''' one hot encoding of categorical variables '''
def getDummies(columnName,df):
    dummyTemp = pd.get_dummies(df[columnName],prefix=columnName)
    df = pd.concat([df,dummyTemp],axis=1)
    return df

trainData = FeatureProcessing(trainData)
testData = FeatureProcessing(testData)

#using Product average purchase per user as a feature
userAvgPurchase = trainData.groupby('User_ID').agg({'Purchase':np.mean}).reset_index().rename(columns={'Purchase':'User_Avg_Purchase'})
trainData = pd.merge(trainData,userAvgPurchase,on='User_ID',how='left')
testData = pd.merge(testData,userAvgPurchase,on='User_ID',how='left')

#using average purchase per product as a feature
productAvgPurchase = trainData.groupby('Product_ID').agg({'Purchase':np.mean}).reset_index().rename(columns={'Purchase':'Product_Avg_Purchase'})
trainData = pd.merge(trainData,productAvgPurchase,on='Product_ID',how='left')
testData = pd.merge(testData,productAvgPurchase,on='Product_ID',how='left')

colsForPrediction = [ u'Gender', u'Age', u'Occupation', u'Stay_In_Current_City_Years',
                     u'Marital_Status',u'Product_Category_1', u'Product_Category_2',
                     u'Product_Category_3',u'City_Category_A', u'City_Category_B',
                     u'City_Category_C','User_Avg_Purchase','Product_Avg_Purchase']


''' Average of 3 XGB Models'''

def xgboostModel(depth,trees,iterations):
    xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9, 
            'max_depth': depth, 'subsample': 0.9, 'lambda': 1., 'nthread': -1,
            'booster' : 'gbtree', 'silent': 1, 'eval_metric': 'rmse',
            'objective': 'reg:linear','n_estimators':trees}
    model = xgb.train(xgb_pars, dtrain, iterations, watchlist, early_stopping_rounds=2, maximize=False, verbose_eval=1)   
    return model

# fit model no training data
model = XGBClassifier()

X_train,y_train = trainData[colsForPrediction],trainData['Purchase']
X_test = testData[colsForPrediction]

dtrain = xgb.DMatrix(X_train, label=y_train)

dtest = xgb.DMatrix(X_test)
watchlist = [(dtrain, 'train')]


#model 1 (depth:8 trees:1450 iterations:20)
model1 = xgboostModel(8,1450,20)

#model 2 (depth:12 trees:800 iterations:20)
model2 = xgboostModel(12,800,20)

#model 3 (depth:6 trees:3000 iterations:35)
model3 = xgboostModel(6,3000,35)


y_pred1 = model1.predict(dtest)
y_pred2 = model2.predict(dtest)
y_pred3 = model3.predict(dtest)


testData['Purchase'] = (y_pred1 + y_pred2 + y_pred3) / 3

testData['Purchase'].plot(kind='hist')

testData = testData[['User_ID','Product_ID','Purchase']]
testData.to_csv(path+r'\testOutputXGB_AVG.csv',index=False)


