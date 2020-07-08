# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 18:54:25 2018

@author: shashank
"""

''' Problem Statement:
    A retail company “ABC Private Limited” wants to understand the customer purchase 
    behaviour (specifically, purchase amount) against various products of different
    categories. They have shared purchase summary of various customers for selected 
    high volume products from last month.
    The data set also contains customer demographics (age, gender, marital status,/
    city_type, stay_in_current_city), product details (product_id and product category)
    and Total purchase_amount from last month.
    Now, they want to build a model to predict the purchase amount of customer 
    against various products which will help them to create personalized offer 
    for customers against different products.
    https://datahack.analyticsvidhya.com/contest/black-friday'''

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from xgboost import XGBClassifier
import xgboost as xgb
import matplotlib.pyplot as plt

path = r'C:\Users\nardekars.BSG\Documents\icarus-master\br personal\study\analytics_vidhya_projects\\'

trainData = pd.read_csv(path+r'\train.csv')

trainData.head(10)

trainData.describe()

trainData.info()


#trainData = trainData.fillna(0.0)

#trainData['Age'].value_counts().plot(kind='bar')
#trainData['Gender'].value_counts().plot(kind='bar')
#trainData['Occupation'].value_counts().plot(kind='bar')
#trainData['City_Category'].value_counts().plot(kind='bar')
#trainData['Stay_In_Current_City_Years'].value_counts().plot(kind='bar')
#trainData['Product_Category_1'].value_counts().plot(kind='bar')
#trainData['Product_Category_2'].value_counts().plot(kind='bar')
#trainData['Product_Category_3'].value_counts().plot(kind='bar')
#trainData['Purchase'].plot(kind='hist')


#gender to binary
trainData['Gender'][trainData['Gender']=='M'] = 1
trainData['Gender'][trainData['Gender']=='F'] = 0
trainData['Gender'] = trainData['Gender'].map(int)

#age to numeric
trainData['Age'][trainData['Age']=="0-17"] = 15
trainData['Age'][trainData['Age']=="18-25"] = 21
trainData['Age'][trainData['Age']=="26-35"] = 30
trainData['Age'][trainData['Age']=="36-45"] = 40
trainData['Age'][trainData['Age']=="46-50"] = 48
trainData['Age'][trainData['Age']=="51-55"] = 53
trainData['Age'][trainData['Age']=="55+"] = 60
trainData['Age'] = trainData['Age'].map(int)

#Stay_In_Current_City_Years to numeric
trainData['Stay_In_Current_City_Years'][trainData['Stay_In_Current_City_Years']=='4+'] = 4
trainData['Stay_In_Current_City_Years'] = trainData['Stay_In_Current_City_Years'].map(int)


''' one hot encoding of categorical variables '''
def getDummies(columnName,trainData):
    dummyTemp = pd.get_dummies(trainData[columnName],prefix=columnName)
    trainData = pd.concat([trainData,dummyTemp],axis=1)
    return trainData

#city_category to one-hot
trainData = getDummies('City_Category',trainData)

#using Product average purchase per user as a feature
userAvgPurchase = trainData.groupby('User_ID').agg({'Purchase':np.mean}).reset_index().rename(columns={'Purchase':'User_Avg_Purchase'})
trainData = pd.merge(trainData,userAvgPurchase,on='User_ID',how='left')

#using average purchase per product as a feature
productAvgPurchase = trainData.groupby('Product_ID').agg({'Purchase':np.mean}).reset_index().rename(columns={'Purchase':'Product_Avg_Purchase'})
trainData = pd.merge(trainData,productAvgPurchase,on='Product_ID',how='left')

trainData.columns


''' RANDOM SAMPLING FOR TRAIN AND TEST '''
trainData = trainData.sample(frac=1).reset_index(drop=True)

colsForPrediction = [ u'Gender', u'Age', u'Occupation', u'Stay_In_Current_City_Years',
                     u'Marital_Status',u'Product_Category_1', u'Product_Category_2',
                     u'Product_Category_3',u'City_Category_A', u'City_Category_B',
                     u'City_Category_C','User_Avg_Purchase','Product_Avg_Purchase']

trainDataTrain = trainData[0:440054]
trainDataValidate = trainData[440055:550068]

''' XGB '''

def xgboostModel(depth,trees,iterations):
    xgb_pars = {'min_child_weight': 1, 'eta': 0.5, 'colsample_bytree': 0.9, 
            'max_depth': depth, 'subsample': 0.9, 'lambda': 1., 'nthread': -1,
            'booster' : 'gbtree', 'silent': 1, 'eval_metric': 'rmse',
            'objective': 'reg:linear','n_estimators':trees}
    model = xgb.train(xgb_pars, dtrain, iterations, watchlist, early_stopping_rounds=2, maximize=False, verbose_eval=1)   
    return model

# fit model no training data
model = XGBClassifier()

X_train,y_train = trainDataTrain[colsForPrediction],trainDataTrain['Purchase']
X_test,y_test = trainDataValidate[colsForPrediction],trainDataValidate['Purchase']

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
watchlist = [(dtrain, 'train'), (dtest, 'test')]


#model 1 (depth:8 trees:1450 iterations:20)
model1 = xgboostModel(8,1450,20)

#model 2 (depth:12 trees:800 iterations:20)
model2 = xgboostModel(12,800,20)


#model 3 (depth:6 trees:3000 iterations:35)
model3 = xgboostModel(6,3000,35)


print('Modeling RMSLE %.5f' % model1.best_score)
print('Modeling RMSLE %.5f' % model2.best_score)
print('Modeling RMSLE %.5f' % model3.best_score)

y_pred1 = model1.predict(dtest)
y_pred2 = model2.predict(dtest)
y_pred3 = model3.predict(dtest)


trainDataValidate['Purchase_Predicted_1'] = y_pred1
trainDataValidate['Purchase_Predicted_2'] = y_pred2
trainDataValidate['Purchase_Predicted_3'] = y_pred3

trainDataValidate['Purchase_Predicted_XGB_avg'] = (trainDataValidate['Purchase_Predicted_1'] + trainDataValidate['Purchase_Predicted_2'] + trainDataValidate['Purchase_Predicted_3']) / 3

sqrt(mean_squared_error(trainDataValidate['Purchase'],trainDataValidate['Purchase_Predicted_XGB_avg']))

#model1 : 2966.21 || 2922.16 || 2917.36 (8,1450) office || 2546.12 office || 2486.04 office || 2486.04 office 
#model2 : 2970.96 || 2948.76 || 2953.09 (12,800) office || 2551.82 office || 2500.00 office || 2500.00 office
#mode3 : 2986.19 || 2930.26 || 2925.07 (6,3000) office || 2580.66 office || 2497.84 office || 2487.13 office
#model_avg : 2947.39 || 2908.17 || 2903.03 office     || 2520.62 office || 2466.48 office || 2463.16 office

trainDataValidate['Purchase'].plot(kind='hist')

trainDataValidate['Purchase_Predicted_XGB_avg'].plot(kind='hist')


trainDataValidate.to_csv(path+r'\validationOutputXGB_AVG4.csv',index=False)


