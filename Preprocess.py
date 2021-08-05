# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:41:36 2020

@author: LYL
"""

from pandas import read_csv
#from sklearn.preprocessing import Imputer
import numpy as np
import pandas as pd
import math


bank=read_csv('.../bank-full.csv',sep=';')

print(bank.head(5))
print(bank.shape)
print(bank.describe)
bank.isna().sum()

#Boxplot
import seaborn as sns
from matplotlib import pyplot as plt
plt.figure(figsize=(15,7))
sns.boxplot('age','y',data=bank,orient='h', sym=None)
plt.yticks(fontsize=15.0)
plt.show()


#It seems that there is no "null" value in the frame, however, we can see that there are many "unknown" values, so we still need to deal with them.
#Count the "unknown" value
for col in bank.columns:
    if bank[col].dtype == object:
        print("Percentage of \"unknown\" in %s：" %col ,bank[bank[col] == "unknown"][col].count(),"/",bank[col].count())
print(bank.info())
bank.replace('unknown',np.NaN, inplace=True)
bank.isnull().sum()

bank.drop(columns=["poutcome"])
#bank.drop(rows=[]) drop the outliners rows

#Count the 'Yes' and 'No' in variable 'y'
print("Yes:",bank['y'][bank['y']== 'yes'].count())
print("No:",bank['y'][bank['y']== 'no'].count())
#count the frequency of each value in each variable
for col in bank.columns:
    if bank[col].dtype == object:
        print(bank.groupby(bank[col]).apply(lambda x: x['y'][x['y']== 'yes'].count()/x['y'].count()))

#Change the English words of months into numbers


#Group the variables
month = bank['month']
string_features = bank.columns[bank.dtypes == "object"].to_series().values
int_features = bank.columns[bank.dtypes == "int64"].to_series().values
float_features = bank.columns[bank.dtypes == "float64"].to_series().values
numeric_features = np.append(int_features,float_features)

bin_features = ['default', 'housing', 'loan','y']  #binary features
order_features = ['education']
disorder_features = [ 'contact', 'job']

#Functions to deal with the missing value
def Missing_value_perprocessing_mean (bank_data_small_train,bank_data_small_test):
    col  = bank_data_small_train.columns
    #Train_copy = Train.copy()
    #Fill the missing value with average value
    from sklearn.preprocessing import Imputer
    imp = Imputer(missing_values=np.nan, strategy='mean', axis=0)
    imp.fit(bank_data_small_train)
    bank_data_small_train = imp.transform(bank_data_small_train) 
    bank_data_small_test = imp.transform(bank_data_small_test) 
    bank_data_small_train = pd.DataFrame(bank_data_small_train,columns = col)
    bank_data_small_test = pd.DataFrame(bank_data_small_test,columns = col)
    return bank_data_small_train,bank_data_small_test 

def Missing_value_perprocessing_rf (bank_data_small_train,bank_data_small_test):
    Missing_features_dict = {}
    Missing_features_name = []
    #Find the columns with missing value
    for feature in bank_data_small_train.columns:
        Missing_count = bank_data_small_train[bank_data_small_train[feature].isnull()]['age'].count() 
        if Missing_count > 0:
            Missing_features_dict.update({feature: Missing_count})
    #Sort the columns by amount of missing value        
    Missing_features_name = sorted(Missing_features_dict.keys(),reverse=True) 
    #print(Missing_features_name)
    for feature in Missing_features_name:     
        train_miss_data = bank_data_small_train[bank_data_small_train[feature].isnull()]
        train_miss_data_X = train_miss_data.drop(Missing_features_name, axis=1)

        train_full_data = bank_data_small_train[bank_data_small_train[feature].notnull()]     
        train_full_data_Y = train_full_data[feature]
        train_full_data_X = train_full_data.drop(Missing_features_name, axis=1)

        test_miss_data = bank_data_small_test[bank_data_small_test[feature].isnull()]
        test_miss_data_X = test_miss_data.drop(Missing_features_name, axis=1)

        test_full_data = bank_data_small_test[bank_data_small_test[feature].notnull()]     
        test_full_data_Y = test_full_data[feature]
        test_full_data_X = test_full_data.drop(Missing_features_name, axis=1)
        from sklearn.ensemble import RandomForestClassifier
        #Random forest       
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(train_full_data_X, train_full_data_Y)
        train_miss_data_Y = rf.predict(train_miss_data_X)
        test_miss_data_Y = rf.predict(test_miss_data_X) 
        train_miss_data[feature] = train_miss_data_Y  
        test_miss_data[feature] = test_miss_data_Y 
        bank_data_small_train = pd.concat([train_full_data, train_miss_data])
        bank_data_small_test = pd.concat([test_full_data, test_miss_data])      
        
    return bank_data_small_train,bank_data_small_test

def Missing_value_perprocessing_knn (bank_data_small_train,bank_data_small_test):
    Missing_features_dict = {}
    Missing_features_name = []
    for feature in bank_data_small_train.columns:
        Missing_count = bank_data_small_train[bank_data_small_train[feature].isnull()]['age'].count() 
        if Missing_count > 0:
            Missing_features_dict.update({feature: Missing_count})        
    Missing_features_name = sorted(Missing_features_dict.keys(),reverse=True)
    from sklearn.neighbors import KNeighborsClassifier 
    for feature in Missing_features_name:     
        train_miss_data = bank_data_small_train[bank_data_small_train[feature].isnull()]
        train_miss_data_X = train_miss_data.drop(Missing_features_name, axis=1)

        train_full_data = bank_data_small_train[bank_data_small_train[feature].notnull()]     
        train_full_data_Y = train_full_data[feature]
        train_full_data_X = train_full_data.drop(Missing_features_name, axis=1)

        test_miss_data = bank_data_small_test[bank_data_small_test[feature].isnull()]
        test_miss_data_X = test_miss_data.drop(Missing_features_name, axis=1)

        test_full_data = bank_data_small_test[bank_data_small_test[feature].notnull()]     
        test_full_data_Y = test_full_data[feature]
        test_full_data_X = test_full_data.drop(Missing_features_name, axis=1)
        
        #K-NN        
        knn = KNeighborsClassifier()
        forest = knn.fit(train_full_data_X, train_full_data_Y)
        
        train_miss_data_Y = knn.predict(train_miss_data_X)
        test_miss_data_Y = knn.predict(test_miss_data_X) 
        
        train_miss_data[feature] = train_miss_data_Y      
        test_miss_data[feature] = test_miss_data_Y 

        bank_data_small_train = pd.concat([train_full_data, train_miss_data])
        bank_data_small_test = pd.concat([test_full_data, test_miss_data])      
        
    return bank_data_small_train,bank_data_small_test

#Normalization
#def Scale_perprocessing (Train):
#    col  = Train.columns
#    copy = Train.copy()
#    scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
#    copy = scaler.fit_transform(copy)
#    Train = pd.DataFrame(copy,columns = col)
#    return Train

import calendar
list(calendar.month_abbr).index('jan')#input the English words of month

#Divide the age group
bank.loc[(bank.age < 35),  'AgeGroup'] = 'group 1'
bank.loc[(bank.age > 35 and bank.age<50),  'AgeGroup'] = 'group 2'
bank.loc[(bank.age >50 and bank.age <65),  'AgeGroup'] = 'group 3'
bank.loc[(bank.age > 65),  'AgeGroup'] = 'group 4'

#Deal with binary variables
def bin_features_perprocessing (bin_features, bank_data):
    for feature in bin_features:      
        new = np.zeros(bank_data[feature].shape[0])
        for rol in range(bank_data[feature].shape[0]):
            if bank_data[feature][rol] == 'yes' :
                new[rol] = 1
            elif bank_data[feature][rol]  == 'no':
                new[rol] = 0
            else:
                new[rol] = None
        bank_data[feature] =  new   
    return bank_data

#Sort the order variables
def order_features_perprocessing (order_features,bank_data):
    education_values = [ "primary", "secondary","tertiary", 
    "unknown"]
    replace_values = list(range(1,5))    #len(education_values)
    bank_data[order_features] = bank_data[order_features].replace(education_values,replace_values)
    bank_data[order_features] = bank_data[order_features].astype("float")
    return bank_data

#Deal with the rest with onehot
#def disorder_features_perprocessing (disorder_features, bank_data):
#    for features in disorder_features:
#        features_onehot = pd.get_dummies(bank_data[features])
#        features_onehot = features_onehot.rename(columns=lambda x: features+'_'+str(x))
#        bank_data = pd.concat([bank_data,features_onehot],axis=1)
#        bank_data = bank_data.drop(features, axis=1)
#    return bank_data


bank = bin_features_perprocessing(bin_features, bank)
bank = order_features_perprocessing(order_features, bank)
bank = disorder_features_perprocessing(disorder_features, bank)


#Divide the bank data into train and test data
bank.shape[0]
round(bank.shape[0]*0.8)
bank = bank.sample(frac=1,random_state=12)

#import numpy as np
#from sklearn.impute import KNNImputer
#nan = np.nan
#imputer = KNNImputer(n_neighbors=2, weights="uniform") 
#using the mean feature value of the two nearest neighbors of samples with missing values:
#imputer.fit_transform(bank)
#bank_train = Scale_perprocessing (bank_train)
#Missing_value_perprocessing
#Average
#bank_train,bank_test = Missing_value_perprocessing_mean(bank_train,bank_test)
#K-NN
#bank_train,bank_test=Missing_value_perprocessing_knn(bank_train,bank_test)
#Random forest
#bank_train,bank_test=Missing_value_perprocessing_rf(bank_train,bank_test) 

#X_train = bank_train.drop(['y'], axis=1).copy()
#y_train = pd.DataFrame(bank_train['y'],columns = ['y'])

#X_test = bank_test.drop(['y'], axis=1).copy()
#y_test = pd.DataFrame(bank_test['y'],columns = ['y'])

#X_train = Scale_perprocessing(X_train)
#X_test = Scale_perprocessing(X_test)

#Output to documents
#outputpath1=r'D:\Study\Postgraduate\NTU-ANALYTICS\Data Mining\Project\Preprocess\bank_train.csv'
#bank_train.to_csv(outputpath1, sep=',',index=False,header=True)
#outputpath2=r'D:\Study\Postgraduate\NTU-ANALYTICS\Data Mining\Project\Preprocess\bank_test.csv'
#bank_test.to_csv(outputpath2,sep=',',index=False,header=True)
#outputpath3=r'D:\Study\Postgraduate\NTU-ANALYTICS\Data Mining\Project\Preprocess\X_train.csv'
#X_train.to_csv(outputpath3,sep=',',index=False,header=True)
#outputpath4=r'D:\Study\Postgraduate\NTU-ANALYTICS\Data Mining\Project\Preprocess\X_test.csv'
#X_test.to_csv(outputpath4,sep=',',index=False,header=True)
#outputpath5=r'D:\Study\Postgraduate\NTU-ANALYTICS\Data Mining\Project\Preprocess\y_train.csv'
#y_train.to_csv(outputpath5,sep=',',index=False,header=True)
#outputpath6=r'D:\Study\Postgraduate\NTU-ANALYTICS\Data Mining\Project\Preprocess\y_test.csv'
#y_test.to_csv(outputpath6,sep=',',index=False,header=True)

outputpath7=r'D:\Study\Postgraduate\NTU-ANALYTICS\Data Mining\Project\bank_new.csv'
bank.to_csv(outputpath7, sep=',',index=False,header=True)
