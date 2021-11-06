# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import pickle
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from math import log
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from feature_selection import extract_delta_Q_variance, extract_fea_for_discharge_model, \
    extract_fea_for_full_model,extract_bat_cycle_life

parser = argparse.ArgumentParser()
parser.add_argument('-f','--feature',
                    help='Choose feature for Model: 0-Variance, 1-Discharge, 2-Full',
                    type=int,
                    choices=[0,1,2],
                    default=0)

parser.add_argument('-m','--model',
                    help='Choose model: 0-elastic, 1-SVR, 2-RFR, 3-AdaBoost, 4-XGboost',
                    type=int,
                    choices=[0,1,2,3,4],
                    default=0)


args = parser.parse_args()
feature_choice = args.feature
model_choice = args.model


# Load Data
batch1_file = './Data/batch1_corrected.pkl'
batch2_file = './Data/batch2_corrected.pkl'
batch3_file = './Data/batch3_corrected.pkl'
if os.path.exists(batch1_file) and os.path.exists(batch2_file) and os.path.exists(batch3_file):
    batch1 = pickle.load(open(batch1_file, 'rb'))
    batch2 = pickle.load(open(batch2_file, 'rb'))
    batch3 = pickle.load(open(batch3_file, 'rb'))
else:
    print("Can't find the batch data in Directory './Data' ")
    exit()

numBat1 = len(batch1.keys())
numBat2 = len(batch2.keys())
numBat3 = len(batch3.keys())
numBat = numBat1 + numBat2 + numBat3  #124
bat_dict = {**batch1,**batch2,**batch3}

test_ind= np.hstack((np.arange(0,(numBat1+numBat2),2),83))
test_ind = np.delete(test_ind,[21])
train_ind = np.arange(1,(numBat1+numBat2-1),2)
secondary_test_ind = np.arange(numBat-numBat3,numBat)


trainY,trainY_log = extract_bat_cycle_life(bat_dict,train_ind)
testY_Primary,testY_Primary_log = extract_bat_cycle_life(bat_dict,test_ind)
testY_Secondary,testY_Secondary_log = extract_bat_cycle_life(bat_dict,secondary_test_ind)

if feature_choice == 0:
    print("Feature Preparing for Variance Model")
    trainX = extract_delta_Q_variance(bat_dict,train_ind,start_cycle=10,end_cycle=100)
    testX_Primary = extract_delta_Q_variance(bat_dict,test_ind,start_cycle=10,end_cycle=100)
    testX_Secondary = extract_delta_Q_variance(bat_dict,secondary_test_ind,start_cycle=10,end_cycle=100)

elif feature_choice == 1:
    print("Feature Preparing for Discharge Model")
    trainX = extract_fea_for_discharge_model(bat_dict, train_ind)
    testX_Primary = extract_fea_for_discharge_model(bat_dict,test_ind)
    testX_Secondary = extract_fea_for_discharge_model(bat_dict,secondary_test_ind)

elif feature_choice == 2:
    print("Feature Preparing for FULL Model")
    trainX = extract_fea_for_full_model(bat_dict, train_ind)
    testX_Primary = extract_fea_for_full_model(bat_dict, test_ind)
    testX_Secondary = extract_fea_for_full_model(bat_dict, secondary_test_ind)

scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX_Primary = scaler.transform(testX_Primary)
testX_Secondary = scaler.transform(testX_Secondary)

print('TrainX shape = ', np.shape(trainX))
print('TrainY shape = ', np.shape(trainY))
print('TrainY_log shape = ', np.shape(trainY_log))
print('TestX_Primary shape = ', np.shape(testX_Primary))
print('TestY_Primary_log shape = ', np.shape(testY_Primary))
print('TestY_Second_log shape = ', np.shape(testY_Primary_log))
print('TestX_Secondary shape = ', np.shape(testX_Secondary))
print('TestY_Secondary shape = ', np.shape(testY_Secondary))
print('TestY_Secondary_log shape = ', np.shape(testY_Secondary_log))

print("Start Training")
# regr = ElasticNet(alpha=0.001,l1_ratio=0.001) # Variance Model(alpha=0.001,l1_ration=0.001)
# regr.fit(trainX,trainY_log)
#Grid Search for alpha and l1_ration
Grid_search = False

# Build Model
if model_choice == 0:
    print("Elastic Net")
    if Grid_search:
        regr = ElasticNet(normalize=False)
        # alpha = np.arange(start=0.001, stop=0.1, step=0.001)
        # l1_ratio =np.arange(start=0.001, stop=0.1, step=0.001)
        alpha=[0.001]
        l1_ratio=[0.01]
        print('alpha=',alpha)
        print('l1_ratio=',l1_ratio)
        parameters = {'alpha':alpha,'l1_ratio':l1_ratio}
    else:
        if feature_choice==0:
            # Variance Model
            alphas = np.arange(start=0.0001, stop=0.1, step=0.001)
            l1_ratio = np.arange(start=0.001, stop=1.0, step=0.001)
            regr = ElasticNetCV(cv=5,l1_ratio=0.5,alphas=[0.0001])
        elif feature_choice==1:
            # Discharge Model
            alphas = np.arange(start=0.0001, stop=0.01, step=0.0001)
            # alphas =[0.001,0.01,0.1,0.5]
            l1_ratio = np.arange(start=0.001, stop=1.0, step=0.001)
            regr = ElasticNetCV(cv=5, l1_ratio=l1_ratio, alphas=alphas, max_iter=6000, random_state=0)
            # regr = ElasticNetCV(cv=5,l1_ratio=0.002,alphas=[0.001],max_iter=6000,random_state=0)
        elif feature_choice==2:
            # # Full Model
            # alphas= np.arange(start=0.001, stop=0.01, step=0.001)
            # l1_ratio = np.arange(start=0.001, stop=1.0, step=0.001)
            # regr = ElasticNetCV(cv=5, l1_ratio=0.22, alphas=[0.009], max_iter=6000, random_state=0)
            regr = ElasticNetCV(cv=5,l1_ratio=0.083,alphas=[0.006],max_iter=6000,random_state=0)
        elif feature_choice==3:
            # # Self Model
            alphas= np.arange(start=0.001, stop=0.1, step=0.001)
            l1_ratio = np.arange(start=0.001, stop=1.0, step=0.001)
            regr = ElasticNetCV(cv=5, l1_ratio=l1_ratio, alphas=alphas, max_iter=6000, random_state=0)
elif model_choice == 1:
    print("SVR")
    if Grid_search:
        regr = SVR()
        parameters = {'kernel':['rbf'], 'C':[1e0,1e1,1e2,1e3],'gamma':np.logspace(-2,2,5),'epsilon':[0.1,0.01,0.001]}
    else:
        regr=SVR(kernel='rbf',C=10,epsilon=0.001,gamma=0.01)
elif model_choice == 2:
    print("RFR")
    if Grid_search:
        regr = RandomForestRegressor()
        parameters = {'n_estimators':[50,100,200,300],
                      'max_depth':[1,2,3],
                      'max_features':['sqrt','log2',None]}
    else:
        regr = RandomForestRegressor(n_estimators=1000,max_depth=3,max_features='log2',criterion='mae')
elif model_choice == 3:
    print("AdaBoost")
    if Grid_search:
        regr = AdaBoostRegressor()
        parameters ={'n_estimators':[50,100,200,300,400],
                     'learning_rate':[1e-4,1e-3,1e-2,1e-1],
                     'loss':['linear','square','exponential']} # Best learning_rate = 0.1, loss=linear, n_estimator=400
    else:
        from sklearn.tree import DecisionTreeRegressor
        regr=AdaBoostRegressor(DecisionTreeRegressor(max_depth=3,max_features='log2'),learning_rate=0.01,
                               n_estimators=1000,loss='square')
elif model_choice == 4:
    print("XGBoost")
    if Grid_search:
        regr = XGBRegressor()
        # alpha_l1 = np.arange(start=0.01, stop=0.2, step=0.01)
        # lambda_l2 = np.arange(start=0.01, stop=0.2, step=0.01)
        parameters = {'max_depth':[2,3,4],
                      'learning_rate':[1e-3,1e-2,1e-1],
                      'n_estimator':[10,50,100,200],
                      # 'reg_alpha':alpha_l1,
                      # 'reg_lambda':lambda_l2
                      }
    else:
        regr = XGBRegressor(max_depth=3,n_estimators=1000)
# Fit training data
if Grid_search:
    print("Grid Search")
    clf = GridSearchCV(estimator=regr,param_grid=parameters,cv=5,verbose=1,scoring='neg_mean_squared_error')
    clf.fit(X=trainX,y=trainY_log)
    # print(clf.cv_results_)
    print("Best parameters",clf.best_params_)
else:
    regr.fit(trainX, trainY_log)
    # print('alpha=',regr.alpha_,' l1_ratio=',regr.l1_ratio_)
    # print('mse_path',regr.mse_path_)

# Predict
if Grid_search:
    y_pred = clf.predict(trainX)
else:
    y_pred = regr.predict(trainX)
y_pred = np.power(10,y_pred)
rmse =np.sqrt(mean_squared_error(trainY,y_pred))
print("Training RMSE:",rmse)
err = np.mean(np.abs(trainY-y_pred)/trainY)
print("Training Error:",err*100)


if Grid_search:
    y_pred = clf.predict(testX_Primary)
else:
    y_pred = regr.predict(testX_Primary)

y_pred = np.power(10, y_pred)
rmse =np.sqrt(mean_squared_error(testY_Primary,y_pred))
print("Primary Test RMSE:",rmse)
err = np.mean(np.abs(testY_Primary-y_pred)/testY_Primary)
print("Primary Test Error:",err*100)

if Grid_search:
    y_pred = clf.predict(testX_Secondary)
else:
    y_pred = regr.predict(testX_Secondary)

y_pred = np.power(10, y_pred)
rmse =np.sqrt(mean_squared_error(testY_Secondary,y_pred))
print("Secondary Test RMSE:",rmse)
err = np.mean(np.abs(testY_Secondary-y_pred)/testY_Secondary)
print("Secondary Test Error:",err*100)




