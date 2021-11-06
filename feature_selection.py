# -*- coding: utf-8 -*-
import argparse
import numpy as np
import pickle
from statistics import variance
from math import log
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from scipy.stats import pearsonr,spearmanr
from sklearn.feature_selection import RFE, RFECV
import seaborn as sns
import os

def extract_bat_cycle_life(batch,index):
    """
    Extract the battery cycle life(RUL) and log RUL value
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :return: list, RUL and log(RUL)
    """
    Y_log=[]
    Y= []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        Y_log.append(log(np.ravel(batch[cell_no]['cycle_life']),10))
        Y.append(batch[cell_no]['cycle_life'])
    Y = np.ravel(Y)
    return Y, Y_log
    pass

# Features as Supplementary
def extract_delta_Q_min_mean(batch,index,start_cycle,end_cycle):
    """
    Extract delta Q(V) log minimum and mean value over dataset
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :param start_cycle: start cycle index e.g  start_cycle = 2, cycle2
    :param end_cycle: end cycle index e.g end_cycle = 100, cycle 100
    :return: 2 feature, log(|min(delta_Q)|) and log(|mean(delta_Q)|)
    """
    dQ_min= []
    dQ_mean = []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        Qd_100 = batch[cell_no]['cycles'][str(end_cycle-1)]['Qdlin']
        Qd_10 = batch[cell_no]['cycles'][str(start_cycle-1)]['Qdlin']
        delta = Qd_100-Qd_10
        min_log = log(abs(min(delta)),10) # log base 10
        mean_log = log(abs(np.average(delta)), 10)  # log base 10
        dQ_min.append(min_log)
        dQ_mean.append(mean_log)
    dQ_min = np.reshape(dQ_min,(-1,1))
    dQ_mean = np.reshape(dQ_mean,(-1,1))
    return dQ_min,dQ_mean
    pass


def extract_delta_Q_variance(batch,index,start_cycle,end_cycle):
    """
    Extract delta Q(V) variance over dataset
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :param start_cycle: start cycle index e.g  start_cycle = 2, cycle2
    :param end_cycle: end cycle index e.g end_cycle = 100, cycle 100
    :return: 1 feature, log(|variance delta(Q)|)
    """
    X= []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        Qd_100 = batch[cell_no]['cycles'][str(end_cycle-1)]['Qdlin']
        Qd_10 = batch[cell_no]['cycles'][str(start_cycle-1)]['Qdlin']
        #Calculte the log of variance of (Qd100 - Qd10)
        var_log = log(abs(variance(Qd_100-Qd_10)),10) # log base 10
        X.append(var_log)
    X = np.reshape(X,(-1,1))
    return X
    pass


def extract_delta_Q_skewness(batch,index,start_cycle,end_cycle):
    """
    Extract delta Q(V) log skewness value over dataset
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :param start_cycle: start cycle index e.g  start_cycle = 2, cycle2
    :param end_cycle: end cycle index e.g end_cycle = 100, cycle 100
    :return: 1 feature, log(abs(skewness))
    """
    from scipy.stats import skew
    X= []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        Qd_100 = batch[cell_no]['cycles'][str(end_cycle-1)]['Qdlin']
        Qd_10 = batch[cell_no]['cycles'][str(start_cycle-1)]['Qdlin']
        delta = Qd_100-Qd_10
        # delta_rv_mean = delta - np.average(delta)
        # temp = np.average(np.power(delta_rv_mean,3)) / np.power(np.sum(np.power(delta_rv_mean,2)),1.5)
        # Note: Supplementary formular is wrong
        temp = skew(delta)
        skewness = log(abs(temp),10)
        X.append(skewness)
    X = np.reshape(X,(-1,1))
    return X
    pass


def extract_delta_Q_kurtosis(batch,index,start_cycle,end_cycle):
    """
    Extract delta Q(V) log kurtosis value over dataset
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :param start_cycle: start cycle index e.g  start_cycle = 2, cycle2
    :param end_cycle: end cycle index e.g end_cycle = 100, cycle 100
    :return:1 feature, log(abs(kurtosis))
    """
    from scipy.stats import kurtosis
    X= []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        Qd_100 = batch[cell_no]['cycles'][str(end_cycle-1)]['Qdlin']
        Qd_10 = batch[cell_no]['cycles'][str(start_cycle-1)]['Qdlin']
        delta = Qd_100-Qd_10
        # delta_rv_mean = delta - np.average(delta)
        # temp = np.average(np.power(delta_rv_mean,4)) / np.power(np.average(np.power(delta_rv_mean,2)),2)
        temp = kurtosis(delta,fisher=False)
        kurt = log(abs(temp),10)
        X.append(kurt)
    X = np.reshape(X,(-1,1))
    return X
    pass


def extract_delta_Q_2V(batch,index,start_cycle,end_cycle):
    """
    Extract delta Q(V) value at 2V
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :param start_cycle: start cycle index e.g  start_cycle = 2, cycle2
    :param end_cycle: end cycle index e.g end_cycle = 100, cycle 100
    :return:1 feature, delta Q(V=2)
    """
    X= []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        Qd_100 = batch[cell_no]['cycles'][str(end_cycle-1)]['Qdlin']
        Qd_10 = batch[cell_no]['cycles'][str(start_cycle-1)]['Qdlin']
        delta = Qd_100-Qd_10
        delta_q_2v = log(abs(delta[-1]),10)
        X.append(delta_q_2v)
    X = np.reshape(X,(-1,1))
    return X
    pass


def extract_cycle_QDischarge(batch,index,cycle):
    """
    Extract the Discharge capacity at cycle 2 over dataset
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :return: 1 feature, cycle 2 discharge capacity list
    """
    X= []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        qd_ = batch[cell_no]['summary']['QD'][cycle-1]
        qd_ = log(abs(qd_),10)
        X.append(qd_)
    X = np.reshape(X,(-1,1))
    return X
    pass


def extract_max_minus_cycle_2_QDischarge(batch,index):
    """
    Extract the Discharge capacity difference of cycle 2 and max discharge capacity over dataset
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :return:1 feature,  cycle 2 discharge capacity list
    """
    X= []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        max_minus_2 = max(batch[cell_no]['summary']['QD'][0:100]) - batch[cell_no]['summary']['QD'][1]
        max_minus_2 = log(abs(max_minus_2),10)
        X.append(max_minus_2)
    X = np.reshape(X,(-1,1))
    return X
    pass


def extract_slope_intercept_cycle_to_cycle(batch,index,start_cycle,end_cycle):
    """
    Extract the slope and intercept value from cycle 2 to cycle 100
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :param start_cycle: eg. cycle 100
    :param end_cycle: eg cycle 2
    :return:2 features, slope and intercept
    """
    slope= []
    intercept = []
    nums = end_cycle - start_cycle + 1
    for ind in index:
        cell_no = list(batch.keys())[ind]

        x_axis = np.arange(1,nums+1,1)
        y_axis = batch[cell_no]['summary']['QD'][start_cycle-1:end_cycle]

        n = np.max(x_axis.shape)
        X = np.vstack([np.ones(n), x_axis]).T
        y = y_axis.reshape((n, 1))
        slope_,intercept_ = np.linalg.lstsq(X, y,rcond=-1)[0]
        # slope.append(slope_)
        # intercept.append(intercept_)
        slope.append(log(abs(slope_),10))
        intercept.append(log(abs(intercept_),10))
    slope = np.reshape(slope,(-1,1))
    intercept = np.reshape(intercept,(-1,1))
    return slope,intercept
    pass


def extract_avg_charge_time_5(batch,index):
    """
    Extract the average charge time of first 5 cycles
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :return:1 feature, average charge time of first 5 cycles
    """
    avg_time = []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        avg_time_ = np.average(batch[cell_no]['summary']['chargetime'][1:6]) #Cycle 2 to cycle 6
        # avg_time.append(avg_time_)
        avg_time.append(log(abs(avg_time_),10))
    avg_time = np.reshape(avg_time,(-1,1))
    return avg_time
    pass


def extract_temp_2_to_100_max_min(batch,index):
    """
    Extract the max and min temperature over cycle 2 to 100
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :return:2 feature, temperature max and min value
    """
    max_temp = []
    min_temp = []

    for ind in index:
        cell_no = list(batch.keys())[ind]
        temp= batch[cell_no]['summary']['Tavg'][1:100]
        # integral.append(integrate_)
        max_temp.append(log(abs(max(temp)),10))
        min_temp.append(log(abs(min(temp)), 10))
    max_temp = np.reshape(max_temp,(-1,1))
    min_temp = np.reshape(min_temp, (-1, 1))
    return max_temp,min_temp
    pass


def extract_temp_integral_2_to_100(batch,index):
    """
    Extract the integral of temperature over time, cycle 2 to 100
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :return:1 feature, temperature integral value
    """
    from scipy import integrate
    integral = []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        integrate_ = integrate.simps(batch[cell_no]['summary']['Tavg'][1:100])
        # integral.append(integrate_)
        integral.append(log(abs(integrate_),10))
    integral = np.reshape(integral,(-1,1))
    return integral
    pass


def extract_min_ir_2_to_100(batch,index):
    """
    Extract the min internal resistance over time, cycle 2 to 100
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :return:1 feature minimum internal resistance between cycle 2 to 100
    """
    min_ir = []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        ir = batch[cell_no]['summary']['IR'][1:100]
        min_ir_ = min(ir[ir>0]) # Remove 0 value for log
        # min_ir_ = min(batch[cell_no]['summary']['IR'][1:100])
        # min_ir.append(min_ir_)
        min_ir.append(log(abs(min_ir_),10))
    min_ir = np.reshape(min_ir,(-1,1))
    return min_ir
    pass


def extract_diff_ir_2_100(batch,index):
    """
    Extract the difference of internal resistance between cycle 2 and 100
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :return:1 feature, IR difference value
    """
    diff_ir = []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        diff_ir_ = batch[cell_no]['summary']['IR'][99] - batch[cell_no]['summary']['IR'][1]
        # diff_ir.append(diff_ir_)
        diff_ir.append(log(abs(diff_ir_),10))
    diff_ir = np.reshape(diff_ir,(-1,1))
    return diff_ir
    pass


def extract_ir_cycle(batch,index,cycle):
    """
    Extract the internal resistance on one specifical cycle
    :param batch: the total batch data, dict type
    :param index: train_ind, test_ind or secondary_test_ind, list type
    :param cycle: the cycle
    :return:1 feature, IR value
    """
    ir = []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        ir_ = batch[cell_no]['summary']['IR'][cycle-1]
        ir.append(log(abs(ir_),10))
    ir = np.reshape(ir,(-1,1))
    return ir
    pass


def extract_fea_for_total(batch,index):
    """
    Extract all features
    :param batch:
    :param index:
    :return:
    """
    var = extract_delta_Q_variance(batch,index,start_cycle=10,end_cycle=100)
    min, mean = extract_delta_Q_min_mean(batch,index,start_cycle=10,end_cycle=100)
    skew = extract_delta_Q_skewness(batch,index,start_cycle=10,end_cycle=100)
    kurt = extract_delta_Q_kurtosis(batch,index,start_cycle=10,end_cycle=100)
    dQ_2V = extract_delta_Q_2V(batch,index,start_cycle=10,end_cycle=100)
    slope_2,intercept_2 = extract_slope_intercept_cycle_to_cycle(batch,index,2,100)
    slope_91,intercept_91 = extract_slope_intercept_cycle_to_cycle(batch,index,91,100)
    qd_2 = extract_cycle_QDischarge(batch,index,cycle=2)
    qd_100 = extract_cycle_QDischarge(batch,index,cycle=100)
    max_minus_2 = extract_max_minus_cycle_2_QDischarge(batch,index)
    avg_time = extract_avg_charge_time_5(batch,index)
    max_temp, min_temp = extract_temp_2_to_100_max_min(batch,index)
    integtal_t = extract_temp_integral_2_to_100(batch,index)
    ir_2 = extract_ir_cycle(batch,index,cycle=2)
    min_ir = extract_min_ir_2_to_100(batch,index)
    diff_ir = extract_diff_ir_2_100(batch,index)

    X = np.hstack((var,min,mean,skew,kurt,dQ_2V,slope_2,intercept_2,slope_91,intercept_91,
                   qd_2,max_minus_2,qd_100,avg_time,max_temp,min_temp,integtal_t,ir_2,min_ir,diff_ir))
    return X
    pass


def extract_fea_for_self_model(batch,index):
    """
    Extract features for self model, total 12
    :param batch:
    :param index:
    :return:
    """
    var = extract_delta_Q_variance(batch,index,start_cycle=10,end_cycle=100)
    min, mean = extract_delta_Q_min_mean(batch,index,start_cycle=10,end_cycle=100)
    dq_2v = extract_delta_Q_2V(batch,index,start_cycle=10,end_cycle=100)
    slope_91_100, intercept_91_100 = extract_slope_intercept_cycle_to_cycle(batch,index,91,100)
    qd_100 = extract_cycle_QDischarge(batch,index,cycle=100)
    avg_time_first_5 = extract_avg_charge_time_5(batch, index)
    ir_2 = extract_ir_cycle(batch, index, cycle=2)
    diff_ir_2_100 = extract_diff_ir_2_100(batch,index)

    X = np.hstack((var, min, mean, dq_2v, intercept_91_100,
                   qd_100, avg_time_first_5, ir_2, diff_ir_2_100))  # 9 features based on spearman correlation threshold

    return X
    pass


def extract_fea_for_full_model(batch,index):
    """
    Extract features for full model
    :param batch:
    :param index:
    :return:
    """
    var = extract_delta_Q_variance(batch,index,start_cycle=10,end_cycle=100)
    min,_ = extract_delta_Q_min_mean(batch,index,start_cycle=10,end_cycle=100)
    slope_2,intercept_2 = extract_slope_intercept_cycle_to_cycle(batch,index,2,100)
    qd_2 = extract_cycle_QDischarge(batch,index,cycle=2)
    avg_time = extract_avg_charge_time_5(batch,index)
    integtal_t = extract_temp_integral_2_to_100(batch,index)
    min_ir = extract_min_ir_2_to_100(batch,index)
    diff_ir = extract_diff_ir_2_100(batch,index)

    X = np.hstack((var,min,slope_2,intercept_2,qd_2,avg_time,integtal_t,min_ir,diff_ir))
    return X
    pass


def extract_fea_for_discharge_model(batch,index):
    """
    Extract features for discharge model
    :param batch:
    :param index:
    :return:
    """
    var = extract_delta_Q_variance(batch,index,start_cycle=10,end_cycle=100)
    min,_ = extract_delta_Q_min_mean(batch,index,start_cycle=10,end_cycle=100)
    skew = extract_delta_Q_skewness(batch,index,start_cycle=10,end_cycle=100)
    kurt = extract_delta_Q_kurtosis(batch,index,start_cycle=10,end_cycle=100)
    qd_2 = extract_cycle_QDischarge(batch,index,cycle=2)
    max_minus_2 = extract_max_minus_cycle_2_QDischarge(batch,index)

    X = np.hstack((min,var,skew,kurt,qd_2,max_minus_2))
    return X
    pass


def extract_fea_for_hybrid_model(batch,index):
    """
    Extract features for self model, total 12
    :param batch:
    :param index:
    :return:
    """
    var = extract_delta_Q_variance(batch,index,start_cycle=10,end_cycle=100)
    min, mean = extract_delta_Q_min_mean(batch,index,start_cycle=10,end_cycle=100)
    slope_91_100, intercept_91_100 = extract_slope_intercept_cycle_to_cycle(batch,index,91,100)
    qd_2 = extract_cycle_QDischarge(batch,index,cycle=2)
    qd_100 = extract_cycle_QDischarge(batch,index,cycle=100)
    min_ir_2_100 = extract_min_ir_2_to_100(batch,index)
    diff_ir_2_100 = extract_diff_ir_2_100(batch,index)

    X = np.hstack((var, min, mean, slope_91_100, qd_2, qd_100, min_ir_2_100,diff_ir_2_100))  # 8 features after RFE

    return X
    pass


def cal_Fea_RUL_correlation(features,RUL):
    cols = np.shape(features)[1]
    cyc_fea_corrs = []
    for col in range(cols):
        x = features[:, col]
        cyc_fea_corrs.append(spearmanr(x, RUL)[0])
    cyc_fea_corrs = np.asarray(cyc_fea_corrs)
    return cyc_fea_corrs


def main():

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
    numBat = numBat1 + numBat2 + numBat3  # 124
    bat_dict = {**batch1, **batch2, **batch3}

    train_ind = np.arange(1, (numBat1 + numBat2 - 1), 2)

    trainX = extract_fea_for_total(bat_dict, train_ind)
    trainY, trainY_log = extract_bat_cycle_life(bat_dict, train_ind)
    cyc_fea_corrs_train = cal_Fea_RUL_correlation(trainX, trainY_log)
    print("Spearman Correlation:", cyc_fea_corrs_train)

    trainX = extract_fea_for_self_model(bat_dict,train_ind)
    trainY,trainY_log = extract_bat_cycle_life(bat_dict,train_ind)

    print('TrainX shape = ', np.shape(trainX))
    print('TrainY shape = ', np.shape(trainY))
    print('TrainY_log shape = ', np.shape(trainY_log))

    ticklabels_sub = ['dQ_Var','dQ_Min','dQ_Mean','dQ_2V','Qd_Intercept_91_100','Qd_100',
                  'Avg_time_first_5','IR_2','IR_Diff_2_100',]

    estimator = XGBRegressor(booster="gbtree")
    selector = RFE(estimator, n_features_to_select=8, step=1)
    selector = selector.fit(trainX,trainY_log)

    for i in range(selector.support_.size):
        if selector.support_[i]:
            print(ticklabels_sub[i])

    print(selector.support_)

if __name__ == "__main__":
    main()

