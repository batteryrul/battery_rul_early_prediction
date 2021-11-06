import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import pickle
import os
import torch
from sklearn import preprocessing
from sklearn.externals import joblib
import argparse
from statistics import variance
from scipy.stats import skew,kurtosis,pearsonr
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from feature_selection import extract_delta_Q_variance, extract_fea_for_discharge_model, \
    extract_fea_for_full_model,extract_bat_cycle_life,extract_fea_for_hybrid_model
from sklearn.metrics import auc

def cycle_feat_extract(x):
    x_data = np.arange(len(x)) + 1
    Qdlin_min = np.min(x)
    Qdlin_max = np.max(x)
    Qdlin_mean = np.mean(x)
    Qdlin_var = np.var(x)
    Qdlin_median = np.median(x)
    Qdlin_skewness = skew(x)
    Qdlin_auc = auc(x_data, x)
    cycle_feat = list((Qdlin_min, Qdlin_max, Qdlin_mean, Qdlin_var, Qdlin_median, Qdlin_skewness, Qdlin_auc))
    return cycle_feat


def extract_local_fea(batch, index, start_cycle, end_cycle):

    bat_data = []
    for ind in index:
        cell_no = list(batch.keys())[ind]
        cycle_data = []
        # extract data on cycle level
        for cycle in range(start_cycle, end_cycle + 1):
            cycle_temp=[]
            #raw
            cycle_temp.extend(cycle_feat_extract(batch[cell_no]['cycles'][str(cycle)]['Qdlin']))
            cycle_temp.extend(cycle_feat_extract(batch[cell_no]['cycles'][str(cycle)]['Tdlin']))
            cycle_temp.extend(cycle_feat_extract(batch[cell_no]['cycles'][str(cycle)]['dQdV']))

            cycle_data.append(cycle_temp)
        bat_data.append(cycle_data)
    return np.asarray(bat_data)


def main():

    for model_choice in range(0,4):

        if model_choice==0:
            # Variance Model
            model_name = 'var'
        elif model_choice==1:
            # Discharge Model
            model_name ='dis'
        elif model_choice==2:
            # Full Model
            model_name ='full'
        else:
            # Hybird Model
            model_name ='hybird'

        print("Start to create dataset for: ", model_name)
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

        test_ind = np.hstack((np.arange(0, (numBat1 + numBat2), 2), 83))
        test_ind = np.delete(test_ind,[21]) # Remove 1 bad battery as paper
        train_ind = np.arange(1, (numBat1 + numBat2 - 1), 2)
        secondary_test_ind = np.arange(numBat - numBat3, numBat)

        _, train_y = extract_bat_cycle_life(bat_dict, train_ind)
        test_y_pri,_= extract_bat_cycle_life(bat_dict, test_ind)
        test_y_sec,_ = extract_bat_cycle_life(bat_dict, secondary_test_ind)

        train_y = np.expand_dims(np.array(train_y),axis=1)
        test_y_pri = np.expand_dims(np.array(test_y_pri),axis=1)
        test_y_sec = np.expand_dims(np.array(test_y_sec),axis=1)

        train_x = extract_local_fea(bat_dict, train_ind, start_cycle=1, end_cycle=100)
        test_x_pri = extract_local_fea(bat_dict, test_ind, start_cycle=1, end_cycle=100)
        test_x_sec = extract_local_fea(bat_dict, secondary_test_ind, start_cycle=1, end_cycle=100)

        if model_choice == 0:
            print("Feature Preparing for Variance Model")
            train_x_fc = extract_delta_Q_variance(bat_dict, train_ind, start_cycle=10, end_cycle=100)
            test_x_pri_fc = extract_delta_Q_variance(bat_dict, test_ind, start_cycle=10, end_cycle=100)
            test_x_sec_fc = extract_delta_Q_variance(bat_dict, secondary_test_ind, start_cycle=10, end_cycle=100)
        elif model_choice == 1:
            print("Feature Preparing for Discharge Model")
            train_x_fc = extract_fea_for_discharge_model(bat_dict, train_ind)
            test_x_pri_fc = extract_fea_for_discharge_model(bat_dict, test_ind)
            test_x_sec_fc = extract_fea_for_discharge_model(bat_dict, secondary_test_ind)

        elif model_choice == 2:
            print("Feature Preparing for FULL Model")
            train_x_fc = extract_fea_for_full_model(bat_dict, train_ind)
            test_x_pri_fc = extract_fea_for_full_model(bat_dict, test_ind)
            test_x_sec_fc = extract_fea_for_full_model(bat_dict, secondary_test_ind)

        elif model_choice == 3:
            print("Feature Preparing for Hybird Model")
            train_x_fc = extract_fea_for_hybrid_model(bat_dict, train_ind)
            test_x_pri_fc = extract_fea_for_hybrid_model(bat_dict, test_ind)
            test_x_sec_fc = extract_fea_for_hybrid_model(bat_dict, secondary_test_ind)


        print("train_x shape ={}, train_y shape ={}, train_x_fc shape={}".format(train_x.shape,train_y.shape,train_x_fc.shape))
        print("test_x_pri shape ={}, test_y_pri shape ={}, test_x_pri_fc shape={}".format(test_x_pri.shape,test_y_pri.shape,test_x_pri_fc.shape))
        print("test_x_sec shape ={}, test_y_sec shape ={}, test_x_sec_fc shape={}".format(test_x_sec.shape,test_y_sec.shape, test_x_sec_fc.shape))


        # Max_min Normalization
        v_max = train_x.max(axis=(0, 1), keepdims=True)
        v_min = train_x.min(axis=(0, 1), keepdims=True)

        train_x_nor = (train_x - v_min) / (v_max-v_min)
        test_x_pri_nor = (test_x_pri - v_min) / (v_max-v_min)
        test_x_sec_nor = (test_x_sec - v_min) / (v_max-v_min)

        # scaler = StandardScaler()

        scaler_fc = StandardScaler()
        train_x_fc_nor = scaler_fc.fit_transform(train_x_fc)
        test_x_pri_fc_nor = scaler_fc.transform(test_x_pri_fc)
        test_x_sec_fc_nor = scaler_fc.transform(test_x_sec_fc)

        # No Normalization for y
        train_y_nor = train_y

        dataset={}

        dataset['train_x'] = train_x_nor
        dataset['train_x_fc'] = train_x_fc_nor
        dataset['train_y'] = train_y_nor

        dataset['eva_x'] = train_x_nor
        dataset['eva_x_fc'] = train_x_fc_nor
        dataset['eva_y'] = train_y_nor

        dataset['test_x_pri'] = test_x_pri_nor
        dataset['test_x_pri_fc'] = test_x_pri_fc_nor
        dataset['test_y_pri'] = test_y_pri

        dataset['test_x_sec'] = test_x_sec_nor
        dataset['test_x_sec_fc'] = test_x_sec_fc_nor
        dataset['test_y_sec'] = test_y_sec

        data_path = './processed_data/first_100_cycle_data_' + model_name + '.pt'
        torch.save(dataset,data_path)

    print("End")

if __name__ == '__main__':
    main()


