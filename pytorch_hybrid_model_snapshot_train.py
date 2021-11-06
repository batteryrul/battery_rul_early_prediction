import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchsummary import  summary
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.externals import joblib
from utils.sgdr import CosineAnnealingLR_with_Restart

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyDataset(Dataset):
    def __init__(self, x, x_fc, y):
        super(MyDataset, self).__init__()
        self.len = x.shape[0]
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.x_data = torch.as_tensor(x, device=device, dtype=torch.float)
        self.x_fc_data = torch.as_tensor(x_fc, device=device, dtype=torch.float)
        self.y_data = torch.as_tensor(y, device=device, dtype=torch.float)

    def __getitem__(self, index):
        return self.x_data[index], self.x_fc_data[index], self.y_data[index]
    def __len__(self):
        return self.len


class cnn_model_kp(nn.Module):
    def __init__(self, input_dim, input_fc_dim, hidden_dim):
        super(cnn_model_kp, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_fc_dim = input_fc_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=21, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3),
            nn.Flatten(),
            nn.Linear(in_features=32, out_features=self.hidden_dim),
        )
        self.dropout_layer_cnn = nn.Dropout(p=0.5)
        self.dropout_layer_fc = nn.Dropout(p=0.0)

        self.fc = nn.Sequential(
            nn.Linear(in_features=input_fc_dim, out_features=self.hidden_dim),
            nn.Sigmoid(),
            nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
            # nn.Sigmoid(),
        )
        self.regressor=nn.Sequential(
            nn.Linear(in_features=self.hidden_dim * 2, out_features = 1),
        )

    def forward(self, src, src_fc):
        fea_cnn = self.dropout_layer_cnn(self.encoder(src))
        fea_fc = self.dropout_layer_fc(self.fc(src_fc))
        fea = torch.cat((fea_cnn,fea_fc),1)
        x = self.regressor(fea)
        return x

epochs = 4000
model_name ='hybrid'
iterations = 5
t_max = 200
t_mult = 1
lr_min = 1e-6
snapshot_em_start = 10
hidden_dim = 16

model_saved_dir = './result'
data_path = './processed_data/first_100_cycle_data_hybrid.pt'

data_set = torch.load(data_path)
train_x = torch.from_numpy(data_set['train_x']).permute(0,2,1).float()
train_y = torch.from_numpy(data_set['train_y']).float()
train_x_fc = torch.from_numpy(data_set['train_x_fc']).float()

train_x_eva = torch.from_numpy(data_set['eva_x']).permute(0,2,1).float()
train_y_eva = torch.from_numpy(data_set['eva_y']).float()
train_x_eva_fc = torch.from_numpy(data_set['eva_x_fc']).float()

test_x_pri = torch.from_numpy(data_set['test_x_pri']).permute(0,2,1).float()
test_y_pri = torch.from_numpy((data_set['test_y_pri'])).float()
test_x_pri_fc = torch.from_numpy(data_set['test_x_pri_fc']).float()

test_x_sec = torch.from_numpy(data_set['test_x_sec']).permute(0,2,1).float()
test_y_sec = torch.from_numpy(data_set['test_y_sec']).float()
test_x_sec_fc = torch.from_numpy(data_set['test_x_sec_fc']).float()

print("train_x shape ={}, train_y shape ={}, train_x_fc shape={}".format(train_x.shape,train_y.shape,train_x_fc.shape))
print("train_x_eva shape ={}, train_y_eva shape ={}, train_x_eva_fc shape={}".format(train_x_eva.shape,train_y_eva.shape,train_x_eva_fc.shape))
print("test_x_pri shape ={}, test_y_pri shape ={}, test_x_pri_fc shape={}".format(test_x_pri.shape,test_y_pri.shape,test_x_pri_fc.shape))
print("test_x_sec shape ={}, test_y_sec shape ={}, test_x_sec_fc shape={}".format(test_x_sec.shape,test_y_sec.shape, test_x_sec_fc.shape))


train_rmse_snapshot,train_err_snapshot = [],[]
test_pri_rmse_snapshot, test_pri_err_snapshot = [], []
test_sec_rmse_snapshot, test_sec_err_snapshot= [], []


for iteration in range(iterations):

    train_enable = True

    if train_enable:

        model = cnn_model_kp(input_dim = train_x.shape[2],input_fc_dim=train_x_fc.shape[1],hidden_dim=hidden_dim).to(device)

        optimizer = optim.SGD(model.parameters(),lr=0.1, weight_decay=0.001, momentum=0.9)
        criterion = nn.MSELoss()
        mySGDR = CosineAnnealingLR_with_Restart(optimizer=optimizer,T_max=t_max, T_mult=t_mult, model=model,out_dir=model_saved_dir,
                                              take_snapshot=True, eta_min=lr_min)

        for epoch in range(1, epochs+1):
            mySGDR.step()
            model.train()
            Loss = 0.0
            train_x_fc_ = train_x_fc.to(device)
            train_x_ = train_x.to(device)
            train_y_ = train_y.to(device)
            optimizer.zero_grad()
            pred = model(train_x_,train_x_fc_)
            loss = criterion(pred, train_y_)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            Loss += loss.item()

            if epoch % 100 ==0:
                model.eval()
                with torch.no_grad():
                    train_x_eva_ = train_x_eva.to(device)
                    train_x_eva_fc_ = train_x_eva_fc.to(device)
                    pred = model(train_x_eva_, train_x_eva_fc_)
                    pred_ = pred.to('cpu').numpy()
                    pred_ = np.power(10, pred_)
                    train_y_eva_ = np.power(10, train_y_eva)
                    rmse = sqrt(mean_squared_error(train_y_eva_, pred_))
                print("Epoch[{}/{}] | Train Loss = {:.5f}, Train RMSE={:.2f}".format(epoch, epochs, Loss, rmse))

    import glob

    checkpoints = sorted(glob.glob(model_saved_dir + '/*.tar'))

    models = []

    for path in checkpoints:
        model =cnn_model_kp(input_dim = train_x.shape[2],input_fc_dim=train_x_fc.shape[1],hidden_dim=hidden_dim).to(device)
        ch = torch.load(path)
        model.load_state_dict(ch['state_dict'])
        models.append(model)

    i=0
    train_rmse, train_err = [], []
    test_pri_rmse, test_pri_err = [], []
    test_sec_rmse, test_sec_err = [], []

    # Ensemble models:
    for model in models[snapshot_em_start:]:
        i+=1
        model.eval()
        with torch.no_grad():
            train_x_eva_ = train_x_eva.to(device)
            train_x_eva_fc_ = train_x_eva_fc.to(device)
            y_pred = model(train_x_eva_,train_x_eva_fc_)
            y_pred = y_pred.to('cpu').numpy()
            # y_pred = scaler.inverse_transform(y_pred)
            y_pred = np.power(10, y_pred)
            train_y_eva_ = np.power(10, train_y_eva)
            rmse = sqrt(mean_squared_error(train_y_eva_,y_pred))
            err = np.average(np.divide(np.abs(train_y_eva_-y_pred),train_y_eva_)) *100
            print("Snapshot-{} Prediction | Training RMSE = {:.2f}, Error ={:.2f}".format(i, rmse,err))
            train_rmse.append(rmse)
            train_err.append(err)

            test_x_pri_= test_x_pri.to(device)
            test_x_pri_fc_ = test_x_pri_fc.to(device)
            y_pred = model(test_x_pri_,test_x_pri_fc_)
            y_pred = y_pred.to('cpu').numpy()
            # y_pred = scaler.inverse_transform(y_pred)
            y_pred = np.power(10, y_pred)
            rmse = sqrt(mean_squared_error(test_y_pri,y_pred))
            err = np.average(np.divide(np.abs(test_y_pri-y_pred),test_y_pri)) *100
            print("Snapshot-{} Prediction | Primary Test RMSE = {:.2f}, Error ={:.2f}".format(i, rmse,err))
            test_pri_rmse.append(rmse)
            test_pri_err.append(err)

            test_x_sec_ = test_x_sec.to(device)
            test_x_sec_fc_ = test_x_sec_fc.to(device)
            y_pred = model(test_x_sec_,test_x_sec_fc_)
            y_pred = y_pred.to('cpu').numpy()
            # y_pred = scaler.inverse_transform(y_pred)
            y_pred = np.power(10, y_pred)
            rmse = sqrt(mean_squared_error(test_y_sec,y_pred))
            err = np.average(np.divide(np.abs(test_y_sec-y_pred),test_y_sec)) *100
            print("Snapshot-{} Prediction | Secondary RMSE = {:.2f}, Error ={:.2f}".format(i, rmse,err))
            test_sec_rmse.append(rmse)
            test_sec_err.append(err)
            print()
    train_rmse_snapshot.append(sum(train_rmse)/len(train_rmse))
    train_err_snapshot.append(sum(train_err)/len(train_err))
    test_pri_rmse_snapshot.append(sum(test_pri_rmse)/len(test_pri_rmse))
    test_pri_err_snapshot.append(sum(test_pri_err)/len(test_pri_err))
    test_sec_rmse_snapshot.append(sum(test_sec_rmse)/len(test_sec_rmse))
    test_sec_err_snapshot.append(sum(test_sec_err)/len(test_sec_err))

print()
print("Snapshot | train_rmse_snapshot:", train_rmse_snapshot)
print("Snapshot | train_err_snapshot:", train_err_snapshot)
print("Snapshot | test_pri_rmse_snapshot:", test_pri_rmse_snapshot)
print("Snapshot | test_pri_err_snapshot:", test_pri_err_snapshot)
print("Snapshot | test_sec_rmse_snapshot:", test_sec_rmse_snapshot)
print("Snapshot | test_sec_err_snapshot:", test_sec_err_snapshot)
print()
print("Snapshot | Average Training RMSE={:.2f}({:.2f}), Error={:.2f}({:.2f})".format(
    sum(train_rmse_snapshot)/len(train_rmse_snapshot),
    np.std(train_rmse_snapshot),
    sum(train_err_snapshot)/len(train_err_snapshot),
    np.std(train_err_snapshot)))
print("Snapshot | Average Primary Test RMSE={:.2f}({:.2f}), Error={:.2f}({:.2f})".format(
    sum(test_pri_rmse_snapshot)/len(test_pri_rmse_snapshot),
    np.std(test_pri_rmse_snapshot),
    sum(test_pri_err_snapshot)/len(test_pri_err_snapshot),
    np.std(test_pri_err_snapshot)))
print("Snapshot | Average Secondary Test RMSE={:.2f}({:.2f}), Error={:.2f}({:.2f})".format(
    sum(test_sec_rmse_snapshot)/len(test_sec_rmse_snapshot),
    np.std(test_sec_rmse_snapshot),
    sum(test_sec_err_snapshot)/len(test_sec_err_snapshot),
    np.std(test_sec_err_snapshot)))

print("End")