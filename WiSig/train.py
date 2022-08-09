import numpy as np
import pickle
import matplotlib.pyplot as plt

from data_utilities import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
#from torchsummary import summary

from models import Wisig_net_classifier, MobileNetV2_Classifier, ResNet_50_first_layer, MobileNetV2_encoder, Wisig_net_contrastive
from pytorchtools import EarlyStopping
from SupConLoss import SupConLoss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print("The model will be running on", device, "device\n")

class data_preparation(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return data, label

def data_process(dataset_name, dataset_path, device_index, test_dataset_path):

    compact_dataset = load_compact_pkl_dataset(dataset_path,dataset_name)

    tx_list = compact_dataset['tx_list']
    rx_list = compact_dataset['rx_list']
    capture_date_list = compact_dataset['capture_date_list']

    ntx_list = [10, 45, 80, 115,150]
    tx_train_list= tx_list[0:device_index]

    equalized = 0
    dataset = merge_compact_dataset(compact_dataset,capture_date_list,tx_train_list,rx_list, equalized=equalized)
    train_augset,val_augset,test_augset_smRx =  prepare_dataset(dataset,tx_train_list, val_frac=0.1, test_frac=0.1)
    [sig_train,txidNum_train,txid_train,cls_weights] = train_augset 
    [sig_valid,txidNum_valid,txid_valid,_] = val_augset
    [sig_smTest,txidNum_smTest,txid_smTest,cls_weights] = test_augset_smRx

    # reshape the input data as channels_first data format
    sig_train = np.reshape(sig_train, (sig_train.shape[0], 1, sig_train.shape[1], sig_train.shape[2]))
    # covert the one-hot vector to class values
    txid_train = np.argmax(txid_train, axis=1) 
    sig_valid = np.reshape(sig_valid, (sig_valid.shape[0], 1, sig_valid.shape[1], sig_valid.shape[2]))
    txid_valid = np.argmax(txid_valid, axis=1) 
    sig_smTest = np.reshape(sig_smTest, (sig_smTest.shape[0], 1, sig_smTest.shape[1], sig_smTest.shape[2]))
    txid_smTest = np.argmax(txid_smTest, axis=1) 

    train_dataset = data_preparation(sig_train, txid_train)
    valid_dataset = data_preparation(sig_valid, txid_valid)
    test_dataset = data_preparation(sig_smTest, txid_smTest)

    with open(test_dataset_path, 'wb') as f:
        pickle.dump(test_dataset, f)

    return train_dataset, valid_dataset, test_dataset


def train_classifier(model_name, dataset_name, dataset_path, device_index, saved_model_path, test_dataset_path):

    train_dataset, valid_dataset, test_dataset = data_process(dataset_name, dataset_path, device_index, test_dataset_path)
    
    if model_name == 'wisig_net_classifier':
        model = Wisig_net_classifier(num_classes=device_index)
        scheduler_patience = 5
        early_stopping_patience = 10
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        loss_fn = nn.CrossEntropyLoss()
    elif model_name == 'mobilenetv2_classifier':
        model = MobileNetV2_Classifier(num_classes=device_index, width_mult=0.5)
        scheduler_patience = 10 
        early_stopping_patience = 20
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
    elif model_name == 'resnet_50_classifier':
        first_layer = ResNet_50_first_layer()
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_ftrs, device_index)
        model = nn.Sequential(first_layer, resnet50)
        scheduler_patience = 10
        early_stopping_patience = 20
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()

    model.to(device)
    #summary(model, input_size=(1, 256, 2))
    loss_fn.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=scheduler_patience, verbose=True)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=False, path=saved_model_path)

    batch_size = 512
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    valid_losses = []

    for epoch in range(500):
        model.to(device)
        model.train(True)
        for data, label in train_data_loader:
            data = data.type(torch.cuda.FloatTensor) 
            label = label.type(torch.cuda.LongTensor)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, label) 
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        for vdata, vlabels in valid_data_loader:
            vdata = vdata.type(torch.FloatTensor) 
            vlabels = vlabels.type(torch.LongTensor)
            voutputs = model(vdata)
            vloss = loss_fn(voutputs, vlabels)
            valid_losses.append(vloss.item())
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        print('Epoch:', epoch+1, 'train-loss: {} validation-loss: {}'.format(train_loss, valid_loss))

        train_losses = []
        valid_losses = []

        scheduler.step(valid_loss)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def train_contrastive_extractor(model_name, dataset_name, dataset_path, device_index, saved_model_path, test_dataset_path):

    train_dataset, valid_dataset, test_dataset = data_process(dataset_name, dataset_path, device_index, test_dataset_path)

    if model_name == 'mobilenetv2_contrastive':
        model = MobileNetV2_encoder(width_mult=0.5)
        scheduler_patience = 10
        early_stopping_patience = 20
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = SupConLoss(temperature=0.07)
    elif model_name == 'wisig_net_contrastive':
        model = Wisig_net_contrastive()
        scheduler_patience = 5
        early_stopping_patience = 20
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = SupConLoss(temperature=0.07)
    elif model_name == 'resnet_50_contrastive':
        first_layer = ResNet_50_first_layer()
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_ftrs, 128)
        model = nn.Sequential(first_layer, resnet50)
        scheduler_patience = 5
        early_stopping_patience = 20
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = SupConLoss(temperature=0.07)

    model.to(device)
    #summary(model, input_size=(1, 256, 2))
    loss_fn.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=scheduler_patience, verbose=True)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=False, path=saved_model_path)

    batch_size = 512
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    valid_losses = []

    for epoch in range(500):
        model.to(device)
        model.train(True)
        for data, label in train_data_loader:
            data = data.type(torch.cuda.FloatTensor) 
            label = label.type(torch.cuda.LongTensor)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, label) 
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        for vdata, vlabels in valid_data_loader:
            vdata = vdata.type(torch.FloatTensor) 
            vlabels = vlabels.type(torch.LongTensor)
            voutputs = model(vdata)
            vloss = loss_fn(voutputs, vlabels)
            valid_losses.append(vloss.item())
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        print('Epoch:', epoch+1, 'train-loss: {} validation-loss: {}'.format(train_loss, valid_loss))

        train_losses = []
        valid_losses = []

        scheduler.step(valid_loss)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


def MLDG(network, optimizer, mldg_beta, meta_train_data, meta_train_label, all_meta_data, all_mata_label):
    
    optimizer.zero_grad()

    for p in network.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    inner_net = type(network)()
    inner_net.load_state_dict(network.state_dict())
    inner_net.to(device)

    inner_opt = torch.optim.SGD(inner_net.parameters(),
        lr=optimizer.param_groups[0]['lr'],
        momentum=optimizer.param_groups[0]['momentum']
    )
    inner_opt.zero_grad()

    loss_fn = SupConLoss(temperature=0.07)
    loss_fn.to(device)
    inner_loss = loss_fn(inner_net(meta_train_data), meta_train_label) 
    inner_loss.backward() 
    inner_opt.step()

    for p_tgt, p_src in zip(network.parameters(), inner_net.parameters()):
        if p_src.grad is not None:
            p_tgt.grad.data.add_(p_src.grad.data)

    loss_inner_j = loss_fn(inner_net(all_meta_data), all_mata_label) 
    grad_inner_j = autograd.grad(loss_inner_j, inner_net.parameters(), allow_unused=True) 

    for p, g_j in zip(network.parameters(), grad_inner_j):
            if g_j is not None:
                p.grad.data.add_(mldg_beta * g_j.data) 

    optimizer.step()

    model_outputs = network(all_meta_data)
    model_loss = loss_fn(model_outputs, all_mata_label)

    return loss_inner_j.item(), model_loss.item()


def train_meta_contrastive_extractor(model_name, dataset_name, dataset_path, device_index, saved_model_path, test_dataset_path):

    train_dataset, valid_dataset, test_dataset = data_process(dataset_name, dataset_path, device_index, test_dataset_path)

    if model_name == 'wisig_net_meta':
        model = Wisig_net_contrastive()
        scheduler_patience = 5
        early_stopping_patience = 20
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = SupConLoss(temperature=0.07)
    elif model_name == 'mobilenetv2_meta':
        model = MobileNetV2_encoder(width_mult=0.5)
        scheduler_patience = 10
        early_stopping_patience = 20
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = SupConLoss(temperature=0.07)
    elif model_name == 'resnet_50_meta':
        first_layer = ResNet_50_first_layer()
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_ftrs, 128)
        model = nn.Sequential(first_layer, resnet50)
        scheduler_patience = 5
        early_stopping_patience = 20
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = SupConLoss(temperature=0.07)

    print(ntx_list[ntx_list_index])
    model.to(device)
    loss_fn.to(device)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=scheduler_patience, verbose=True)
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=False, path=saved_model_path)

    batch_size = 512
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    inner_train_losses = []
    model_train_losses = []
    valid_losses = []

    meta_train_data_list = []
    meta_train_label_list = []

    for epoch in range(500):
        model.train(True)
        for data, label in train_data_loader:
            meta_train_classes = np.random.choice(ntx_list[ntx_list_index], int(ntx_list[ntx_list_index]*0.8), replace=False) # device class for meta_train.
            label_index = [i for i,v in enumerate(label) if np.isin(v, meta_train_classes)]
            for i in range(batch_size):
                if i in label_index:
                    meta_train_data_list.append(data[i])
                    meta_train_label_list.append(label[i])
            meta_train_data = torch.stack((meta_train_data_list))
            meta_train_label = torch.stack((meta_train_label_list))
            meta_train_data_list.clear()
            meta_train_label_list.clear()
            meta_train_data = meta_train_data.type(torch.cuda.FloatTensor)
            meta_train_label = meta_train_label.type(torch.cuda.LongTensor)
            data = data.type(torch.cuda.FloatTensor)
            label = label.type(torch.cuda.LongTensor)
            
            inner_loss_value, model_loss_value = MLDG(model, optimizer, 1, meta_train_data, meta_train_label, data, label)
            inner_train_losses.append(inner_loss_value)
            model_train_losses.append(model_loss_value)
        model.eval()
        for vdata, vlabels in valid_data_loader:
            vdata = vdata.type(torch.cuda.FloatTensor) 
            vlabels = vlabels.type(torch.cuda.LongTensor)
            voutputs = model(vdata)
            vloss = loss_fn(voutputs, vlabels)
            valid_losses.append(vloss.item())
        inner_train_loss = np.average(inner_train_losses)
        model_train_loss = np.average(model_train_losses)
        valid_loss = np.average(valid_losses)
        print('Epoch:', epoch+1, 'inner-train-loss: {} model-train-loss: {} vloss: {}'.format(inner_train_loss, model_train_loss, valid_loss))

        # clear lists to track next epoch
        inner_train_losses = []
        model_train_losses = []
        valid_losses = []

        scheduler.step(valid_loss)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

if __name__ == '__main__':

    dataset_name = 'ManyTx'
    dataset_path = './'
    saved_model_path = './checkpoint.pt'
    test_dataset_path = './test_dataset.pkl' # the path to save the test dataset of seen device
    ntx_list = [45, 80, 115, 150]
    ntx_list_index = 1

    run_for = 'MCRFF' # classifier or MCRFF- or MCRFF

    if run_for == 'classifier':
        model_name = 'wisig_net_classifier' # wisig_net_classifier // mobilenetv2_classifier // resnet_50_classifier
        train_classifier(model_name, dataset_name, dataset_path, ntx_list[ntx_list_index], saved_model_path, test_dataset_path)

    elif run_for == 'MCRFF-':
        model_name = 'wisig_net_contrastive' # wisig_net_contrastive // mobilenetv2_contrastive //resnet_50_contrastive
        train_contrastive_extractor(model_name, dataset_name, dataset_path, ntx_list[ntx_list_index], saved_model_path, test_dataset_path)

    elif run_for == 'MCRFF':
        model_name = 'wisig_net_contrastive' # wisig_net_meta // mobilenetv2_meta //resnet_50_meta
        train_meta_contrastive_extractor(model_name, dataset_name, dataset_path, ntx_list[ntx_list_index], saved_model_path, test_dataset_path)