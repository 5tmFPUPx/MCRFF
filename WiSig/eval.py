import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

from data_utilities import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

from models import Wisig_net_classifier, MobileNetV2_Classifier, ResNet_50_first_layer, MobileNetV2_encoder, Wisig_net_contrastive
from pytorchtools import EarlyStopping
from SupConLoss import SupConLoss

from sklearn.metrics import roc_curve, auc , confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn import utils

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

def data_process(dataset_name, dataset_path, seen_device_index):

    compact_dataset = load_compact_pkl_dataset(dataset_path, dataset_name)

    tx_list = compact_dataset['tx_list']
    rx_list = compact_dataset['rx_list']
    capture_date_list = compact_dataset['capture_date_list']

    tx_train_list= tx_list[seen_device_index:150]

    equalized = 0
    dataset = merge_compact_dataset(compact_dataset,capture_date_list,tx_train_list,rx_list, equalized=equalized)
    all_augset =  prepare_dataset(dataset,tx_train_list,split=False)
    [sig,_,txid,_] = all_augset
    sig = np.reshape(sig, (sig.shape[0], 1, sig.shape[1], sig.shape[2]))
    txid = np.argmax(txid, axis=1) 

    return sig, txid

def test_with_seen_classes(model_name, saved_model_path, test_dataset_path, seen_device_index):
    f = open(test_dataset_path, 'rb')
    test_dataset = pickle.load(f)

    batch_size = 64
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    tdata_list = []
    tlabels_list = []

    for tdata, tlabels in test_data_loader:
        tdata_list.append(tdata)
        tlabels_list.append(tlabels)

    test_data = torch.cat((tdata_list),dim=0) 
    test_labels = torch.cat((tlabels_list), dim=0)

    test_data = tdata[0:tdata.shape[0]//5] 
    test_labels = tlabels[0:tlabels.shape[0]//5]

    test_data = test_data.to(torch.float32)

    if model_name == 'wisig_net_classifier':
        model = Wisig_net_classifier(num_classes=seen_device_index)
    elif model_name == 'mobilenetv2_classifier':
        model = MobileNetV2_Classifier(num_classes=seen_device_index)
    elif model_name == 'resnet_50_classifier':
        first_layer = ResNet_50_first_layer()
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_ftrs, seen_device_index)
        model = nn.Sequential(first_layer, resnet50)
    elif model_name == 'mobilenetv2_contrastive':
        model = MobileNetV2_encoder(width_mult=0.5)
    elif model_name == 'wisig_net_contrastive':
        model = Wisig_net_contrastive()
    elif model_name == 'resnet_50_contrastive':
        first_layer = ResNet_50_first_layer()
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_ftrs, 128)
        model = nn.Sequential(first_layer, resnet50)
    elif model_name == 'wisig_net_meta':
        model = Wisig_net_contrastive()
    elif model_name == 'mobilenetv2_meta':
        model = MobileNetV2_encoder(width_mult=0.5)
    elif model_name == 'resnet_50_meta':
        first_layer = ResNet_50_first_layer()
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_ftrs, 128)
        model = nn.Sequential(first_layer, resnet50)

    model.load_state_dict(torch.load(saved_model_path, map_location='cpu'))

    model.eval()

    feature_enrol = model(test_data)
    feature_enrol = feature_enrol.detach().numpy()

    knnclf=KNeighborsClassifier(n_neighbors=5, metric='cosine') 
    knnclf.fit(feature_enrol, np.ravel(test_labels)) 

    pred_label = knnclf.predict(feature_enrol)
    acc = accuracy_score(test_labels, pred_label)
    print('Overall accuracy of seen devices = %.4f' % acc)


def test_with_unseen_classes(model_name, saved_model_path, dataset_name, dataset_path, seen_device_index):

    tdata, tlabels = data_process(dataset_name, dataset_path, seen_device_index)

    tdata = torch.from_numpy(tdata) 
    tlabels = torch.from_numpy(tlabels)
    tdata, tlabels = utils.shuffle(tdata, tlabels)

    data_enrol = tdata[0:tdata.shape[0]//50]
    labels_enrol = tlabels[0:tlabels.shape[0]//50]
    data_enrol = data_enrol.type(torch.FloatTensor)

    if model_name == 'mobilenetv2_contrastive':
        model = MobileNetV2_encoder(width_mult=0.5)
    elif model_name == 'wisig_net_contrastive':
        model = Wisig_net_contrastive()
    elif model_name == 'resnet_50_contrastive':
        first_layer = ResNet_50_first_layer()
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_ftrs, 128)
        model = nn.Sequential(first_layer, resnet50)
    elif model_name == 'wisig_net_meta':
        model = Wisig_net_contrastive()
    elif model_name == 'mobilenetv2_meta':
        model = MobileNetV2_encoder(width_mult=0.5)
    elif model_name == 'resnet_50_meta':
        first_layer = ResNet_50_first_layer()
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False)
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_ftrs, 128)
        model = nn.Sequential(first_layer, resnet50)

    model.load_state_dict(torch.load(saved_model_path, map_location='cpu'))
    model.eval()

    feature_enrol = model(data_enrol)
    feature_enrol = feature_enrol.detach().numpy()
    knnclf=KNeighborsClassifier(n_neighbors=5, metric='cosine') 
    knnclf.fit(feature_enrol, np.ravel(labels_enrol)) 
    del feature_enrol, labels_enrol

    data_cls = tdata[-int(tdata.shape[0]//1000):]
    labels_cls = tlabels[-int(tlabels.shape[0]//1000):]
    data_cls = data_cls.type(torch.FloatTensor)

    feature_cls = model(data_cls)
    feature_cls = feature_cls.detach().numpy()
    pred_label = knnclf.predict(feature_cls)
    acc = accuracy_score(labels_cls, pred_label)
    print('Overall accuracy of unseen devices = %.4f' % acc)

if __name__ == '__main__':

    test_dataset_path='./test_data.pkl'
    saved_model_path = './checkpoint.pt'
    seen_device_index = 80
    dataset_name = 'ManyTx'
    dataset_path= './'

    run_for = 'MCRFF' # 'classifier' or 'MCRFF-' or 'MCRFF'

    if run_for == 'classifier':
        model_name = 'wisig_net_classifier' # wisig_net_classifier // mobilenetv2_classifier // resnet_50_classifier
        test_with_seen_classes(model_name, saved_model_path, test_dataset_path, seen_device_index)

    elif run_for == 'MCRFF-':
        model_name = 'wisig_net_contrastive' # wisig_net_contrastive // mobilenetv2_contrastive //resnet_50_contrastive
        test_with_seen_classes(model_name, saved_model_path, test_dataset_path, seen_device_index)
        test_with_unseen_classes(model_name, saved_model_path, dataset_name, dataset_path, seen_device_index)

    elif run_for == 'MCRFF':
        model_name = 'wisig_net_meta' # wisig_net_meta // mobilenetv2_meta //resnet_50_meta
        test_with_seen_classes(model_name, saved_model_path, test_dataset_path, seen_device_index)
        test_with_unseen_classes(model_name, saved_model_path, dataset_name, dataset_path, seen_device_index)