import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.autograd as autograd
#from torchsummary import summary
from pytorchtools import EarlyStopping

from sklearn.model_selection import train_test_split

from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram
from models import MobileNetV2_extractor, LoRa_Net
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

def train_contrastive_extractor(dataset_path, saved_model_path, model_name):

    file_path = dataset_path
    dev_range = np.arange(0,30, dtype = int) 
    pkt_range = np.arange(0,500, dtype = int)
    snr_range = np.arange(20,80)

    LoadDatasetObj = LoadDataset()
    data, label = LoadDatasetObj.load_iq_samples(file_path, dev_range, pkt_range)
    
    data = awgn(data, snr_range)
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)
    data = np.reshape(data, (data.shape[0],1,102,62))
    label = np.reshape(label, (label.shape[0]))

    data_train, data_valid, label_train, label_valid = train_test_split(data, label, test_size=0.1, shuffle=True)

    if model_name == 'MobileNetV2':
        model = MobileNetV2_extractor(width_mult=0.5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif model_name == 'LoRa_Net':
        model = LoRa_Net()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)

    batch_size = 256
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=20, verbose=True)
    early_stopping = EarlyStopping(patience=40, verbose=False, path=saved_model_path)

    loss_fn = SupConLoss(temperature=0.05)
    loss_fn.to(device)
    model.to(device) 
    #summary(model, input_size=(1, 102, 62))

    train_dataset = data_preparation(data_train, label_train)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    validation_dataset = data_preparation(data_valid, label_valid)
    validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    valid_losses = []
    
    for epoch in range(500):
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
        model.train(False)
        for vdata, vlabels in validation_data_loader:
            vdata = vdata.type(torch.cuda.FloatTensor) 
            vlabels = vlabels.type(torch.cuda.LongTensor)
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

    print('Finished Training')

def MLDG(network, optimizer, mldg_beta, meta_train_data, meta_train_label, all_meta_data, all_mata_label):
    
    optimizer.zero_grad()

    for p in network.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)

    inner_net = type(network)()
    inner_net.load_state_dict(network.state_dict())
    inner_net.to(device)

    inner_opt = type(optimizer)(inner_net.parameters(), lr=optimizer.param_groups[0]['lr'], momentum=optimizer.param_groups[0]['momentum'])
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

def train_meta_contrstive_extractor(dataset_path, saved_model_path, model_name):

    file_path = dataset_path
    dev_range = np.arange(0,30, dtype = int) 
    pkt_range = np.arange(0,500, dtype = int)
    snr_range = np.arange(20,80)

    LoadDatasetObj = LoadDataset()
    data, label = LoadDatasetObj.load_iq_samples(file_path, dev_range, pkt_range)
    data = awgn(data, snr_range)
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    data = ChannelIndSpectrogramObj.channel_ind_spectrogram(data)
    data = np.reshape(data, (data.shape[0],1,102,62))
    label = np.reshape(label, (label.shape[0]))

    data_train, data_valid, label_train, label_valid = train_test_split(data, label, test_size=0.1, shuffle=True)

    batch_size = 256

    if model_name == 'MobileNetV2':
        model = MobileNetV2_extractor(width_mult=0.5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif model_name == 'LoRa_Net':
        model = LoRa_Net()
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10, verbose=True)
    early_stopping = EarlyStopping(patience=40, verbose=False, path=saved_model_path)

    loss_fn = SupConLoss(temperature=0.07)
    loss_fn.to(device)
    model.to(device) 
    #summary(model, input_size=(1, 102, 62))

    train_dataset = data_preparation(data_train, label_train)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    validation_dataset = data_preparation(data_valid, label_valid)
    validation_data_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    inner_train_losses = []
    model_train_losses = []
    valid_losses = []
    meta_train_data_list = []
    meta_train_label_list = []

    for epoch in range(500):
        model.train(True)
        for data, label in train_data_loader:
            meta_train_classes = np.random.choice(30, 24, replace=False) # device classes for meta_train.
            label_index = [i for i,v in enumerate(label) if np.isin(v, meta_train_classes)]
            for i in range(batch_size):
                if i in label_index:
                    meta_train_data_list.append(data[i])
                    meta_train_label_list.append(label[i])
            meta_train_data = torch.stack((meta_train_data_list)) # Concatenates a sequence of tensors along a new dimension.
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
        for vdata, vlabels in validation_data_loader:
            vdata = vdata.type(torch.cuda.FloatTensor) 
            vlabels = vlabels.type(torch.cuda.LongTensor)
            voutputs = model(vdata)
            vloss = loss_fn(voutputs, vlabels)
            valid_losses.append(vloss.item())
        inner_train_loss = np.average(inner_train_losses)
        model_train_loss = np.average(model_train_losses)
        valid_loss = np.average(valid_losses)
        print('Epoch:', epoch+1, 'inner-train-loss: {} model-train-loss: {} vloss: {}'.format(inner_train_loss, model_train_loss, valid_loss))

        inner_train_losses = []
        model_train_losses = []
        valid_losses = []

        scheduler.step(valid_loss)

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print('Finished Training')


if __name__ == '__main__':

    dataset_path = './dataset_training_no_aug.h5' 
    saved_model_path = './checkpoint.pt' # path to save the trained model
    model_name = 'MobileNetV2' # MobileNetV2 or LoRa_Net

    run_for = 'MCRFF' # MCRFF or MCRFF-

    if run_for == 'MCRFF-':
        train_contrastive_extractor(dataset_path, saved_model_path, model_name)

    if run_for == 'MCRFF':
        train_meta_contrstive_extractor(dataset_path, saved_model_path, model_name)