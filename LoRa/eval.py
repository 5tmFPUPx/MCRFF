import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_curve, auc , confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
#from sklearn.manifold import TSNE

from dataset_preparation import awgn, LoadDataset, ChannelIndSpectrogram
from models import MobileNetV2_extractor, LoRa_Net

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

def test_classification(file_path_enrol, file_path_clf, model_name, saved_model_path, index_range):

    dev_range_enrol = np.arange(index_range[0],index_range[1], dtype = int)
    pkt_range_enrol = np.arange(0,100, dtype = int)
    dev_range_clf = np.arange(index_range[0],index_range[1], dtype = int)
    pkt_range_clf = np.arange(100,200, dtype = int)

    
    LoadDatasetObj = LoadDataset()
    data_enrol, label_enrol = LoadDatasetObj.load_iq_samples(file_path_enrol, 
                                                             dev_range_enrol, 
                                                             pkt_range_enrol)
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    data_enrol = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_enrol)

    data_enrol = np.reshape(data_enrol, (data_enrol.shape[0],1,102,62))
    label_enrol = np.reshape(label_enrol, (label_enrol.shape[0]))

    if model_name == 'MobileNetV2':
        model = MobileNetV2_extractor(width_mult=0.5)
    elif model_name == 'LoRa_Net':
        model = LoRa_Net()

    model.load_state_dict(torch.load(saved_model_path))
    #model.to(device)
    model.eval()

    data_enrol = torch.from_numpy(data_enrol) 
    data_enrol = data_enrol.type(torch.FloatTensor)
    feature_enrol = model(data_enrol)
    feature_enrol = feature_enrol.detach().numpy()

    knnclf=KNeighborsClassifier(n_neighbors=10, metric='cosine') 
    knnclf.fit(feature_enrol, np.ravel(label_enrol)) 

    data_clf, true_label = LoadDatasetObj.load_iq_samples(file_path_clf, dev_range_clf, pkt_range_clf)
    data_clf = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_clf)
    data_clf = np.reshape(data_clf, (data_clf.shape[0],1,102,62))
    true_label = np.reshape(true_label, (true_label.shape[0]))
    data_clf = torch.from_numpy(data_clf) 
    data_clf = data_clf.type(torch.FloatTensor)
    feature_clf = model(data_clf)
    feature_clf = feature_clf.detach().numpy()
    del data_clf

    pred_label = knnclf.predict(feature_clf)
    acc = accuracy_score(true_label, pred_label)
    print('Overall accuracy = %.4f' % acc)

    return pred_label, true_label, acc

def test_rogue_device_detection(file_path_enrol, file_path_legitimate, file_path_rogue, model_name, saved_model_path, legitimate_index_range, rogue_index_range):

    dev_range_enrol = np.arange(legitimate_index_range[0], legitimate_index_range[1], dtype = int)
    pkt_range_enrol = np.arange(0,100, dtype = int)
    dev_range_legitimate = np.arange(legitimate_index_range[0], legitimate_index_range[1], dtype = int)
    pkt_range_legitimate = np.arange(100,200, dtype = int)
    dev_range_rogue = np.arange(rogue_index_range[0], rogue_index_range[1], dtype = int)
    pkt_range_rogue = np.arange(0,100, dtype = int)

    def _compute_eer(fpr,tpr,thresholds):
        '''
        _COMPUTE_EER returns equal error rate (EER) and the threshold to reach
        EER point.
        '''
        fnr = 1-tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = np.mean((fpr[min_index], fnr[min_index]))
        
        return eer, thresholds[min_index]

    if model_name == 'MobileNetV2':
        model = MobileNetV2_extractor(width_mult=0.5)
    elif model_name == 'LoRa_Net':
        model = LoRa_Net()

    model.load_state_dict(torch.load(saved_model_path))
    model.eval()

    LoadDatasetObj = LoadDataset()
    data_enrol, label_enrol = LoadDatasetObj.load_iq_samples(file_path_enrol, dev_range_enrol, pkt_range_enrol)
    ChannelIndSpectrogramObj = ChannelIndSpectrogram()
    data_enrol = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_enrol)
    data_enrol = np.reshape(data_enrol, (data_enrol.shape[0],1,102,62))
    label_enrol = np.reshape(label_enrol, (label_enrol.shape[0]))

    data_enrol = torch.from_numpy(data_enrol) 
    data_enrol = data_enrol.type(torch.FloatTensor)
    feature_enrol = model(data_enrol)
    feature_enrol = feature_enrol.detach().numpy()
    del data_enrol

    # Build a K-NN classifier.
    knnclf=KNeighborsClassifier(n_neighbors=10, metric='cosine') # euclidean cosine
    knnclf.fit(feature_enrol, np.ravel(label_enrol))
    del feature_enrol

    # Load the test dataset of legitimate devices.
    data_legitimate, label_legitimate = LoadDatasetObj.load_iq_samples(file_path_legitimate, dev_range_legitimate, pkt_range_legitimate)
    # Load the test dataset of rogue devices.
    data_rogue, label_rogue = LoadDatasetObj.load_iq_samples(file_path_rogue, dev_range_rogue, pkt_range_rogue)
    
    data_test = np.concatenate([data_legitimate,data_rogue])
    label_test = np.concatenate([label_legitimate,label_rogue])
    data_test = ChannelIndSpectrogramObj.channel_ind_spectrogram(data_test)
    data_test = np.reshape(data_test, (data_test.shape[0],1,102,62))

    data_test = torch.from_numpy(data_test) 
    data_test = data_test.type(torch.FloatTensor)
    feature_test = model(data_test)
    feature_test = feature_test.detach().numpy()
    del data_test, model

    distances, indexes = knnclf.kneighbors(feature_test)
    detection_score = distances.mean(axis =1)

    true_label = np.zeros([len(label_test),1])
    true_label[(label_test <= dev_range_legitimate[-1]) & (label_test >= dev_range_legitimate[0])] = 1
    fpr, tpr, thresholds = roc_curve(true_label, detection_score, pos_label = 1) 
    fpr = 1-fpr
    tpr = 1-tpr

    eer, _ = _compute_eer(fpr,tpr,thresholds)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, roc_auc, eer


if __name__ == '__main__':

    run_for = 'Classification' # 'Classification' or 'Rogue Device Detection'
    
    if run_for == 'Classification':
        
        file_path_enrol = './dataset_other_device_type.h5'
        file_path_clf = './dataset_other_device_type.h5'
        model_name = 'MobileNetV2' # MobileNetV2 or LoRa_Net
        saved_model_path = './checkpoint.pt'
        # Specify the device index range for classification.
        index_range = [45,60]
        
        # Perform the classification task.
        pred_label, true_label, acc = test_classification(file_path_enrol, file_path_clf, model_name, saved_model_path, index_range)
        
        # Plot the confusion matrix.
        conf_mat = confusion_matrix(true_label, pred_label)
        test_dev_range = np.arange(index_range[0],index_range[1], dtype = int)
        classes = test_dev_range + 1
        
        plt.figure()
        sns.heatmap(conf_mat, annot=True, 
                    fmt = 'd', cmap='Blues',
                    cbar = False,
                    xticklabels=classes, 
                    yticklabels=classes)
        plt.xlabel('Predicted label', fontsize = 20)
        plt.ylabel('True label', fontsize = 20)
        plt.show()
        
    elif run_for == 'Rogue Device Detection':

        file_path_enrol = './dataset_seen_devices.h5',
        file_path_legitimate = './dataset_seen_devices.h5',
        file_path_rogue = './dataset_rogue',
        model_name = 'MobileNetV2' # MobileNetV2 or LoRa_Net
        saved_model_path = './checkpoint.pt'
        legitimate_index_range = [0,30]
        rogue_index_range = [40,45]

        fpr, tpr, roc_auc, eer = test_rogue_device_detection(file_path_enrol, file_path_legitimate, file_path_rogue, model_name, saved_model_path, legitimate_index_range, rogue_index_range)
        
        # Plot the ROC curves.
        plt.figure(figsize=(6.5, 5.1))
        plt.rcParams['font.sans-serif'] = 'Times New Roman'
        parameters = {'xtick.labelsize': 16, 'ytick.labelsize': 16}
        plt.rcParams.update(parameters)
        plt.xlim(-0.01, 1.0)
        plt.ylim(-0.01, 1.0)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='AUC = ' +  str(round(roc_auc,3)) + ', EER = ' + str(round(eer,3)), C='r')
        plt.xlabel('False positive rate', fontsize=18)
        plt.ylabel('True positive rate', fontsize=18)
        plt.title('ROC curve', fontsize=18)
        plt.legend(loc=4)
        # plt.savefig('roc_curve.pdf',bbox_inches='tight')
        plt.show()
