import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from MobileNetV2 import MobileNetV2

class Wisig_net_classifier(nn.Module):
    def __init__(self, num_classes=150):
        super(Wisig_net_classifier,self).__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,2), padding='same'), 
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,2), padding='same'), 
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,2), padding='same'), 
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,1), padding='same'), 
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,1), padding='same'),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, 80),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(80, num_classes),
        )
    
    def forward(self, x):
        x = self.feature_layers(x) # pytorch tensor shape is [N,C,H,W].
        self.feature = x
        x = self.fc(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class Wisig_net_contrastive(nn.Module):
    def __init__(self):
        super(Wisig_net_contrastive,self).__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3,2), padding='same'), 
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,2), padding='same'), 
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,2), padding='same'), 
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,1), padding='same'), 
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3,1), padding='same'),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x = self.feature_layers(x) # pytorch tensor shape is [N,C,H,W].
        self.feature = x
        x = self.fc(x)
        x = F.normalize(x, dim=1) # L2 normalization
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class MobileNetV2_Classifier(nn.Module):
    def __init__(self, num_classes=150, width_mult=0.5):
        super(MobileNetV2_Classifier,self).__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same'), #padding 'same' doesn’t support any stride values other than 1
            MobileNetV2(width_mult=width_mult)
        )
        self.fc = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.feature_layers(x) # pytorch tensor shape is [N,C,H,W]. [64,102,62,1] need to reshape to [64,1,102,62]
        self.feature = x
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class ResNet_50_first_layer(nn.Module):
    def __init__(self):
        super(ResNet_50_first_layer,self).__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same'), #padding 'same' doesn’t support any stride values other than 1
        )
    
    def forward(self, x):
        x = self.feature_layers(x) # pytorch tensor shape is [N,C,H,W]. [64,102,62,1] need to reshape to [64,1,102,62]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class MobileNetV2_encoder(nn.Module):
    def __init__(self, width_mult=0.5):
        super(MobileNetV2_encoder,self).__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same'),
            MobileNetV2(width_mult=0.5)
        )
        self.fc = nn.Sequential(
            #nn.Linear(1280, 512),
            #nn.BatchNorm1d(512),
            #nn.ReLU(),
            #nn.Dropout(p=0.5),
            nn.Linear(1280, 128),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.feature_layers(x) # pytorch tensor shape is [N,C,H,W]. [64,102,62,1] need to reshape to [64,1,102,62]
        self.feature = x
        x = self.fc(x)
        x = F.normalize(x, dim=1) # L2 normalization 
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()