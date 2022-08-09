import torch
import torch.nn as nn
import math
from MobileNetV2 import MobileNetV2
import torch.nn.functional as F

class MobileNetV2_extractor(nn.Module):
    def __init__(self, width_mult=0.5):
        super(MobileNetV2_extractor,self).__init__()
        self.feature_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=2),
            MobileNetV2(width_mult=width_mult)
        )
        self.fc = nn.Sequential(
            nn.Linear(1280, 128),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.feature_layers(x) 
        self.feature = x
        x = self.fc(x)
        x = F.normalize(x, dim=1) 
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


class LoRa_Net(nn.Module):
    def __init__(self):
        super(LoRa_Net,self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7, stride=2), 
            nn.ReLU(),
        )
        self.resblock_c32 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same'),
        )
        self.resblock_ic32_oc64 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
        )
        self.resblock_ic64_oc64 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding='same'),
        )
        self.resblock_1_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, padding='same'), 
        )
        self.relu = nn.Sequential(
            nn.ReLU(),
        )
        self.avgpool = nn.AvgPool2d((2, 2))
        self.fc = nn.Sequential(
            nn.Linear(21504, 512),
        )
    
    def forward(self, x):
        x = self.first_layer(x) 
        #resblock1
        fx = self.resblock_c32(x)
        x = torch.add(x,fx)
        x = self.relu(x)
        #resblock2
        fx = self.resblock_c32(x)
        x = torch.add(x,fx)
        x = self.relu(x)
        #resblock3
        fx = self.resblock_ic32_oc64(x)
        x = self.resblock_1_conv(x)
        x = torch.add(x,fx)
        x = self.relu(x)
        #resblock4
        fx = self.resblock_ic64_oc64(x)
        x = torch.add(x,fx)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1) # flatten()
        x = self.fc(x)
        x = F.normalize(x, dim=1)
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