import torch
import torch.nn as nn
import torchvision
from torchvision import models
import math


class ResNet50Fc(nn.Module):

    def __init__(self, bottleneck_dim=256, class_num=1000):
        super(ResNet50Fc, self).__init__()
        self.model_resnet = models.resnet50(pretrained=True)

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
        self.bn2 = nn.BatchNorm1d(bottleneck_dim)
        self.fc = nn.Linear(bottleneck_dim, class_num)
        # self.fc = nn.Linear(model_resnet.fc.in_features, class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.bottleneck(x)
        x = self.bn2(x)
        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

    def parameters_list(self, lr):
        parameter_list = [
            {'params': self.conv1.parameters(), 'lr': lr / 10},
            {'params': self.bn1.parameters(), 'lr': lr / 10},
            {'params': self.maxpool.parameters(), 'lr': lr / 10},
            {'params': self.layer1.parameters(), 'lr': lr / 10},
            {'params': self.layer2.parameters(), 'lr': lr / 10},
            {'params': self.layer3.parameters(), 'lr': lr / 10},
            {'params': self.layer4.parameters(), 'lr': lr / 10},
            {'params': self.avgpool.parameters(), 'lr': lr / 10},
            {'params': self.bottleneck.parameters()},
            # {'params': self.bn2.parameters()},
            {'params': self.fc.parameters()},
        ]

        return parameter_list


class Discriminator(nn.Module):
    def __init__(self, bottleneck_dim=256):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(bottleneck_dim, 1)
        self.sigmoid = nn.Sigmoid()

        nn.init.kaiming_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        # x = torch.flatten(x)
        return x

    def parameters_list(self, lr):
        parameter_list = [
            {'params': self.fc.parameters()},
            {'params': self.sigmoid.parameters()}
        ]
        return parameter_list


class MultiClassify(nn.Module):

    def __init__(self, bottleneck_dim=256, class_num=1000):
        super(MultiClassify, self).__init__()

        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc2 = nn.Linear(bottleneck_dim, class_num)
        self.fc3 = nn.Linear(bottleneck_dim, class_num)
        self.fc4 = nn.Linear(bottleneck_dim, class_num)
        self.fc5 = nn.Linear(bottleneck_dim, class_num)

        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.kaiming_uniform_(self.fc3.weight)
        nn.init.kaiming_uniform_(self.fc4.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc5.weight, a=math.sqrt(5))

    def forward(self, x):
        y1 = self.fc1(x)
        y2 = self.fc2(x)
        y3 = self.fc3(x)
        y4 = self.fc4(x)
        y5 = self.fc5(x)

        return y1, y2, y3, y4, y5
