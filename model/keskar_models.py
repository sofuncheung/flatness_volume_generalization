# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class C1(nn.Module):
    '''
    C1 network architecture in Keskar's paper
    "ON LARGE-BATCH TRAINING FOR DEEP LEARNING:
    GENERALIZATION GAP AND SHARP MINIMA"
    '''
    def __init__(self, num_classes=10):
        super(C1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            # (N, channels, 7, 7)
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            # (4, 4)
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            # (2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64*2*2, 384),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(384),
            nn.Dropout(),
            nn.Linear(384, 192),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(192),
            nn.Linear(192, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
