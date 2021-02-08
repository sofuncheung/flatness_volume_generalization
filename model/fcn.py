# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class FCN(nn.Module):
    '''
    Fully connected neural network.
    '''
    def __init__(self, input_dim=28*28,
            num_hidden_layers=2,
            hidden_layer_width=40,
            num_classes=1):
        super(FCN, self).__init__()

        layers = OrderedDict()
        layers.update({'fc1': nn.Linear(input_dim,hidden_layer_width)})
        layers.update({'relu1': nn.ReLU()})
        layers.move_to_end('relu1', last=True)
        for i in range(num_hidden_layers-1):
            attr_name = 'fc'+str(i+2)
            relu_name = 'relu'+str(i+2)
            layers.update({attr_name: nn.Linear(hidden_layer_width,hidden_layer_width)})
            layers.move_to_end(attr_name, last=True)
            layers.update({relu_name: nn.ReLU()})
            layers.move_to_end(relu_name, last=True)
        layers.update({'fc'+str(num_hidden_layers+1):
            nn.Linear(hidden_layer_width, num_classes)})
        layers.move_to_end('fc'+str(num_hidden_layers+1), last=True)

        self.layers = nn.Sequential(layers)


    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    net = FCN()
    print(net)
