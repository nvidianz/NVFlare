# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from executorch.exir import to_edge
from torch.export import export
from torch.export.experimental import _export_forward_backward
from torch.nn import functional as F


def export_model(net, input_tensor_example, label_tensor_example):
    # Captures the forward graph. The graph will look similar to the model definition now.
    # Will move to export_for_training soon which is the api planned to be supported in the long term.
    ep = export(net, (input_tensor_example, label_tensor_example), strict=True)
    # Captures the backward graph. The exported_program now contains the joint forward and backward graph.
    ep = _export_forward_backward(ep)
    # Lower the graph to edge dialect.
    ep = to_edge(ep)
    # Lower the graph to executorch.
    ep = ep.to_executorch()
    return ep


class Cifar10ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=2)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class XorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 4)
        self.sigmoid_1 = nn.Sigmoid()
        self.linear2 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.linear1(x)
        x = self.sigmoid_1(x)
        x = self.linear2(x)
        return x


# On device training requires the loss to be embedded in the model (and be the first output).
# We wrap the original model here and add the loss calculation. This will be the model we export.
class DeviceModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input, label):
        pred = self.net(input)
        return self.loss(pred, label), pred.detach().argmax(dim=1)
