
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision.models as models


class resnet_encoder(nn.Module):

    def __init__(self, pretrained = True,  **kwargs):
        super(resnet_encoder, self).__init__()

        # RGB encoder
        resnet18_model = models.resnet18(pretrained=pretrained)
        self.resnet18_encoder = nn.Sequential(*(list(resnet18_model.children())[:-2]))
        
    def forward(self, x):
        ret = self.resnet18_encoder(x)
        return ret


