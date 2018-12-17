import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class HPA_Res18(nn.Module):
    ''' Transfer learning with Resnet18,
        input is 4 channels. '''
    
    def __init__(self):
        super(HPA_Res18, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.in_c = 4
        self.out_f = 28
        # conv1 setting
        conv1 = self.model.conv1
        conv1_w = conv1.weight.data
        out_c = conv1.out_channels
        kernel_s = conv1.kernel_size
        stride = conv1.stride
        padding = conv1.padding
        conv1_new = nn.Conv2d(self.in_c, out_c, kernel_size=kernel_s, stride=stride, padding=padding, bias=False)
        conv1_new.weight.data = torch.cat((conv1_w, conv1_w[:,-1].unsqueeze(1)), dim=1)
        # fc setting
        fc = self.model.fc
        fc_w = fc.weight.data
        fc_b = fc.bias.data
        in_f = fc.in_features
        fc_new = nn.Linear(in_f, self.out_f, bias=True)
        fc_new.weight.data = fc_w[:self.out_f, :]
        fc_new.bias.data = fc_b[:self.out_f]
        
        self.model.conv1 = conv1_new
        self.model.fc = fc_new
        
    def forward(self, x):
        return F.sigmoid(self.model(x))
    
class FocalLoss_Plus(nn.Module):
    ''' Self defined Focal Loss to address imbalance, 
        taking into account relationship between  "Endosomes" and "Lysosomes"'''
    
    def __init__(self, gamma, weight=None):
        ''' 
        Args:
            weight(tensor, optional): balenced weights for each entry of target.
            gamma(int): focusing parameter for focal loss
        '''
        super(FocalLoss_Plus, self).__init__()
        self.weight = weight
        self.gamma = gamma
        
    def forward(self, input, target):
        '''
        Args:
            input: the output of Conv model.
            target: ground truth labels
        '''
        self.focal_w = (target*(1-input)+(1-target)*input).pow(self.gamma)
        if self.weight is not None:
            self.final_weight = self.focal_w * self.weight
        else:
            self.final_weight = self.focal_w
#         self.bce = nn.BCELoss(weight=self.final_weight)
#         focal = self.bce(input, target)
        
        focal = torch.mean(self.final_weight * (target*input.log() + (1-target)*(1-input).log()) * -1.0)
        relate = self.weight[9] * torch.mean((input[:,9]-input[:,10]).abs())

        return focal + relate