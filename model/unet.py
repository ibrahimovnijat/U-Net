import torch 
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, num_classes):
        super(UNet, self).__init__()

        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            
        )

        self.decoder = nn.Sequential(
            
        )



    def forward(self):
        pass 





