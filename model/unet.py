from copy import deepcopy
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop


# 2D U-Net Segmentation model 
class UNet(nn.Module):

    def __init__(self):
        super(UNet, self).__init__()

        self.conv_layers1 = nn.Sequential(
            nn.

        )




    def forward():
        pass 



# 3D U-Net Segmentation model for sparse annotation
class UNet3D(nn.Module):

    def __init__(self, num_classes):
        super(UNet3D, self).__init__()
        
        self.max_pool3d = nn.MaxPool3d(2,2)

        self.conv_layers1 = nn.Sequential(
            nn.Conv3d(4, 32, 3),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
        )

        self.conv_layers2 = nn.Sequential(
            nn.Conv3d(32, 64, 3),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 128, 3),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )

        self.conv_layers3 = nn.Sequential(
            nn.Conv3d(64, 128, 3),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Conv3d(128, 256, 3),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )

        # a.k.a bottleneck 
        self.conv_layers4 = nn.Sequential(
            nn.Conv3d(128, 256, 3),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Conv3d(256, 512, 3),
            nn.BatchNorm3d(512),
            nn.ReLU(),
        )


        self.conv_layers5 = nn.Sequential(
            nn.Conv3d(768, 256, 3),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Conv3d(256, 256, 3),
            nn.BatchNorm3d(256),
            nn.ReLU(),
        )

        self.conv_layers6 = nn.Sequential(
            nn.Conv3d(384, 128, 3),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Conv3d(128, 128, 3),
            nn.BatchNorm3d(128),
            nn.ReLU(),
        )


        self.conv_layers7 = nn.Sequential(
            nn.Conv3d(192, 64,3),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64,64,3),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 3, 1),
        )

        self.upconv1 = nn.ConvTranspose3d(512, 512, 2, 2)
        self.upconv2 = nn.ConvTranspose3d(256, 256, 2, 2)
        self.upconv3 = nn.ConvTranspose3d(128, 128, 2, 2)



    def forward(self, x):
        x = self.conv_layers1(x)
        x_res1 = x  # copy for residual
        x = self.max_pool3d(x)

        x = self.conv_layers2(x)
        x_res2 = x 
        x = self.max_pool3d(x)

        x = self.conv_layers3(x)
        x_res3 = x 
        x = self.max_pool3d(x)

        x = self.conv_layers4(x)
        x = self.upconv1(x)
        
        print("x size:", x.size())
        print("x_res3 size:", x_res3.size())
        
        x = torch.concat((x, x_res3), dim=1)

        x = self.conv_layers5(x)

        x = self.upconv2(x)

        x =  torch.concat((x, x_res2), dim=1)

        x = self.conv_layers6(x)

        x = self.upconv3(x)

        x = torch.concat((x, x_res1), dim=1)

        x = self.conv_layers7(x)

        return x 

