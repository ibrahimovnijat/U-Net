from copy import deepcopy
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import transforms


# 2D U-Net Segmentation model 

class DownConv2D(nn.Module):
    def __init__(self, _in, _out):
        super(DownConv2D, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=_in, out_channels=_out, kernel_size=3),
            # nn.BatchNorm2d(_out)
            nn.ReLU(),
            nn.Conv2d(in_channels=_out, out_channels=_out, kernel_size=3),
            # nn.BatchNorm2d(_out)
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # copy x for skip-connections
        x = self.conv_block(x) 
        x_copy = x  
        return [self.maxpool(x), x_copy]



class UpConv2D(nn.Module):
    def __init__(self, _in, _out):
        super(UpConv2D, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=_in, out_channels=_out, kernel_size=3),
            # nn.BatchNorm2d(_out)
            nn.ReLU(),
            nn.Conv2d(in_channels=_out, out_channels=_out, kernel_size=3),
            # nn.BatchNorm2d(_out)
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=_out, out_channels=_out/2, kernel_size=2),
        )

    def forward(self, x):
        return self.conv_block(x)


class DownConv3D(nn.Module):
    def __init__(self, _in, _out, _middle=None):
        super(DownConv3D, self).__init__()

        if _middle == None:
            _middle = _in

        self.conv_block = nn.Sequential(
            nn.Conv3d(_in, _middle, 3),
            nn.BatchNorm3d(_middle),
            nn.ReLU(),
            nn.Conv3D(_middle, _out, 3),
            nn.BatchNorm3d(_out),
            nn.ReLU(),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        def forward(self, x):
            x = self.conv_block(x)
            x_copy = x 
            return [self.maxpool(x), x_copy]


class UpConv3D(nn.Module):
    def __init__(self, _in, _out, _middle=None):
        super(UpConv3D, self).__init__()
        
        if _middle == None:
            _middle = _out

        self.conv_block = nn.Sequential(
            nn.Conv3d(_in, _middle, 3), 
            nn.BatchNorm3d(_middle),
            nn.ReLU(),
            nn.Conv3d(_middle, _out, 3),
            nn.BatchNorm3d(_out),
            nn.ReLU(),
            nn.ConvTranspose3d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.conv_block(x) 



class UNet2D(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(UNet2D, self).__init__()

        self.block1 = DownConv2D(in_ch, 64)
        self.block2 = DownConv2D(64, 128)
        self.block3 = DownConv2D(128, 256)
        self.block4 = DownConv2D(256, 512)

        self.block5 = UpConv2D(512, 1024)
        self.block6 = UpConv2D(1024, 512)
        self.block7 = UpConv2D(512, 256)
        self.block8 = UpConv2D(256, 128)
        self.block9 = UpConv2D(128, 64)
    
        self.out_layer = nn.Conv2d(64, out_ch, 1)


    def forward(self, x):
        x, x_copy = self.block1(x)
        x_skip1 = self._crop(x_copy, 392)
        
        x, x_copy = self.block2(x)
        x_skip2 = self._crop(x_copy, 200)
        
        x, x_copy = self.block3(x)
        x_skip3 = self._crop(x_copy, 104)
        
        x, x_copy = self.block4(x)
        x_skip4 = self._crop(x_copy, 56)
        
        x = self.block5(x)  

        x = self.block6(torch.cat((x_skip4, x), dim=1))

        x = self.block7(torch.cat((x_skip3, x), dim=1))

        x = self.block8(torch.cat((x_skip2, x), dim=1))

        x = self.block9(torch.cat((x_skip1, x), dim=1))

        x = self.out_layer(x)
    

    def _crop(self, x, size):
        transform = transforms.CenterCrop((size, size)) 
        return transform(x)



class UNet3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet3D, self).__init__()
        pass 

    def forward(self, x):
        pass 

