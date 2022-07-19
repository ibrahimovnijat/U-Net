from copy import deepcopy
from turtle import forward
from numpy import block
import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import transforms


# 2D U-Net Segmentation model 

class DownConv2D(nn.Module):
    
    def __init__(self, _in, _out, kernel=3, stride=1, padding=0, usebatchnorm=False, bias=True):
        super(DownConv2D, self).__init__()

        block_layers = []
        block_layers.append(nn.Conv2d(in_channels=_in, out_channels=_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias))
        if usebatchnorm:
            block_layers.append(nn.BatchNorm2d(_out))
        block_layers.append(nn.ReLU(inplace=True))
        block_layers.append(nn.Conv2d(in_channels=_out, out_channels=_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias))
        if usebatchnorm:
            block_layers.append(nn.BatchNorm2d(_out))
        block_layers.append(nn.ReLU(inplace=True))

        self.conv_block = nn.Sequential(*block_layers)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv_block(x)
        xcopy = x 
        return [self.maxpool(x), xcopy]
    


class UpConv2D(nn.Module):

    def __init__(self, _in, _out, kernel=3, stride=1, padding=0, usebatchnorm=False, upsampling=True, bias=True):
        super(UpConv2D, self).__init__()

        block_layers = []
        block_layers.append(nn.Conv2d(in_channels=_in, out_channels=_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias))
        if usebatchnorm:
            block_layers.append(nn.BatchNorm2d(_out))
        block_layers.append(nn.ReLU(inplace=True))
        block_layers.append(nn.Conv2d(in_channels=_out, out_channels=_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias))
        if usebatchnorm:
            block_layers.append(nn.BatchNorm2d(_out))
        block_layers.append(nn.ReLU(inplace=True))
        if upsampling:
            block_layers.append(nn.ConvTranspose2d(in_channels=_out, out_channels=int(_out/2), kernel_size=2, stride=2))

        self.conv_block = nn.Sequential(*block_layers)

    def forward(self, x):
        return self.conv_block(x)


class DownConv3D(nn.Module):
    def __init__(self, _in, _out, _middle=None, kernel=3, stride=1, padding=0):
        super(DownConv3D, self).__init__()

        if _middle == None:
            _middle = _in

        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels=_in, out_channels=_middle, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm3d(_middle),
            nn.ReLU(inplace=True),
            nn.Conv3D(in_channels=_middle, out_channels=_out, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm3d(_out),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        def forward(self, x):
            x = self.conv_block(x)
            x_copy = x 
            return [self.maxpool(x), x_copy]


class UpConv3D(nn.Module):
    def __init__(self, _in, _out, _middle=None, kernel=3, stride=1, padding=0):
        super(UpConv3D, self).__init__()
        
        if _middle == None:
            _middle = _out

        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels=_in, out_channels=_middle, kernel_size=kernel, stride=stride, padding=padding), 
            nn.BatchNorm3d(_middle),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=_middle, out_channels=_out, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm3d(_out),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.conv_block(x) 


class UNet2D(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.block1 = DownConv2D(in_ch, 64)
        self.block2 = DownConv2D(64, 128)
        self.block3 = DownConv2D(128, 256)
        self.block4 = DownConv2D(256, 512)

        self.block5 = UpConv2D(512, 1024)
        self.block6 = UpConv2D(1024, 512)
        self.block7 = UpConv2D(512, 256)
        self.block8 = UpConv2D(256, 128)
        self.block9 = UpConv2D(128, 64, upsampling=False)
    
        self.out_layer = nn.Conv2d(64, out_ch, 1)
        self.sigmoid = nn.Sigmoid()
    

    def forward(self, x):
        x, x_skip1 = self.block1(x)
        x_skip1 = self._crop(x_skip1, 392)
        
        # print("x size after block1:", x.size())
        
        x, x_skip2 = self.block2(x)
        x_skip2 = self._crop(x_skip2, 200)

        # print("x size after block2:", x.size())

        x, x_skip3 = self.block3(x)
        x_skip3 = self._crop(x_skip3, 104)
        
        # print("x size after block3:", x.size())
        
        x, x_skip4 = self.block4(x)
        x_skip4 = self._crop(x_skip4, 56)
        
        # print("x size after block4:", x.size())
        
        x = self.block5(x)  

        # print("x size after block5:", x.size())

        x = torch.cat((x_skip4, x),dim=1)
        x = self.block6(x)

        # print("x size after block6:", x.size())

        x = torch.cat((x_skip3, x),dim=1)
        x = self.block7(x)

        # print("x size after block7:", x.size())

        x = torch.cat((x_skip2, x), dim=1)
        x = self.block8(x)

        # print("x size after block8:", x.size())

        x = torch.cat((x_skip1, x), dim=1)
        x = self.block9(x)

        # print("x size after block9:", x.size())

        x = self.out_layer(x)

        # print("x size after out_layer:", x.size())

        return torch.sigmoid(x)
        # return self.sigmoid(x)


    def _crop(self, x, size):
        transform = transforms.CenterCrop((size, size)) 
        return transform(x)







# new unet for (256, 256) with padding=1
class UNET2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()      

        self.block1 = DownConv2D(3, 64, stride=1)
        self.block2 = DownConv2D(64, 128, stride=1)
        pass 


    def forward(self, x):
        pass 
