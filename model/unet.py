import torch 
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

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

        self.block1 = DownConv2D(in_ch, 64, padding=1, usebatchnorm=True, bias=False)
        self.block2 = DownConv2D(64, 128,   padding=1, usebatchnorm=True, bias=False)
        self.block3 = DownConv2D(128, 256,  padding=1, usebatchnorm=True, bias=False)
        self.block4 = DownConv2D(256, 512,  padding=1, usebatchnorm=True, bias=False)
        
        self.block5 = UpConv2D(512, 1024, padding=1, usebatchnorm=True, bias=False)
        self.block6 = UpConv2D(1024, 512, padding=1, usebatchnorm=True, bias=False)
        self.block7 = UpConv2D(512, 256,  padding=1, usebatchnorm=True, bias=False)
        self.block8 = UpConv2D(256, 128,  padding=1, usebatchnorm=True, bias=False)
        self.block9 = UpConv2D(128, 64,   padding=1, upsampling=False, usebatchnorm=True, bias=False)
    
        self.out_layer = nn.Conv2d(64, out_ch, 1)
        self.sigmoid = nn.Sigmoid()
    
        

    def forward(self, x):
        x, x_skip1 = self.block1(x)
        
        x, x_skip2 = self.block2(x)

        x, x_skip3 = self.block3(x)
        
        x, x_skip4 = self.block4(x)
        
        x = self.block5(x)  

        x = torch.cat((x_skip4, x),dim=1)
        x = self.block6(x)


        x = torch.cat((x_skip3, x),dim=1)
        x = self.block7(x)


        x = torch.cat((x_skip2, x), dim=1)
        x = self.block8(x)

        x = torch.cat((x_skip1, x), dim=1)
        x = self.block9(x)

        x = self.out_layer(x)
        
        return self.sigmoid(x)

    
    def _crop(self, x, size):
        transform = transforms.CenterCrop((size, size)) 
        return transform(x)
    



class UNet3D(nn.Module):
    
    def __init__(self,in_ch, out_ch):
        pass 
    
    def forward(self, x):
        pass 
    