from torch import nn
import torch
import torch.nn.functional as F

class resblock_torch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.conv_1 = nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1 )
        self.conv_2 = nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1 )
        self.bn2 = nn.BatchNorm2d(self.in_channels)
        
        self.conv_3 = nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1 )
        self.conv_4 = nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1 )
        self.bn3 = nn.BatchNorm2d(self.in_channels)

        self.conv_5 = nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1 )
        self.conv_6 = nn.Conv2d(self.in_channels, self.in_channels, (3, 3), padding=1 )
        self.bn4 = nn.BatchNorm2d(self.in_channels)
        self.bn5 = nn.BatchNorm2d(self.in_channels)
        
    # define forward pass
    def forward(self, input_):
        assert self.in_channels == input_.shape[1]
        x = self.bn1(input_)
        x = self.conv_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = x + input_

        x = self.conv_3(x)
        x = F.relu(x)
        x = self.conv_4(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = x + input_

        x = self.conv_5(x)
        x = F.relu(x)
        x = self.conv_6(x)
        x = F.relu(x)
        x = self.bn4(x)
        x = x + input_
        x = self.bn5(x)
        assert x.shape == input_.shape
        return x



class unet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1=nn.Conv2d(self.in_channels, 16, (2, 2), stride=2)
        self.res_1=resblock_torch(16)

        self.conv_2=nn.Conv2d(16, 32, (2, 2), stride = 2)
        self.res_2=resblock_torch(32)

        self.conv_3=nn.Conv2d(32, 64, (2, 2), stride = 2)
        self.res_3=resblock_torch(64)

        self.conv_4=nn.Conv2d(64, 128, (2, 2), stride = 2)
        self.res_4=resblock_torch(128)

        self.conv_5=nn.Conv2d(128, 256, (2, 2), stride = 2)
        self.res_5=resblock_torch(256)

        self.res_6=resblock_torch(256)
        self.res_7=resblock_torch(256)

        self.trconv_1=nn.ConvTranspose2d(256, 128, (2, 2), stride = 2)
        self.res_8=resblock_torch(128)

        self.trconv_2=nn.ConvTranspose2d(128, 64, (2, 2), stride = 2)
        self.res_9=resblock_torch(64)

        self.trconv_3=nn.ConvTranspose2d(64, 32, (2, 2), stride = 2)
        self.res_10=resblock_torch(32)

        self.trconv_4=nn.ConvTranspose2d(32, 16, (2, 2), stride = 2)
        self.res_11=resblock_torch(16)

        self.trconv_5=nn.ConvTranspose2d(16, 8, (2, 2), stride = 2)
        self.res_12=resblock_torch(8)

        self.conv_6=nn.Conv2d(8, 8, (3, 3), padding=1)
        self.bn1=nn.BatchNorm2d(8)
        self.conv_7=nn.Conv2d(8, 8, (3, 3), padding=1)
        self.bn2=nn.BatchNorm2d(8)
        self.out_conv=nn.Conv2d(8, self.out_channels, (1, 1), padding=0)
    
    # define forward pass
    def forward(self, input_):
        assert input_.shape[1] == self.in_channels
        x = input_
        x = self.conv_1(x)
        x = self.res_1(x)
        res1 = x.clone()

        x = self.conv_2(x)
        x = self.res_2(x)
        res2 = x.clone()

        x = self.conv_3(x)
        x = self.res_3(x)
        res3 = x.clone()

        x = self.conv_4(x)
        x = self.res_4(x)
        res4 = x.clone()
        
        x = self.conv_5(x)
        x = self.res_5(x)

        x = self.res_6(x)
        x = self.res_7(x)

        x = self.trconv_1(x)
        x = self.res_8(x)
        x = x + res4
        
        x = self.trconv_2(x)
        x = self.res_9(x)
        x = x + res3

        x = self.trconv_3(x)
        x = self.res_10(x)
        x = x + res2

        x = self.trconv_4(x)
        x = self.res_11(x)
        x = x + res1

        x = self.trconv_5(x)
        x = self.res_12(x)

        x = self.conv_6(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv_7(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.out_conv(x)
        # x = nn.Softmax(dim=1)(x)
        # print(x.shape)
        return x