from torch import nn
from torchvision import models
import torch.nn.functional as F
import torch
from torch.nn.functional import Variable
class _DenseUpsamplingConvModule(nn.Module):
    def __init__(self, down_factor, in_dim, num_classes):
        super(_DenseUpsamplingConvModule, self).__init__()
        upsample_dim = (down_factor ** 2) * num_classes
        self.conv = nn.Conv2d(in_dim, upsample_dim, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(upsample_dim)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(down_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x

class DenseUpsamplingConvBlock(nn.Module):
    def __init__(self, upfactor, inplanes, outplanes):
        '''
        this block aims to upsample (Batch,C*L*L,W,H)->(Batch,C,L*W,L*H),use DUC Block to learn Upsampling, meanwhile concat the feature map
        :param upfactor: L
        :param inplanes:
        :param outplanes:
        '''
        super(DenseUpsamplingConvBlock, self).__init__()
        upsample_planes = (upfactor ** 2) * outplanes
        self.conv = nn.Conv2d(inplanes,upsample_planes , kernel_size=3, padding=1)#need to keep size
        self.bn = nn.BatchNorm2d(upsample_planes)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upfactor)#after this layer,output size should be B,outplanes,L*W,L*H

    def forward(self, x1,x2):
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


class ResNetDUCHDC(nn.Module):
    # the size of image should be multiple of 8
    def __init__(self, num_classes):
        super(ResNetDUCHDC, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool= resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        for n, m in self.layer3.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv2' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        layer3_group_config = [1, 2, 5, 9]
        for idx in range(len(self.layer3)):
            self.layer3[idx].conv2.dilation = (layer3_group_config[idx % 4], layer3_group_config[idx % 4])
            self.layer3[idx].conv2.padding = (layer3_group_config[idx % 4], layer3_group_config[idx % 4])
        layer4_group_config = [5, 9, 17]
        for idx in range(len(self.layer4)):
            self.layer4[idx].conv2.dilation = (layer4_group_config[idx], layer4_group_config[idx])
            self.layer4[idx].conv2.padding = (layer4_group_config[idx], layer4_group_config[idx])

        self.duc1 = _DenseUpsamplingConvModule(1, 2048,512)
        self.duc2 = DenseUpsamplingConvBlock(4,1024,64)
        self.duc3 = DenseUpsamplingConvBlock(2,128,num_classes)
        #self.duc4 = _DenseUpsamplingConvModule(2,64,num_classes)

    def forward(self, x):
        x0 = self.layer0(x)
        #print(x.size())#1,64,256,256
        x_pooling=self.maxpool(x0)
        #1,64,128,128
        x1 = self.layer1(x_pooling)
        #print(x.size())#1,256,128,128
        x2 = self.layer2(x1)
        #print(x.size())#1,512,64,64
        x3 = self.layer3(x2)
        #print(x.size())#1,1024,64,64
        x4 = self.layer4(x3)
        #print(x.size())#1,2048,64,64
        x = self.duc1(x4)
        x=self.duc2(x,x2)
        x=self.duc3(x,x0)
        #x=self.duc4(x)
        return x

def getmodel():
    model=ResNetDUCHDC(2)
    return model
    
if __name__=='__main__':
    t=Variable(torch.ones((1,3,128,128))).cuda()
    model=ResNetDUCHDC(2).cuda()
    model(t)
