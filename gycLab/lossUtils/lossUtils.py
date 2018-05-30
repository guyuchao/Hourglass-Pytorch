import torch.nn.functional as F
import torch.nn as nn
import torch

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs,dim=1), targets)

class BceLoss2d(nn.Module):
    def __init__(self):
        super(BceLoss2d,self).__init__()
        self.bce=nn.BCELoss()

    def forward(self,inputs,targets):
        return self.bce(inputs,targets)

class MseLoss2d(nn.Module):
    def __init__(self):
        super(MseLoss2d, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, inputs, targets):
        return self.mse(inputs, targets)

class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss,self).__init__()

    def forward(self, input,target):
        input=F.softmax(input,dim=1)[:,1,:,:]
        smooth = 1.
        batch,_,_=input.size()
        iflat = input.view(batch,-1)
        tflat = target.view(batch,-1)
        intersection = torch.sum((iflat * tflat))
        return 1 - ((2. * intersection + smooth) /(iflat.sum() + tflat.sum() + smooth))

class CrossEntropyWithDiceLoss(nn.Module):
    def __init__(self,dice_percent=0.05, weight=None, size_average=True):
        super(CrossEntropyWithDiceLoss, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)
        self.dice_loss =Dice_loss()
        self._dice_percent=dice_percent

    def forward(self, inputs, targets):
        #print(inputs.size())
        return (1-self._dice_percent)*self.nll_loss(F.log_softmax(inputs,dim=1), targets)+self._dice_percent*self.diceloss(inputs,targets)