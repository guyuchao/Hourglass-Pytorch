from __future__ import division
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
from dataset.retinaPatchDataSet import retinaPatchDataSet
from torchvision.transforms import Compose, Normalize, ToTensor
import tqdm
from gycLab.netUtils.hourGlass import gethgmodel
from gycLab.imgUtils.imgAug import Random_horizontal_flip,Random_vertical_flip,Compose_imglabel
from gycLab.imgUtils.imgNorm import Un_normalize
from gycLab.trainUtils.trainSchedule import Scheduler
from gycLab.lossUtils.lossUtils import MseLoss2d
from gycLab.trainUtils.summaryUtils import Summary
#########################################
input_transform = Compose([
        ToTensor(),
        Normalize([.4974, .2706, .1624], [.3317, .1784, .0987]),
])
target_transform = Compose([
        ToTensor(),
])

img_label_transform = Compose_imglabel([
        Random_horizontal_flip(0.5),
        Random_vertical_flip(0.5),
])
traindataset = retinaPatchDataSet('./', img_transform=input_transform,label_transform=target_transform,image_label_transform=img_label_transform)
trainloader = data.DataLoader(traindataset, batch_size=32,shuffle=True)

valdataset = retinaPatchDataSet("./", img_transform=input_transform,split='val',label_transform=target_transform,image_label_transform=img_label_transform)
valloader = data.DataLoader(valdataset, batch_size=1,shuffle=False)
#########################################
#Parameters
train_name='PatchConnectionTrain1'
schedule=Scheduler(root='./',lr=2e-4,total_epoches=5000,lr_decay_rate=5,lr_decay_epoch=500,lr_min=1e-7,eval_epoch=100,train_name=train_name)
summary=Summary(summary_name=train_name)
unNorm=Un_normalize([.4974, .2706, .1624], [.3317, .1784, .0987])
model=gethgmodel().cuda()
criterion = MseLoss2d().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=schedule.get_learning_rate(), betas=(0.5, 0.9))
#########################################

for epoch in range(schedule.get_total_epoches()):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in tqdm.tqdm(enumerate(trainloader)):
        images = Variable(images.cuda())
        labels = Variable(labels.squeeze(1).cuda())
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(F.softmax(output[0],dim=1)[:,1,:,:], labels)
        for j in range(1, len(output)):
            loss += criterion(F.softmax(output[j],dim=1)[:,1,:,:], labels)
        loss.backward()
        running_loss+=loss.data[0]
        optimizer.step()
    print('epoch[%d/%d]  loss:%.5f'%(epoch,schedule.get_total_epoches(),running_loss/i))
    summary.addTrainLoss(running_loss/i,epoch)

    ##################################lr decay###############################################
    if epoch%schedule.get_decay_epoch()==0:
        schedule.decay_learning_rate()
        summary.addLearningRate(schedule.get_learning_rate(),epoch)
        optimizer = torch.optim.Adam(model.parameters(), lr=schedule.get_learning_rate(), betas=(0.5, 0.9))
    #########################################################################################
    ##################################eval epoch#############################################
    if epoch%schedule.get_eval_epoch()==0:
        model.eval()
        eval_loss=0.0
        for i, (images, labels) in tqdm.tqdm(enumerate(valloader)):
            images = Variable(images.cuda(),volatile=True)
            labels = Variable(labels.cuda())
            output = model(images)
            loss = criterion(output, labels)
            eval_loss += loss
            output=F.softmax(output[-1], dim=1)[:, 1, :, :].unsqueeze(1)
            #print(labels.size())
            schedule.store_val_result(epoch,i,unNorm(images.cpu().data[0]),labels.cpu().data[0],output.cpu().data[0])
        summary.addValLoss(eval_loss/i,epoch)
        schedule.store_model_dict(model,epoch)
summary.summaryEnd()
