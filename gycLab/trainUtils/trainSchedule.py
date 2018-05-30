import torch
import os
import shutil
from gycLab.imgUtils.imgNorm import ToPILImage,To_PIL_Label
from Criterion import Criterion
class Scheduler:
    def __init__(self,root, lr, total_epoches,lr_decay_rate,lr_decay_epoch,lr_min,eval_epoch,train_name):
        self._lr = lr
        self._total_epoches = total_epoches
        self._lr_decay_rate=lr_decay_rate
        self._lr_decay_epoch=lr_decay_epoch
        self._lr_min=lr_min
        self._eval_epoch=eval_epoch
        self._store_path=os.path.join(root,'result',train_name)
        if os.path.exists(self._store_path):
            print("store path is exists,do you want to replace it?(yes)")
            check=input()
            assert check=='yes',"error input"
            shutil.rmtree(self._store_path)
            os.makedirs(self._store_path)
        else:
            os.makedirs(self._store_path)


    def get_learning_rate(self):
        return self._lr

    def get_total_epoches(self):
        return self._total_epoches

    def decay_learning_rate(self):
        self._lr=self._lr/self._lr_decay_rate
        if self._lr<self._lr_min:
            self._lr=self._lr_min

    def get_decay_epoch(self):
        return self._lr_decay_epoch


    def get_eval_epoch(self):
        return self._eval_epoch

    def store_val_result(self,epoch,filename,img,label,pred,green=None):
        '''

        :param epoch:
        :param filename:
        :param img:should be 3*h*w tensor
        :param label:should be 1*h*w tensor
        :param pred:should be 1*h*w tensor
        :return:
        '''

        if len(label.size())==2:
            label=label.unsqueeze(0)#h*w
        store_path=os.path.join(self._store_path,'epoch%d'%(epoch))
        if not os.path.exists(store_path):
            os.mkdir(store_path)

        ###################pr#################
        true=label.numpy().flatten()
        fake=pred.numpy().flatten()
        Criterion().precision_recall(store_path , filename, true, fake)

        ######################################

        img=ToPILImage()(img)
        label=To_PIL_Label()(label)
        pred=To_PIL_Label()(pred)

        if green is not None:
           green=ToPILImage()(green)
           green.save(os.path.join(store_path, str(filename) + "_green.jpg"))
        img.save(os.path.join(store_path,str(filename)+"_img.jpg"))
        label.save(os.path.join(store_path, str(filename)+"_label.jpg"))
        pred.save(os.path.join(store_path, str(filename)+"_pred.jpg"))

    def store_model_dict(self,net,epoch):
        store_path=os.path.join(self._store_path,"model")
        if not os.path.exists(store_path):
            os.mkdir(store_path)
        torch.save(net.state_dict(), os.path.join(store_path,"epoch%d.pth"%epoch))
