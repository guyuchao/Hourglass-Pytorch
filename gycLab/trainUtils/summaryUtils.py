from tensorboardX import SummaryWriter
import os
class Summary(object):
    def __init__(self,summary_name):
        '''

        :param summary_epoch:
        :param summary_name:
        '''
        self._logPath=os.path.join("log",summary_name)
        self._writer = SummaryWriter(self._logPath)


    def addTrainLoss(self,loss,epoch):
        self._writer.add_scalar('train loss', loss, epoch)

    def addValLoss(self,loss,epoch):
        self._writer.add_scalar('val loss', loss, epoch)

    def addLearningRate(self,lr,epoch):
        self._writer.add_scalar('learning rate', lr, epoch)

    def summaryEnd(self):
        self._writer.export_scalars_to_json(os.path.join(self._logPath,"all_scalars.json"))
        self._writer.close()

    def addPR_label_pred(self,label,prediction):
        self._writer.add_pr_curve('PR_curve', label,prediction, num_thresholds=1000)


