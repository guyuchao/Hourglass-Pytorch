import numpy as np
from sklearn.metrics import precision_recall_curve
class Metric:
    def f1_score(self,precision, recall,eps=1e-6):
        add_pr = precision + recall
        add_pr[add_pr == 0] = eps
        f1 = 2 * precision * recall / (add_pr)
        f1_argmax = np.argmax(f1)
        return f1.max(), f1_argmax

    def precision_recall(self,ytrue,ypred):
        precision, recall, thresholds = precision_recall_curve(ytrue, ypred)
        precision=precision[:,:,-1]
        recall=recall[:,:,-1]
        return precision,recall
