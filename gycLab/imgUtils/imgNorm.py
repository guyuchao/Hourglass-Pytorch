from PIL import Image
import numpy as np
import torch
from torchvision.transforms import ToTensor,ToPILImage



class Img_to_zero_center(object):
    def __int__(self):
        pass
    def __call__(self, t_img):
        '''

        :param img:tensor be 0-1
        :return:
        '''
        t_img=(t_img-0.5)*2
        return t_img

class Zero_center_to_img(object):
    def __int__(self):
        pass
    def __call__(self, z_img):
        z_img=(z_img+1)*127.5
        return z_img

class Input_vgg_net(object):
    '''
    bad performance
    '''
    def __int__(self):
        pass
    def __call__(self, img_tensor):
        np_img=img_tensor.numpy()
        mean = np.array([123.68, 116.779, 103.939])
        np_img=np.transpose(np_img,[0,2,3,1])
        np_img=np_img-mean
        np_img=np.transpose(np_img,[0,3,1,2])
        return torch.from_numpy(np_img).float()


class ToLabel(object):
    def __call__(self, inputs):
        return torch.from_numpy(np.array(inputs).astype(np.int32)).long()


class ReLabel(object):
    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, inputs):
        # assert isinstance(input, torch.LongTensor), 'tensor needs to be LongTensor'
        for i in inputs:
            i[i == self.olabel] = self.nlabel
        return inputs

class Compute_mean_std(object):
    def __init__(self,file_path_dict):
        self._file_path_dict=file_path_dict

    def compute(self):
        len_data=len(self._file_path_dict)
        mean=torch.zeros(3)
        std=torch.zeros(3)
        for index in range(len_data):
            datafiles = self._file_path_dict[index]
            imgpth=datafiles['img']
            img=np.array(Image.open(imgpth).convert('RGB'))
            img=ToTensor()(img)
            mean += img.view(img.size(0), -1).mean(1)
            std += img.view(img.size(0), -1).std(1)
        mean /= len_data
        std /= len_data
        meanstd = {
            'mean': mean,
            'std': std,
        }
        return meanstd

class Un_normalize(object):
    def __init__(self,mean,std):
        self._mean=mean
        self._std=std
    def __call__(self,tensor):
        '''

        :param tensor:should be C*H*W
        :return:
        '''
        for t, m, s in zip(tensor, self._mean, self._std):
            t.mul_(s).add_(m)
        return tensor

class To_PIL_Image(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        '''

        :param tensor:should be 3*h*w or 1*h*w
        :return:
        '''
        return ToPILImage()(tensor)
class To_PIL_Label(object):
    def __init__(self):
        pass
    def __call__(self, tensor):
        if len(tensor.size())==2:
            return Image.fromarray((tensor.numpy()*255).astype(np.uint8))
        if len(tensor.size())==3:
            return Image.fromarray((tensor.squeeze(0).numpy() * 255).astype(np.uint8))