from PIL import Image,ImageEnhance
import numpy as np
import random
from skimage import exposure
import torch

class Compose_imglabel(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img,label,green=None):
        if green is None:
            for t in self.transforms:
                img,label = t(img,label)
            return img,label
        else:
            for t in self.transforms:
                img,label,green = t(img,label,green)
            return img,label,green
class Compose_imgmask(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img,mask):
        for t in self.transforms:
            img,mask = t(img,mask)
        return img,mask




class Retina_enhance(object):
    def __call__(self, img,mask):
        '''
        green channel
        :param img:should be pil image
        :return:4-dimension image (l,a,b,g-enhance)
        '''
        npimg=np.array(img)
        mask=np.array(mask)
        g_enhance = exposure.equalize_hist(npimg[:,:,1])
        g_enhance = exposure.adjust_gamma(g_enhance, 1)
        g_enhance[mask==0]=0
        g_enhance*=255
        return Image.fromarray(npimg.astype(np.uint8)),Image.fromarray(g_enhance.astype(np.uint8))

class Random_vertical_flip(object):
    def _vertical_flip(self,img,label,green):
        if green is None:
            return img.transpose(Image.FLIP_TOP_BOTTOM),label.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return img.transpose(Image.FLIP_TOP_BOTTOM), label.transpose(Image.FLIP_TOP_BOTTOM),green.transpose(Image.FLIP_TOP_BOTTOM)
    def __init__(self,prob):
        '''

        :param prob: should be (0,1)
        '''
        assert prob>=0 and prob<=1,"prob should be [0,1]"
        self.prob=prob
    def __call__(self, img,label,green=None):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        assert isinstance(img, Image.Image),"should be PIL image"
        assert isinstance(label, Image.Image),"should be PIL image"
        if random.random()<self.prob:
            return self._vertical_flip(img,label,green)
        else:
            if green is None:
                return img,label
            else:
                return img,label,green

class Random_horizontal_flip(object):
    def _horizontal_flip(self,img,label,green):
        #dsa
        if green is None:
            return img.transpose(Image.FLIP_LEFT_RIGHT),label.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img.transpose(Image.FLIP_LEFT_RIGHT), label.transpose(Image.FLIP_LEFT_RIGHT), green.transpose(Image.FLIP_LEFT_RIGHT)

    def __init__(self,prob):
        '''

        :param prob: should be (0,1)
        '''
        assert prob>=0 and prob<=1,"prob should be [0,1]"
        self.prob=prob

    def __call__(self, img,label,green=None):
        '''
        flip img and label simultaneously
        :param img:should be PIL image
        :param label:should be PIL image
        :return:
        '''
        assert isinstance(img, Image.Image),"should be PIL image"
        assert isinstance(label, Image.Image),"should be PIL image"
        if random.random()<self.prob:
            return self._horizontal_flip(img,label,green)
        else:
            if green is None:
                return img,label
            else:
                return img,label,green

class ColorAug(object):
    def _randomColor(self,image):
        """
        :param image: PIL的图像image
        :return:
        """
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)

    def __init__(self,prob):
        self._prob=prob

    def __call__(self, img):
        '''

        :param img:should be PIL image
        :return:
        '''
        assert isinstance(img, Image.Image),"should be PIL image"
        if random.random() < self._prob:
            return self._randomColor(img)
        else:
            return img

class Add_Gaussion_noise(object):
    def _gaussian_noise(self,npimg,mean=0,sigma=0):
        noise=np.random.normal(mean,sigma,npimg.shape)
        newimg=npimg+noise
        newimg[newimg>255]=255
        newimg[newimg<0]=0
        return Image.fromarray(newimg.astype(np.uint8))

    def __init__(self,prob):
        self.prob=prob

    def __call__(self,img,mask):
        return self._gaussian_noise(np.array(img),0,random.randint(0,15))

class Random_rotation(object):
    def _randomRotation(self,image,label, mode=Image.NEAREST):
        """
         对图像进行随机任意角度(0~360度)旋转
        :param mode 邻近插值,双线性插值,双三次B样条插值(default)
        :param image PIL的图像image
        :return: 旋转转之后的图像
        """
        random_angle = np.random.randint(0, 360)
        return image.rotate(random_angle, mode)

    def __init__(self,prob):
        self.prob=prob

    def __call__(self, img,label):
        return self._randomRotation(img,label)

class Random_crop(object):
    def _randomCrop(self,img,label,green):
        width, height = img.size
        x, y = random.randint(0, width - self.width), random.randint(0, height -self.height)
        region = [x, y, x + self.width, y + self.height]
        if green is None:
            return img.crop(region),label.crop(region)
        else:
            return img.crop(region),label.crop(region),green.crop(region)

    def __init__(self,height,width):
        self.height=height
        self.width=width

    def __call__(self,img,label,green=None):
        assert img.size==label.size,"img should have the same shape as label"
        width,height=img.size
        assert height>=self.height and width>=self.width,"Cropimg should larger than origin"
        return self._randomCrop(img,label,green)

