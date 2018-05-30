import os
from PIL import Image
import numpy as np
import collections
from torch.utils import data
from gycLab.imgUtils.imgNorm import Compute_mean_std
from gycLab.imgUtils.imgAug import Retina_enhance,ColorAug

class retinaWithGreenDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None,image_label_transform=None,mask_transform=None):
        self.root = root
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.mask_transform=mask_transform
        self.image_label_transform=image_label_transform

        data_dir = os.path.join(root, "retina",split)
        imgsets_dir = os.path.join(data_dir,  "img")
        for name in os.listdir(imgsets_dir):
            name = os.path.splitext(name)[0]
            img_file = os.path.join(data_dir, "img/%s.tif" % name)
            label_file = os.path.join(data_dir, "label/%s.gif" % name)
            mask_file=os.path.join(data_dir,"mask/%s.gif"%name)

            self.files[split].append({
                "img": img_file,
                "label": label_file,
                "mask":mask_file
            })
            '''
            {'mean': 0.4974 0.2706 0.1624, 'std':0.3317 0.1784 0.0987}
            '''
        '''
        if split=='train':
            computeMeanStd=Compute_mean_std(self.files['train'])
            self.meanstd=computeMeanStd.compute()
        '''
    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("L")

        mask_file = datafiles["mask"]
        mask = Image.open(mask_file).convert('L')
        img, greenimg = Retina_enhance()(img, mask)

        if self.split=='train':
            img=ColorAug(0.5)(img)


        if self.image_label_transform is not None:
            img,label,greenimg=self.image_label_transform(img,label,greenimg)

        if self.img_transform is not None:
            img= np.array(img)
            greenimg=np.array(greenimg)
            img=np.dstack((img,greenimg))
            img= self.img_transform(img)

        if self.label_transform is not None:
            label= self.label_transform(label)
        return img, label

if __name__=='__main__':
    dataset=retinaDataSet(os.path.abspath(os.curdir))
    print(dataset.meanstd)