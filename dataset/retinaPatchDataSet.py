import os
from PIL import Image
import collections
from torch.utils import data
from gycLab.imgUtils.imgNorm import Compute_mean_std

class retinaPatchDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None,image_label_transform=None):
        self.root = root
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.image_label_transform=image_label_transform

        data_dir = os.path.join(root, "retinaPatch",split)
        imgsets_dir = os.path.join(data_dir,  "img")
        for name in os.listdir(imgsets_dir):
            name = os.path.splitext(name)[0]
            img_file = os.path.join(data_dir, "img/%s.tif" % name)
            label_file = os.path.join(data_dir, "label/%s.gif" % name)

            self.files[split].append({
                "img": img_file,
                "label": label_file,
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

        if self.image_label_transform is not None:
            img,label=self.image_label_transform(img,label)

        if self.img_transform is not None:
            img= self.img_transform(img)

        if self.label_transform is not None:
            label= self.label_transform(label)

        return img, label

if __name__=='__main__':
    dataset=retinaDataSet(os.path.abspath(os.curdir))
    print(dataset.meanstd)