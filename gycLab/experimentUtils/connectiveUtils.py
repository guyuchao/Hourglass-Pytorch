from skimage import graph
from PIL import Image
import numpy as np
import random

def getRandomPatch(label_pred,radius=16):
    h,w=label_pred.shape
    row, col = np.where(label_pred==label_pred.max())
    rnd = random.randint(0, len(row))
    rnd_row = row[rnd]
    rnd_col = col[rnd]
    if rnd_row-radius<0 and rnd_row+radius>h-1 and rnd_col-radius<0 and rnd_col-radius>w-1 :
        pass
    else:
        label_patch = label_pred[rnd_row - radius:rnd_row + radius, rnd_col - radius:rnd_col + radius]

def getBorderPoint(label,radius,size_padding):
    label[size_padding:2 * radius - size_padding, size_padding:2 * radius - size_padding] = 0

    label[0:size_padding-1, 0:2*radius] = 0
    label[2*radius-size_padding+1:2*radius,0:2*radius]=0
    label[0:2 * radius,0:size_padding-1] = 0
    label[0:2 * radius,2 * radius - size_padding + 1:2 * radius] = 0

    row, col = np.where(label == 255)

    return row, col

def getRandomImgLabel(img,label,radius=32,max_num=100000):
    img=np.array(img)
    label=np.array(label)
    h,w=label.shape
    row, col = np.where(label == 255)
    assert len(row)==len(col),"assert error"
    rnd = random.randint(0, len(row))
    rnd_row = row[rnd]
    rnd_col = col[rnd]
    if rnd_row-radius<0 or rnd_row+radius>h-1 or rnd_col-radius<0 or rnd_col-radius>w-1 :
        return None,None
    else:
        label_patch=label[rnd_row - radius:rnd_row + radius, rnd_col - radius:rnd_col + radius].astype(np.float32)
        img_patch = img[rnd_row - radius:rnd_row + radius, rnd_col - radius:rnd_col + radius,:]
        center=(radius,radius)
        row,col=getBorderPoint(np.copy(label_patch),radius,size_padding=4)
        label_patch[label_patch==0]=max_num
        label_patch_connect_center=np.zeros(label_patch.shape)
        for point in zip(row,col):
            if isConnect(label_patch,point,center):
                label_patch_connect_center=add_Gaussian_Peak(label_patch_connect_center,point)
                #label_patch_connect_center[point]=255
        label_patch_connect_center[label_patch_connect_center>255]=255
        return Image.fromarray(img_patch.astype(np.uint8)),Image.fromarray(label_patch_connect_center.astype(np.uint8))


def add_Gaussian_Peak(label_connect_center,point,sigma=1, type='Gaussian'):
    cx,cy=point
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    label_connect_center[int(cx-x0):int(cx+x0+1),int(cy-y0):int(cy+y0+1)]+=g*255
    return label_connect_center

def isConnect(img,begin,end,max_num=100000):
    path,cost=graph.route_through_array(img,begin,end)
    if cost>max_num:
        return False
    else:
        return True


