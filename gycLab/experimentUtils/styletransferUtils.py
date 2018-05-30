import torch
def Gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

def get_content_features(Vgg_net,img,mask):
    img=(img+1)*127.5
    img=img*((mask+1)/2)
    content_features = Vgg_net(img)[1]
    return content_features

def get_style_features(Vgg_net,img,mask):
    img=(img+1)*127.5
    img=img*((mask+1)/2)
    style_features = Vgg_net(img)
    # style_gram = [gram(fmap) for fmap in style_features]
    #style_feature_x = {}
    #style_feature_y = {}
    style_feature = {}
    for idx, feature in enumerate(style_features):
        #feature_x = feature[:, :, 1:, :] - feature[:, :, :-1, :]
        #feature_y = feature[:, :, :, 1:] - feature[:, :, :, :-1]
        #gram_x = Gram(feature_x)
        #gram_y = Gram(feature_y)
        gram = Gram(feature)
        #style_feature_x[idx] = gram_x
        #style_feature_y[idx] = gram_y
        style_feature[idx] = gram
    return style_feature

def get_style_loss(style_feature,fake_style_feature):
    style_loss=0.0
    for i in range(4):
        coff=float(1.0/4)
        fake_gram=fake_style_feature[i]
        style_gram=style_feature[i]
        style_loss+=coff*torch.mean(torch.abs((fake_gram-style_gram)))
    style_loss=torch.mean(style_loss)
    return style_loss

def get_content_loss(content_feature_real,content_feature_fake):
    coff=1
    content_loss=coff*torch.mean(torch.abs(content_feature_fake-content_feature_real))
    return content_loss

def get_tv_loss(img):
    x = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    y = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return x+y