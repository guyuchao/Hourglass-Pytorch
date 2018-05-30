import torch
from torch.autograd import Variable,grad

def calc_gradient_penalty(netD, real_data, fake_data,LAMBDA=10):
    '''

    :param netD:
    :param real_data:
    :param fake_data:
    :param LAMBDA:
    :return:
    '''
    BATCH=real_data.size()[0]
    alpha = torch.rand(BATCH, 1)
    #print(alpha.size(),real_data.size())
    alpha = alpha.unsqueeze(-1).unsqueeze(-1).expand(real_data.size())
    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
