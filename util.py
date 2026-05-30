import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import termcolor
from skimage.metrics import peak_signal_noise_ratio as PSNR
from skimage.metrics import structural_similarity as SSIM
import lpips
import copy
import torchvision.models as models
from pytorch_msssim import ssim

class vgg_features(nn.Module):
    def __init__(self):
        super(vgg_features, self).__init__()
        self.net=nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

vgg=models.vgg19(pretrained=True).cuda()
vgg_feats=vgg_features().cuda()
vgg_feats.net.load_state_dict(vgg.features.state_dict())


# loss_fn_alex = lpips.LPIPS(net='vgg').cuda()
loss_fn_alex = lpips.LPIPS(net='alex').cuda()

def vgg_loss(x,y, agg=True):
    l1 = nn.L1Loss().cuda()
    l1_non_reduced=nn.L1Loss(reduction='none').cuda()
    xs = []
    ys = []
    i_s = [2, 7, 14]

    for i in range(len(vgg_feats.net)):
        x = vgg_feats.net[i](x)
        y = vgg_feats.net[i](y)
        if i in i_s:
            xs.append(x)
            ys.append(y)

    loss=0

    for i in range(3):
        loss+=l1(xs[i], ys[i])
    if agg:
        return loss
    pixel_wise_loss=0
    for i in range(3):
        pixel_wise_loss+=l1_non_reduced(xs[i], ys[i])
    return pixel_wise_loss

def freq_loss(x, y):
    x_fft=torch.fft.fft2(x)
    y_fft=torch.fft.fft2(y)
    real_diff=torch.abs(x_fft.real-y_fft.real).mean()
    imag_diff = torch.abs(x_fft.imag - y_fft.imag).mean()
    loss=real_diff+imag_diff
    # loss=torch.abs(x_fft-y_fft).mean()
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return (gauss).cuda()

def gen_gaussian_kernel(window_size, sigma):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(1, 1, window_size, window_size).contiguous())
    return window

def generate_gaussian_mask(size=256, frac=1/4):
    sigma = size * frac
    mask=gen_gaussian_kernel(size, sigma)
    # plt.figure(0)
    # plt.imsave('test.png', mask.squeeze().detach().cpu().numpy())
    # mask_ = 
    return mask

def generate_inverse_gaussian_mask(size=256, frac=1/3):
    """
    生成反向高斯mask，中心区域（低频）权重低，边缘区域（高频）权重高
    """
    # 先生成正常的高斯mask
    sigma = size / 4.5 * frac
    gaussian_mask = gen_gaussian_kernel(size, sigma)
    
    # 将高斯mask归一化到[0,1]范围
    normalized_gaussian = (gaussian_mask - gaussian_mask.min()) / (gaussian_mask.max() - gaussian_mask.min())
    
    # 取反：1 - 归一化的高斯mask，这样中心区域变为低权重，边缘区域变为高权重
    inverse_mask = 1.0 - normalized_gaussian
    mask_ = inverse_mask.squeeze()
    return inverse_mask

def generate_square_mask(size=256, frac=1/2):
    mask=torch.zeros((size, size)).cuda()
    cornerx = int((1-frac)/2*size)
    cornery = int((1 - frac) / 2 * size)
    wid=int(size*frac)
    mask[cornerx:cornerx+wid, cornery:cornery+wid]=1.
    mask=mask.unsqueeze(0).unsqueeze(0)
    # plt.figure(0)
    # plt.imshow(mask.squeeze().detach().cpu().numpy())
    # plt.show()
    return mask

def masked_freq_loss(x,y, frac=1/3):
    x_fft = torch.fft.fftshift(torch.fft.fft2(x))
    y_fft = torch.fft.fftshift(torch.fft.fft2(y))
    size = x.shape[-1]
    mask = generate_gaussian_mask(size, frac).repeat((x.shape[0], x.shape[1], 1, 1))
    # mask = generate_square_mask(size, frac)
    real_diff = (torch.abs(x_fft.real - y_fft.real)*mask).sum()/mask.sum()
    imag_diff = (torch.abs(x_fft.imag - y_fft.imag)*mask).sum()/mask.sum()
    loss = real_diff + imag_diff
    return loss

def charbonnier_loss(pred, target, eps=1e-12):
    """Charbonnier loss.

    Args:
        pred (Tensor): Prediction Tensor with shape (n, c, h, w).
        target ([type]): Target Tensor with shape (n, c, h, w).

    Returns:
        Tensor: Calculated Charbonnier loss.
    """
    return torch.sqrt((pred - target)**2 + eps).mean()

def multi_scale_loss(xs, y, scale_weights=[1.,1.,1.], mse_lambda=0., freq_lambda=0., lpips_lambda=0., l1_lambda=0., char_lambda=0., vgg_lambda=0., ssim_lambda=0.):
    '''

    :param xs: [H//4*W//4, H//2*W//2, H*W]
            y: H*W
    :return:
    '''
    ys=[]
    weights=scale_weights

    for x in xs:
        ys.append(F.interpolate(y, size=(x.shape[-2], x.shape[-1]), mode='bilinear'))

    mse=nn.MSELoss().cuda()
    l1=nn.L1Loss().cuda()

    global loss_fn_alex

    vgg_los=0
    char_loss=0
    l1_loss=0
    mse_loss=0
    freqloss=0
    lpips_loss=0
    ssim_loss=0
    for w, x, y in zip(weights, xs, ys):
        vgg_los += w * vgg_loss(x, y)
        char_loss += w * charbonnier_loss(x,y)
        l1_loss += w * l1(x, y)
        mse_loss += w * mse(x,y)
        freqloss += w * freq_loss(x, y)
        # freqloss += w * masked_freq_loss(x, y, frac=1/2, inverse=True)
        lpips_loss += w * loss_fn_alex.forward(x*2.-1., y*2.-1.).mean()
        ssim_loss += w * (1. - ssim(x, y, data_range=1.))

    loss=mse_lambda*mse_loss\
        +freq_lambda*freqloss+\
            lpips_lambda*lpips_loss+\
                l1_lambda*l1_loss+\
                    char_lambda*char_loss+\
                        vgg_lambda*vgg_los+\
                            ssim_lambda*ssim_loss

    psnr = 10 * torch.log(1 ** 2 / mse(xs[-1], ys[-1])) / np.log(10)
    return {
        'loss':loss,
        'mse': mse_loss,
        'freq loss': freqloss,
        'lpips loss': lpips_loss,
        'psnr':psnr
    }


def single_scale_loss(x, y, mse_lambda=0., l1_lambda=0., freq_lambda=0., lpips_lambda=0., ssim_lambda=0., char_lambda=0., vgg_lambda=0.):
    mse = nn.MSELoss().cuda()
    ae = nn.L1Loss().cuda()
    mse_loss=mse(x,y)
    ae_loss=ae(x,y)
    freq_los=freq_loss(x,y)
    lpips_los=loss_fn_alex.forward(x*2.-1, y*2.-1).mean()
    ssim_los = 1. - ssim(x,y, data_range=1.)
    char_los=charbonnier_loss(x,y)
    vgg_los=vgg_loss(x,y)
    loss = mse_lambda*mse_loss+freq_lambda*freq_los+lpips_lambda*lpips_los+ssim_lambda*ssim_los+l1_lambda*ae_loss+char_lambda*char_los+vgg_lambda*vgg_los

    psnr = 10 * torch.log(1 ** 2 / mse_loss) / np.log(10)

    return {
        'loss': loss,
        'mse': mse_loss,
        'freq loss': freq_los,
        'lpips loss': lpips_los,
        'psnr': psnr
    }


def cross_scale_loss(xs, mse_lambda=1., freq_lambda=0.1, lpips_lambda=0.2):
    xs_d_0=F.interpolate(xs[1], scale_factor=0.5, mode='bilinear')
    xs_d_1=F.interpolate(xs[2], scale_factor=0.5, mode='bilinear')

    mse = nn.MSELoss().cuda()
    mse_loss = mse(xs[0], xs_d_0) + mse(xs[1], xs_d_1)
    freqloss = freq_loss(xs[0], xs_d_0) + freq_loss(xs[1], xs_d_1)
    lpips_loss = loss_fn_alex.forward(xs[0] * 2. - 1., xs_d_0 * 2. - 1.).mean() \
                 + loss_fn_alex.forward(xs[1] * 2. - 1., xs_d_1 * 2. - 1.).mean()

    loss = mse_lambda * mse_loss + freq_lambda * freqloss + lpips_lambda * lpips_loss

    return loss



def curricular_ms_loss(blurred, xs, xs_c, y, mse_lambda=1., freq_lambda=0.1, lpips_lambda=0.2, l1_lambda=0.1, char_lambda=0.1, vgg_lambda=0.1,
                       mse_lambda_c=1., freq_lambda_c=0.1, lpips_lambda_c=0.2, l1_lambda_c=0.1, char_lambda_c=0.1, vgg_lambda_c=0.1):
    ys = [F.interpolate(y, scale_factor=0.25, mode='bilinear'), F.interpolate(y, scale_factor=0.5, mode='bilinear'), y]
    weights = [1, 1, 1]

    mse = nn.MSELoss().cuda()
    l1 = nn.L1Loss().cuda()

    global loss_fn_alex

    b = xs[0].shape[0]  # batch size

    losses=[]
    for k in range(b):
        vgg_los = 0
        char_loss = 0
        l1_loss = 0
        mse_loss = 0
        freqloss = 0
        lpips_loss = 0
        for i in range(3):
            vgg_los += weights[i] * vgg_loss(xs[i][k:k + 1], ys[i][k:k + 1])
            char_loss += weights[i] * charbonnier_loss(xs[i][k:k + 1], ys[i][k:k + 1])
            l1_loss += weights[i] * l1(xs[i][k:k + 1], ys[i][k:k + 1])
            mse_loss += weights[i] * mse(xs[i][k:k + 1], ys[i][k:k + 1])
            freqloss += weights[i] * freq_loss(xs[i][k:k + 1], ys[i][k:k + 1])
            lpips_loss += weights[i] * loss_fn_alex.forward(xs[i][k:k + 1] * 2. - 1., ys[i][k:k + 1] * 2. - 1.).mean()

        loss = mse_lambda * mse_loss + freq_lambda * freqloss + lpips_lambda * lpips_loss + l1_lambda * l1_loss + char_lambda * char_loss + vgg_lambda * vgg_los
        losses.append(loss)

    sample_weights = []
    for k in range(b):
        lpips_loss = 0
        for i in range(3):
            lpips_loss += weights[i] * loss_fn_alex.forward(xs_c[i][k:k + 1] * 2. - 1., ys[i][k:k + 1] * 2. - 1.).mean()
        freqloss = 0
        for i in range(3):
            freqloss += weights[i] * freq_loss(xs_c[i][k:k + 1], ys[i][k:k + 1])
        mse_loss = 0
        for i in range(3):
            mse_loss += weights[i] * mse(xs_c[i][k:k + 1], ys[i][k:k + 1])

        curriculum_criteria = mse_lambda_c * mse_loss + freq_lambda_c * freqloss + lpips_lambda_c * lpips_loss

        sample_weights.append(torch.exp(-1.5*curriculum_criteria))

        # import matplotlib.pyplot as plt
        # plt.figure(f'{k} blur')
        # plt.imshow(blurred[k].squeeze().detach().cpu().numpy().transpose(1, 2, 0))
        # plt.show()
        #
        # plt.figure(f'{k} 0')
        # plt.imshow(xs_c[-1][k].squeeze().detach().cpu().numpy().transpose(1,2,0))
        # plt.show()
        #
        # plt.figure(f'{k} 1')
        # plt.imshow(ys[-1][k].squeeze().detach().cpu().numpy().transpose(1,2,0))
        # plt.show()

    sum_sample_weights = sum(sample_weights)
    for k in range(b):
        sample_weights[k] = sample_weights[k]/sum_sample_weights

    total_loss=0
    for k in range(b):
        total_loss += sample_weights[k]*losses[k]
    total_loss = total_loss
    psnr = 10 * torch.log(1 ** 2 / mse(xs[2], ys[2])) / np.log(10)
    return {
        'loss': total_loss,
        'psnr': psnr
    }

def curriculum_ms_loss(xs, y, mse_lambda=1., freq_lambda=0., lpips_lambda=0., l1_lambda=0., char_lambda=0., vgg_lambda=0., stage=1, sigmas = [1/6, 1/5, 1/4]):

    ys = [F.interpolate(y, scale_factor=0.25, mode='bilinear'), F.interpolate(y, scale_factor=0.5, mode='bilinear'), y]

    weights = [1, 1, 1]
    # if stage == 1:
    #     weights = [1, 0, 0]
    # if stage == 2:
    #     weights = [0, 1, 0]
    # if stage == 3:
    #     weights = [0, 0, 1]

    mse = nn.MSELoss().cuda()
    l1 = nn.L1Loss().cuda()

    global loss_fn_alex

    vgg_los = weights[0] * vgg_loss(xs[0], ys[0]) + weights[1] * vgg_loss(xs[1], ys[1]) + weights[
        2] * vgg_loss(xs[2], ys[2])
    char_loss = weights[0] * charbonnier_loss(xs[0], ys[0]) + weights[1] * charbonnier_loss(xs[1], ys[1]) + weights[
        2] * charbonnier_loss(xs[2], ys[2])
    l1_loss = weights[0] * l1(xs[0], ys[0]) + weights[1] * l1(xs[1], ys[1]) + weights[2] * l1(xs[2], ys[2])
    mse_loss = weights[0] * mse(xs[0], ys[0]) + weights[1] * mse(xs[1], ys[1]) + weights[2] * mse(xs[2], ys[2])

    frac=1
    if len(sigmas)>=stage:
        frac=sigmas[stage-1]

    freqloss = weights[0] * masked_freq_loss(xs[0], ys[0], frac) + weights[1] * masked_freq_loss(xs[1], ys[1], frac) + \
               weights[2] * masked_freq_loss(xs[2], ys[2], frac)
    # freqloss = weights[0] * freq_loss(xs[0], ys[0]) + weights[1] * freq_loss(xs[1], ys[1]) + weights[2] * freq_loss(
    #     xs[2], ys[2])
    lpips_loss = weights[0] * loss_fn_alex.forward(xs[0] * 2. - 1., ys[0] * 2. - 1.).mean() \
                 + weights[1] * loss_fn_alex.forward(xs[1] * 2. - 1., ys[1] * 2. - 1.).mean() \
                 + weights[2] * loss_fn_alex.forward(xs[2] * 2. - 1., ys[2] * 2. - 1.).mean()

    loss = mse_lambda * mse_loss + freq_lambda * freqloss + lpips_lambda * lpips_loss + l1_lambda * l1_loss + char_lambda * char_loss + vgg_lambda * vgg_los

    psnr = 10 * torch.log(1 ** 2 / mse(xs[2], ys[2])) / np.log(10)
    return {
        'loss': loss,
        'mse': mse_loss,
        'freq loss': freqloss,
        'lpips loss': lpips_loss,
        'psnr': psnr
    }



def compute_metrics(out, gt):
    lpips_val=loss_fn_alex.forward(out * 2. - 1., gt * 2. - 1.).mean()
    out_numpy=out.squeeze().cpu().numpy().transpose(1,2,0).clip(0,1)
    gt_numpy=gt.squeeze().cpu().numpy().transpose(1,2,0).clip(0,1)
    psnr = PSNR(gt_numpy, out_numpy)
    ssim = SSIM(gt_numpy, out_numpy, channel_axis=-1, data_range=1)


    return {
        'psnr':psnr,
        'ssim':ssim,
        'lpips': lpips_val.item(),
        'out_numpy':out_numpy
    }


class SWA:
    def __init__(self, model: nn.Module, steps=100):
        self.params=copy.deepcopy(model.state_dict())
        self.total_steps=steps
        self.executed_step=0
        for k, v in self.params.items():
            self.params[k]*=0

    def progress(self, new_model: nn.Module):
        if self.executed_step==self.total_steps:
            print('SWA Finish')
        new_params=new_model.state_dict()
        for k, v in new_params.items():
            self.params[k]=(self.params[k]*self.executed_step+v)/(self.executed_step+1)
        self.executed_step += 1
        print(f'progress {self.executed_step}/{self.total_steps} steps ...')

    def save(self, save_p):
        torch.save(self.params, save_p)

def toRed(content):
    return termcolor.colored(content, "red", attrs=["bold"])


def toGreen(content):
    return termcolor.colored(content, "green", attrs=["bold"])


def toBlue(content):
    return termcolor.colored(content, "blue", attrs=["bold"])


def toCyan(content):
    return termcolor.colored(content, "cyan", attrs=["bold"])


def toYellow(content):
    return termcolor.colored(content, "yellow", attrs=["bold"])


def toMagenta(content):
    return termcolor.colored(content, "magenta", attrs=["bold"])


def toGrey(content):
    return termcolor.colored(content, "grey", attrs=["bold"])


def toWhite(content):
    return termcolor.colored(content, "white", attrs=["bold"])

if __name__ == '__main__':
    generate_gaussian_mask(256, 1/2)
