import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
import einops
from ODConv import DOConv2d
from fft_arch_util import fft_bench_complex_mlp


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=False, norm=False, relu=True,
                 transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            # layers.append(
            #     nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias,
            #               padding_mode='reflect'))
            layers.append(
                DOConv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.ReLU(inplace=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class LayerNorm2d(nn.Module):
    def __init__(self, dim):
        super(LayerNorm2d, self).__init__()
        self.norm=nn.LayerNorm(dim)
    def forward(self,x):
        '''

        :param x: b c h w
        :return:
        '''
        x=x.permute(0,2,3,1)
        x=self.norm(x)
        x=x.permute(0,3,1,2)
        return x


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = AttentionGate()

    def forward(self, x):
        if not self.no_spatial:
            x_out = 1 / 3 * (self.hw(x) + self.cw(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1,
                                                                                              3).contiguous() + self.hc(
                x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous())
        else:
            x_out = 1 / 2 * (self.cw(x.permute(0, 2, 1, 3).contiguous()).permute(0, 2, 1, 3).contiguous() + self.hc(
                x.permute(0, 3, 2, 1).contiguous()).permute(0, 3, 2, 1).contiguous())
        return x_out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class CLSTM_cell(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_chans, num_features, filter_size):
        super(CLSTM_cell, self).__init__()
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        # self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
        #                       self.padding)
        # self.fuse_out=nn.Conv2d(self.num_features*4, self.num_features, 1)

        self.spatial_fuse = nn.Conv2d(self.input_chans + self.num_features, self.num_features, self.filter_size, 1,
                              self.padding)
        # self.fuse = nn.Conv2d(self.num_features*3, self.num_features, self.filter_size, 1, self.padding)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        # A=self.conv(combined)
        A = torch.sigmoid(self.spatial_fuse(combined))
        next_h = A*input+(1-A)*hidden
        # next_h = self.fuse(torch.cat([input, next_h, hidden], dim=1))
        # (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)
        # i = torch.sigmoid(ai)
        # f = torch.sigmoid(af)
        # o = torch.sigmoid(ao)
        # g = torch.tanh(ag)

        # next_c = f * c + i * g
        # next_h = f * hidden + i * g

        # i = ai
        # f = af
        # o = ao
        # g = ag

        # next_c = (f + c) * (i + g)
        # next_c = (f * c) + (i * g)
        # next_h = o * next_h
        # next_h=self.fuse_out(A)
        return next_h, A

    def init_hidden(self, batch_size, shape):
        return torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda()


class CLSTM_cell_original(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_chans, num_features, filter_size):
        super(CLSTM_cell_original, self).__init__()
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.conv = nn.Conv2d(self.input_chans + self.num_features, 4 * self.num_features, self.filter_size, 1,
                              self.padding)

    def forward(self, input, hidden_state):
        hidden, c = hidden_state
        # print(hidden.shape, c.shape)
        combined = torch.cat((input, hidden), 1)
        A = self.conv(combined)
        (ai, af, ao, ag) = torch.split(A, self.num_features, dim=1)
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)

        next_c = f * c + i * g
        next_h = o * torch.tanh(next_c)
        return next_h, next_c

    def init_hidden(self, batch_size, shape):
        return (torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda(),
                torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda())

class CLSTM_cell_FFT_Fuse(nn.Module):
    """Initialize a basic Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the hidden states h and c()
      filter_size: int that is the height and width of the filters
      num_features: int thats the num of channels of the states, like hidden_size

    """

    def __init__(self, input_chans, num_features, filter_size):
        super(CLSTM_cell_FFT_Fuse, self).__init__()
        self.input_chans = input_chans
        self.filter_size = filter_size
        self.num_features = num_features
        self.padding = (filter_size - 1) // 2
        self.real_fuse = nn.Conv2d(self.input_chans + self.num_features, self.num_features, self.filter_size, 1,
                              self.padding)
        self.imag_fuse = nn.Conv2d(self.input_chans + self.num_features, self.num_features, self.filter_size, 1,
                              self.padding)

    def forward(self, input, hidden):
        b, c, h, w = input.shape
        # 对 h 和 w 维度进行 2D 实数 FFT
        input_fft = torch.fft.rfft2(input, dim=(-2, -1))
        # 获取振幅 (Amplitude)
        input_real = input_fft.real  # 或者 amp = rfft_result.abs()
        # 获取相位 (Phase)
        input_imag = input_fft.imag

        # 对 h 和 w 维度进行 2D 实数 FFT
        hidden_fft = torch.fft.rfft2(hidden, dim=(-2, -1))
        # 获取振幅 (Amplitude)
        hidden_real = hidden_fft.real  # 或者 amp = rfft_result.abs()
        # 获取相位 (Phase)
        hidden_imag = hidden_fft.imag

        fused_real = self.real_fuse(torch.cat([input_real, hidden_real], dim=1))
        fused_imag = self.imag_fuse(torch.cat([input_imag, hidden_imag], dim=1))

        # 将实部和虚部合成为复数张量
        complex_tensor = torch.complex(fused_real, fused_imag)

        # 进行 2D 逆变换，恢复原始的图像特征
        next_h = torch.fft.irfft2(complex_tensor, s=(h, w), dim=(-2, -1))

        return next_h

    def init_hidden(self, batch_size, shape):
        return torch.zeros(batch_size, self.num_features, shape[0], shape[1]).cuda()


class res_block(nn.Module):
    def __init__(self, ch_in):
        super(res_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True))

    def forward(self, x):
        y = x + self.conv(x)
        return y + self.conv1(y)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.ta = TripletAttention()
        self.res_block = res_block(ch_out)

    def forward(self, x):
        return self.ta(self.res_block(self.conv(x)))

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )

    def forward(self, x):
        return self.main(x) + x


class ResBlockFFT(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlockFFT, self).__init__()
        self.main = nn.Sequential(
            BasicConv(in_channel, out_channel, kernel_size=3, stride=1, relu=True),
            BasicConv(out_channel, out_channel, kernel_size=3, stride=1, relu=False)
        )
        self.main_fft = fft_bench_complex_mlp(out_channel, norm='backward')

    def forward(self, x):
        # b, c, h, w = x.shape
        return self.main(x) + x + self.main_fft(x)


class conv_block_i(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_i, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.ta = TripletAttention()
        self.res_block = res_block(ch_out)

    def forward(self, x):
        return self.ta(self.res_block(self.conv(x)))


class conv_block1(nn.Module):
    def __init__(self, ch_in, ch_out, kernelsize=3):
        super(conv_block1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=kernelsize, stride=1, padding=int((kernelsize - 1) / 2), bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class conv_block_d(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.ta = TripletAttention()
        self.res_block = res_block(ch_out)

    def forward(self, x):
        return self.ta(self.res_block(self.conv(x)))


class conv_block_u(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block_u, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.ta = TripletAttention()
        self.res_block = res_block(ch_out)

    def forward(self, x):
        return self.ta(self.res_block(self.conv(x)))


class SqueezeAttentionBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlock, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = BasicConv(ch_in, ch_out, 3, 1)
        self.conv_atten = CLSTM_cell(ch_in, ch_out, 5)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden_state):
        x_res = self.conv(x)
        y = self.avg_pool(x)
        h, c = self.conv_atten(y, hidden_state)
        y = self.upsample(h)
        return self.sigmoid((y * x_res) + y) * 2 - 1, h, c
        # return (y * x_res) + y, h, c


class SqueezeAttentionBlockNoAct(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlockNoAct, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.avg_pool = nn.Identity()
        self.conv = BasicConv(ch_in, ch_out, 1, 1, relu=False)
        self.conv_atten = CLSTM_cell(ch_in, ch_out, 5)
        # self.conv_atten = CLSTM_cell_FFT_Fuse(ch_in, ch_out, 5)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.upsample = nn.Identity()
        self.out=BasicConv(ch_out, ch_out, 3, 1, relu=False)

    def forward(self, x, h):
        x_res = self.conv(x)
        y = self.avg_pool(x)
        h, A = self.conv_atten(y, h)
        y = self.upsample(h)
        # y = F.interpolate(h, (x.shape[-2], x.shape[-1]), mode='bilinear')
        y=self.out(y)+x_res
        return y, h, A
        # return (y * x_res) + y, h, c

class SqueezeAttentionBlockConvLSTM(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(SqueezeAttentionBlockConvLSTM, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.avg_pool = nn.Identity()
        self.conv = BasicConv(ch_in, ch_out, 1, 1, relu=False)
        self.conv_atten = CLSTM_cell_original(ch_in, ch_out, 5)
        # self.conv_atten = CLSTM_cell_FFT_Fuse(ch_in, ch_out, 5)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        # self.upsample = nn.Identity()
        self.out=BasicConv(ch_out, ch_out, 3, 1, relu=False)

    def forward(self, x, h, c):
        x_res = self.conv(x)
        y = self.avg_pool(x)
        h, c = self.conv_atten(y, (h, c))
        y = self.upsample(h)
        # y = F.interpolate(h, (x.shape[-2], x.shape[-1]), mode='bilinear')
        y=self.out(y)+x_res
        return y, h, c
        # return (y * x_res) + y, h, c


def gaussian(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return (gauss / gauss.sum()).cuda()


def gen_gaussian_kernel(window_size, sigma):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(1, 1, window_size, window_size).contiguous())
    return window


def shape2coordinate(shape=(3,3), device='cuda', normalize_range=(0.,1.)):
    h, w=shape
    x=torch.arange(0, h, device=device)
    y=torch.arange(0, w, device=device)

    x, y=torch.meshgrid(x, y)

    min, max=normalize_range
    x=x/(h-1)*(max-min)+min
    y=y/(w-1)*(max-min)+min
    cord=torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)

    return cord



def shape2polar_coordinate(shape=(3,3), device='cuda'):
    h, w=shape
    x=torch.arange(0, h, device=device)
    y=torch.arange(0, w, device=device)

    x, y=torch.meshgrid(x, y)

    min=-1
    max=1
    x=x/(h-1)*(max-min)+min
    y=y/(w-1)*(max-min)+min
    cord=x+1j*y


    r=torch.abs(cord)/np.sqrt(2)
    theta=torch.angle(cord)
    theta_code=torch.cat([(torch.cos(theta).unsqueeze(-1)+1)/2, (torch.sin(theta).unsqueeze(-1)+1)/2], dim=-1)

    cord=torch.cat([r.unsqueeze(-1), theta_code], dim=-1)

    return cord


class Sine(nn.Module):
    def __init__(self, w0 = 1.):
        super().__init__()
        self.w0 = w0
    def forward(self, x):
        return torch.sin(self.w0 * x)

class KernelINR_Polar(nn.Module):
    def __init__(self, hidden_dim=64, w=1.):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(3, hidden_dim),
            Sine(w),
            nn.Linear(hidden_dim, hidden_dim),
            Sine(w),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, cord):
        k=self.layers(cord).squeeze(-1)
        return k

class SizeGroupINRConvPolar(nn.Module):
    def __init__(self, max_kernel_size=17, num_ch=3, basis_num=5, w_max=7., w_min=1., w_list=None, learnable_freq=False):
        super(SizeGroupINRConvPolar, self).__init__()
        if w_list is None:
            w_list=[w_min+(w_max-w_min)/(basis_num-1)*i for i in range(basis_num)]
            if learnable_freq:
                newwlist=[torch.nn.Parameter(torch.scalar_tensor(w_list[i], dtype=torch.float32)) for i in range(basis_num)]
                w_list=newwlist
        assert len(w_list)==basis_num
        self.w_list=w_list
        self.num_ch=num_ch
        self.kernelINR_list=nn.ModuleList(KernelINR_Polar(hidden_dim=64, w=w_list[i]) for i in range(basis_num))

        self.basis_num=basis_num
        self.max_kernel_size=max_kernel_size
        self.kernel_sizes=[(2*(i+1)+1, 2*(i+1)+1) for i in range(max_kernel_size//2)]
        # self.kernel_sizes = [(max_kernel_size, max_kernel_size) for i in range(max_kernel_size // 2)]

        self.padding=max_kernel_size//2
        self.group_num=len(self.kernel_sizes)

        masks=[] # [1x1, 3x3, ..., 15x15, ...]

        cords=[] # [1x1xc, 3x3xc, ..., 15x15xc, ...]

        empty=torch.zeros(self.basis_num, 1, 1, 1, device='cuda')
        # delta[0, :,max_kernel_size//2, max_kernel_size//2]=1
        delta = torch.ones(self.basis_num, 1, 1, 1, device='cuda')

        self.delta=delta
        self.empty=empty

        for siz in self.kernel_sizes:
            mask = torch.ones(siz, device='cuda', dtype=torch.float32) * (3 ** 2) / (siz[0] * siz[1])
            # mask=torch.ones(siz, device='cuda', dtype=torch.float32)*(max_kernel_size**2)/(siz[0]*siz[1])
            # mask = torch.ones(siz, device='cuda', dtype=torch.float32)
            masks.append(mask)

            cord=shape2polar_coordinate(shape=siz, device='cuda')
            cords.append(cord)

        self.masks=masks
        self.cords=cords


    def forward(self, x):
        b, c, h, w = x.shape
        kernels = []
        maps = []
        for k in range(self.group_num) :
            kernels_g = []
            for i in range(self.basis_num):
                kernel = self.kernelINR_list[i](self.cords[k])  # h w
                kernel = kernel*self.masks[k]
                kernels_g.append(kernel.unsqueeze(0))
            kernels_g=torch.cat(kernels_g, dim=0)  # m h w
            maps_g = F.conv2d(x, kernels_g.repeat(self.num_ch, 1, 1).unsqueeze(1),
                              padding=self.kernel_sizes[k][0]//2,
                              groups=self.num_ch)  # b 3*m h w
            maps.append(maps_g.unsqueeze(1))
            kernels.append(kernels_g)
        maps=torch.cat(maps, dim=1)  # b gn 3*m h w

        null_map=torch.zeros(b, 1, self.num_ch*self.basis_num, h, w, device='cuda')

        maps=torch.cat([null_map, maps], dim=1)  # b gn+1 3*m h w

        return maps

class GaussianBlurLayer(nn.Module):
    def __init__(self, num_kernels=21, max_kernel_size=21, mode='TG', channels=3):
        super(GaussianBlurLayer, self).__init__()
        self.channels = channels
        kernel_size = max_kernel_size
        weight = torch.zeros(num_kernels + 1, 1, max_kernel_size, max_kernel_size)
        for i in range(num_kernels):
            weight[i+1] = (gen_gaussian_kernel(kernel_size, sigma=0.25 * (i+1)).cuda()).squeeze(0)
            if i >= 2 and i % 2 == 0 and kernel_size < max_kernel_size:
                kernel_size += 2
        pad = int((max_kernel_size - 1) / 2)
        weight[0] = (F.pad(torch.FloatTensor([[[[1.]]]]).cuda(),
                           [pad, pad, pad, pad])).squeeze(0)

        # kernel=weight
        kernel = torch.tile(weight, dims=(3,1,1,1)).cuda()
        if mode == 'TG':
            self.weight = kernel
            self.weight.requires_grad = True
        elif mode == 'TR':
            self.weight = nn.Parameter(data=torch.randn(num_kernels * 3, 1, max_kernel_size, max_kernel_size),
                                       requires_grad=True)
        else:
            self.weight = kernel
            self.weight.requires_grad = False
        self.padding = int((max_kernel_size - 1) / 2)

    def __call__(self, x, groups=None):
        if groups is None:
            x = F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)
        else:
            x = F.conv2d(x, self.weight, padding=self.padding, groups=groups)
        return x


class GaussianBlurWithOcclusionLayer(nn.Module):
    def __init__(self, num_kernels=21, max_kernel_size=21, mode='TG', channels=3):
        super(GaussianBlurWithOcclusionLayer, self).__init__()
        self.channels = channels
        kernel_size = max_kernel_size
        weight = torch.zeros(num_kernels + 1, 1, max_kernel_size, max_kernel_size)
        for i in range(num_kernels):
            weight[i] = (gen_gaussian_kernel(kernel_size, sigma=0.25 * (i+1)).cuda()).squeeze(0)
            if i >= 2 and i % 2 == 0 and kernel_size < max_kernel_size:
                kernel_size += 2
        pad = int((max_kernel_size - 1) / 2)
        weight[0] = (F.pad(torch.FloatTensor([[[[1.]]]]).cuda(),
                           [pad, pad, pad, pad])).squeeze(0)

        # kernel=weight
        kernel = torch.tile(weight, dims=(3,1,1,1)).cuda()
        if mode == 'TG':
            self.weight = kernel
            self.weight.requires_grad = True
        elif mode == 'TR':
            self.weight = nn.Parameter(data=torch.randn(num_kernels * 3, 1, max_kernel_size, max_kernel_size),
                                       requires_grad=True)
        else:
            self.weight = kernel
            self.weight.requires_grad = False
        self.padding = int((max_kernel_size - 1) / 2)

    def __call__(self, x, mask):
        # temp = self.weight.detach().unsqueeze(1).cpu().numpy()
        # for i in range(len(temp)//3):
        #     cv2.imwrite('kernels1/TG/' + str(i) + '.png', temp[i, 0, 0] * 255. * 1)
        c=x.shape[1]
        x_masked=einops.rearrange(x, 'b c h w -> b c 1 h w')*einops.rearrange(mask, 'b (c d) h w -> b c d h w', c=c)
        x_masked=einops.rearrange(x_masked, 'b c d h w -> b (c d) h w')
        x = F.conv2d(x_masked, self.weight, padding=self.padding, groups=x_masked.shape[1])
        return x



class SumLayer(nn.Module):
    def __init__(self, num_kernels=21, trainable=False):
        super(SumLayer, self).__init__()
        self.conv = nn.Conv2d(2 * (num_kernels + 1) * 3, 3, 1)

    def forward(self, x):
        return self.conv(x)


class MultiplyLayer1(nn.Module):
    def __init__(self):
        super(MultiplyLayer1, self).__init__()

    def forward(self, x, y):
        return x * torch.cat([y, y, y], dim=1)


class MultiplyLayer(nn.Module):
    def __init__(self):
        super(MultiplyLayer, self).__init__()
        self.ml = MultiplyLayer1()

    def forward(self, x, y):
        b, c, h, w = x.shape
        b1, c1, h1, w1 = y.shape
        return torch.cat([self.ml(x[:, :c // 2], y[:, :c1 // 2]), self.ml(x[:, c // 2:], y[:, c1 // 2:])], dim=1)


class SobelEntropyLayer(nn.Module):
    def __init__(self, window_size=9):
        super(SobelEntropyLayer, self).__init__()
        self.window_size = window_size
        
        # 定义Sobel算子核
        self.register_buffer('sobel_x_kernel', 
                            torch.FloatTensor([[1, 0, -1], 
                                              [2, 0, -2], 
                                              [1, 0, -1]]).view(1, 1, 3, 3))
        self.register_buffer('sobel_y_kernel', 
                            torch.FloatTensor([[1, 2, 1], 
                                              [0, 0, 0], 
                                              [-1, -2, -1]]).view(1, 1, 3, 3))
        
        # 创建用于计算邻域和的卷积核
        self.register_buffer('sum_kernel', 
                            torch.ones(1, 1, window_size, window_size))

    def forward(self, x):
        """
        计算输入图像的Sobel熵（边缘强度邻域和）
        
        Args:
            x (Tensor): 输入张量，形状为 (B, C, H, W)
            
        Returns:
            Tensor: Sobel熵图，形状为 (B, C, H, W)
        """
        # 如果输入是多通道图像，转换为灰度图
        if x.shape[1] == 3:
            # 使用标准权重转换为灰度图: 0.299*R + 0.587*G + 0.114*B
            gray_weights = torch.tensor([0.299, 0.587, 0.114], 
                                      device=x.device).view(1, 3, 1, 1)
            x_gray = torch.sum(x * gray_weights, dim=1, keepdim=True)
        else:
            x_gray = x
            
        # 计算梯度
        grad_x = F.conv2d(x_gray, self.sobel_x_kernel, padding=1)
        grad_y = F.conv2d(x_gray, self.sobel_y_kernel, padding=1)
        
        # 计算梯度幅值
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        # 计算邻域和作为"熵"
        sobel_entropy = F.conv2d(magnitude, self.sum_kernel, 
                                padding=self.window_size // 2)
        
        return sobel_entropy

# ... existing code ...

if __name__ == '__main__':
    ml = MultiplyLayer().cuda()
