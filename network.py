import math

from layers import *
import torch
from time import time
import torch.nn.functional as F
import cv2
from torchvision import transforms
import numpy as np


class EBlock(nn.Module):
    def __init__(self, out_channel, num_res=8):
        super(EBlock, self).__init__()

        layers = [ResBlock(out_channel, out_channel) for _ in range(num_res)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel, num_res=8):
        super(DBlock, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
class UNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=32, num_res=2):
        super(UNet, self).__init__()
        self.Encoder = nn.ModuleList([
            EBlock(base_ch, num_res),
            EBlock(base_ch * 2, num_res),
            EBlock(base_ch * 4, num_res),
        ])

        self.Decoder = nn.ModuleList([
            DBlock(base_ch*2, num_res),
            DBlock(base_ch, num_res)
        ])

        self.feat_extract = nn.ModuleList([
            BasicConv(in_ch, base_ch, kernel_size=3, relu=True, stride=1),
            BasicConv(base_ch * 1, base_ch * 2, kernel_size=3, relu=True, stride=2),
            BasicConv(base_ch * 2, base_ch * 4, kernel_size=3, relu=True, stride=2),
            BasicConv(base_ch * 4, base_ch * 2, kernel_size=3, relu=True, stride=1),
            BasicConv(base_ch * 2, base_ch * 1, kernel_size=3, relu=True, stride=1)
        ])

        self.up1=BasicConv(base_ch * 4, base_ch * 2, kernel_size=4, relu=True, stride=2, transpose=True)
        self.up2=BasicConv(base_ch * 2, base_ch * 1, kernel_size=4, relu=True, stride=2, transpose=True)

    def forward(self, x):
        '''Feature Extract 0'''
        x_ = self.feat_extract[0](x)
        res1 = self.Encoder[0](x_)

        '''Down Sample 1'''
        z = self.feat_extract[1](res1)
        res2 = self.Encoder[1](z)

        '''Down Sample 2'''
        z = self.feat_extract[2](res2)
        res3 = self.Encoder[2](z)

        '''Up Sample 2'''
        z=self.up1(res3)
        z = self.feat_extract[3](torch.cat([z, res2], dim=1))
        z = self.Decoder[0](z)

        '''Up Sample 1'''
        z=self.up2(z)
        z = self.feat_extract[4](torch.cat([z, res1], dim=1))
        z = self.Decoder[1](z)

        return z


class GKMNetNewModel(nn.Module):
    def __init__(self, num_res=2, base_ch=32, apu_dim=32):
        super(type(self), self).__init__()
        super().__init__()
        '''backbone'''
        self.unet=UNet(3, base_ch=base_ch, num_res=num_res)
        '''APU'''
        self.APU_forward = SqueezeAttentionBlockNoAct(base_ch, apu_dim)
        '''summation'''
        self.SumLayer = nn.Conv2d(apu_dim, 3, kernel_size=1, bias=False)


    def forward(self, x):
        x1 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x2 = F.interpolate(x, scale_factor=0.25, mode='bilinear')

        input_blurs = [x2, x1, x]
        fs = [self.unet(x) for x in input_blurs]

        h = self.APU_forward.conv_atten.init_hidden(
            input_blurs[0].shape[0],
            (input_blurs[0].shape[-2]//2, input_blurs[0].shape[-1]//2)
        )
        betas_forward=[]
        As = []
        for f in fs:
            beta, h, A = self.APU_forward(f, h)
            betas_forward.append(beta)
            As.append(torch.mean(A, dim=1, keepdim=True))
            h = F.interpolate(h, scale_factor=2,
                              mode='bilinear')

        outs=[]
        betas=[]
        for beta_f, x in zip(betas_forward, input_blurs):
            beta = beta_f
            betas.append(beta)
            out = x + self.SumLayer(beta)
            outs.append(out.clip(0,1))

        # x_ = x.permute(2,3,1,0).float().squeeze().detach().cpu().numpy()
        # A0 = As[0].float().squeeze().detach().cpu().numpy()
        # A1 = As[1].float().squeeze().detach().cpu().numpy()
        # A2 = As[2].float().squeeze().detach().cpu().numpy()

        return outs, time(), betas

    def coefficient_insert(self, beta, x):
        # only the finest scale matters
        pass


if __name__ == '__main__':
    torch.cuda.set_device(0)
    net = GKMNetNewModel(num_res=10, base_ch=32, apu_dim=32).cuda()
    total_samples = 20
    batch_size = 1
    # batch_size = 20
    input = torch.rand(batch_size, 3, 1280, 720).cuda()
    depth = torch.rand(batch_size, 1, 1280, 720).cuda()
    from thop import profile
    from tqdm import tqdm
    flops, params = profile(net, [input])
    total = sum([param.nelement() for param in net.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))
    print(f"FLOPs:{flops / 1e9:.2f}G")
    total_time = 0
    iters = total_samples // batch_size
    for i in tqdm(range(iters)):
        # input = torch.rand(1, 1, 256, 256).cuda()
        torch.cuda.synchronize()
        start = time()
        with torch.no_grad():
            out = net(input)
        torch.cuda.synchronize()
        # if i>burn:
        #     total_time+=time.time()-start
        total_time += time() - start
    print(total_time / total_samples)