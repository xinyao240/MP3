import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
from network import GKMNetNewModel as Net
import torch
from util import multi_scale_loss
from data import Dataset, TestDataset
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from util import toRed, toBlue, toCyan, toGreen, toYellow, compute_metrics

import cv2
import json
from PIL import Image


ckpt_name='ckpt-DPDD'
# ckpt_name='ckpt-LFDOF'
model_name = 'model'
net=Net(num_res=10, base_ch=32, apu_dim=32).cuda()
net.load_state_dict(torch.load(f'logs/{ckpt_name}/ckpt/{model_name}.pth'))
# evaluate
save_out_dir=f'out/{ckpt_name}/{model_name}/whole'
os.makedirs(save_out_dir, exist_ok=True)
print('start evaluating ...')
net.eval()

data_dir = '/hdd/sda/yaoxin/data'

test_config = {
    'rtf': {
        'img_path': f'{data_dir}/RTFDataset/image/0',
        'gt_path': f'{data_dir}/RTFDataset/GT'
    },
    'dpdd': {
        'img_path': f'{data_dir}/dd_dp_dataset_png/test_c/source',
        'gt_path': f'{data_dir}/dd_dp_dataset_png/test_c/target'
    },
    'realdof': {
        'img_path': f'{data_dir}/RealDOF/source',
        'gt_path': f'{data_dir}/RealDOF/target'
    },
    'CUHK': {
        'img_path': f'{data_dir}/CUHK-DBD-Dataset/CUHK604S_Training',
        'gt_path': f'{data_dir}/CUHK-DBD-Dataset/CUHK604S_Training'
    },
    'lfdof': {
        'img_path': f'{data_dir}/LFDOF_reformat/test_data/input',
        'gt_path': f'{data_dir}/LFDOF_reformat/test_data/ground_truth'
    },
}

test_data_rtf=TestDataset(img_path=test_config['rtf']['img_path'],
                          gt_path=test_config['rtf']['gt_path'])
test_data_dpdd=TestDataset(img_path=test_config['dpdd']['img_path'],
                          gt_path=test_config['dpdd']['gt_path'])
test_data_realdof=TestDataset(img_path=test_config['realdof']['img_path'],
                          gt_path=test_config['realdof']['gt_path'])
test_data_cuhk=TestDataset(img_path=test_config['CUHK']['img_path'],
                          gt_path=test_config['CUHK']['gt_path'])
test_data_lfdof=TestDataset(img_path=test_config['lfdof']['img_path'],
                          gt_path=test_config['lfdof']['gt_path'])

test_loader_dpdd=DataLoader(test_data_dpdd, shuffle=False, batch_size=1)
test_loader_realdof=DataLoader(test_data_realdof, shuffle=False, batch_size=1)
test_loader_rtf=DataLoader(test_data_rtf, shuffle=False, batch_size=1)
test_loader_cuhk=DataLoader(test_data_cuhk, shuffle=False, batch_size=1)  
test_loader_lfdof=DataLoader(test_data_lfdof, shuffle=False, batch_size=1)  

save_img=True

# dpdd
save_out_dataset_dir=os.path.join(save_out_dir, 'dpdd')
os.makedirs(save_out_dataset_dir, exist_ok=True)
f = open(f'{save_out_dataset_dir}/metrics.txt', 'w')

sum_psnr = []
sum_ssim = []
sum_lpips = []
for i, batch in enumerate(tqdm(test_loader_dpdd)):
    gt = batch['gt'].cuda()
    img = batch['img'].cuda()

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            out_result = net(img)[0][-1]
            metrics = compute_metrics(out_result, gt)

    psnr=metrics['psnr']
    ssim=metrics['ssim']
    lpips_val=metrics['lpips']
    sum_ssim.append(metrics['ssim'])
    sum_psnr.append(metrics['psnr'])
    sum_lpips.append(metrics['lpips'])
    img_name=test_data_dpdd.img_names[i]
    save_p = os.path.join(save_out_dataset_dir, test_data_dpdd.img_names[i])
    if save_img:
        Image.fromarray(np.astype(np.clip(metrics['out_numpy'], 0,1)*255, np.uint8)).save(save_p)

    f.write(f'{img_name} psnr/ssim/lpips {psnr}/{ssim}/{lpips_val}\n')


avg_ssim = sum(sum_ssim) / len(sum_ssim)
avg_psnr = sum(sum_psnr) / len(sum_psnr)
avg_lpips = sum(sum_lpips) / len(sum_lpips)
print(f'dpdd avg val ssim:{toBlue(str(avg_ssim)),} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.write(f'dpdd avg val ssim:{toBlue(str(avg_ssim)),} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.close()
torch.cuda.empty_cache()

# realdof
sum_psnr = []
sum_ssim = []
sum_lpips = []
save_out_dataset_dir=os.path.join(save_out_dir, 'realdof')
os.makedirs(save_out_dataset_dir, exist_ok=True)
f = open(f'{save_out_dataset_dir}/metrics.txt', 'w')

for i, batch in enumerate(tqdm(test_loader_realdof)):
    gt = batch['gt'].cuda()
    img = batch['img'].cuda()

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            out_result = net(img)[0][-1]
            metrics = compute_metrics(out_result, gt)

    psnr = metrics['psnr']
    ssim = metrics['ssim']
    lpips_val = metrics['lpips']
    sum_ssim.append(metrics['ssim'])
    sum_psnr.append(metrics['psnr'])
    sum_lpips.append(metrics['lpips'])
    img_name = test_data_realdof.img_names[i]
    save_p = os.path.join(save_out_dataset_dir, test_data_realdof.img_names[i])
    if save_img:
        Image.fromarray(np.astype(np.clip(metrics['out_numpy'], 0,1)*255, np.uint8)).save(save_p)


    f.write(f'{img_name} psnr/ssim/lpips {psnr}/{ssim}/{lpips_val}\n')


avg_ssim = sum(sum_ssim) / len(sum_ssim)
avg_psnr = sum(sum_psnr) / len(sum_psnr)
avg_lpips = sum(sum_lpips) / len(sum_lpips)
print(f'realdof avg val ssim:{toBlue(str(avg_ssim))} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.write(f'realdof avg val ssim:{toBlue(str(avg_ssim))} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.close()

torch.cuda.empty_cache()

# rtf
sum_psnr = []
sum_ssim = []
sum_lpips = []
save_out_dataset_dir=os.path.join(save_out_dir, 'rtf')
os.makedirs(save_out_dataset_dir, exist_ok=True)
f = open(f'{save_out_dataset_dir}/metrics.txt', 'w')

for i, batch in enumerate(tqdm(test_loader_rtf)):
    gt = batch['gt'].cuda()
    img = batch['img'].cuda()

    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            out_result = net(img)[0][-1]
            metrics = compute_metrics(out_result, gt)

    psnr = metrics['psnr']
    ssim = metrics['ssim']
    lpips_val = metrics['lpips']
    sum_ssim.append(metrics['ssim'])
    sum_psnr.append(metrics['psnr'])
    sum_lpips.append(metrics['lpips'])
    img_name = test_data_rtf.img_names[i]
    save_p = os.path.join(save_out_dataset_dir, test_data_rtf.img_names[i])
    if save_img:
        Image.fromarray(np.astype(np.clip(metrics['out_numpy'], 0,1)*255, np.uint8)).save(save_p)


    f.write(f'{img_name} psnr/ssim/lpips {psnr}/{ssim}/{lpips_val}\n')


avg_ssim = sum(sum_ssim) / len(sum_ssim)
avg_psnr = sum(sum_psnr) / len(sum_psnr)
avg_lpips = sum(sum_lpips) / len(sum_lpips)
print(f'rtf avg val ssim:{toBlue(str(avg_ssim))} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.write(f'rtf avg val ssim:{toBlue(str(avg_ssim))} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.close()

# cuhk
sum_psnr = []
sum_ssim = []
sum_lpips = []
save_out_dataset_dir=os.path.join(save_out_dir, 'cuhk')
os.makedirs(save_out_dataset_dir, exist_ok=True)
f = open(f'{save_out_dataset_dir}/metrics.txt', 'w')

for i, batch in enumerate(tqdm(test_loader_cuhk)):
    gt = batch['gt'].cuda()
    img = batch['img'].cuda()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out_result = net(img)[0][-1]
            metrics = compute_metrics(out_result, gt)

    psnr = metrics['psnr']
    ssim = metrics['ssim']
    lpips_val = metrics['lpips']
    sum_ssim.append(metrics['ssim'])
    sum_psnr.append(metrics['psnr'])
    sum_lpips.append(metrics['lpips'])
    img_name = test_data_cuhk.img_names[i]
    save_p = os.path.join(save_out_dataset_dir, test_data_cuhk.img_names[i])
    if save_img:
        Image.fromarray((np.clip(metrics['out_numpy'], 0,1)*255).astype(np.uint8)).save(save_p)


    f.write(f'{img_name} psnr/ssim/lpips {psnr}/{ssim}/{lpips_val}\n')


avg_ssim = sum(sum_ssim) / len(sum_ssim)
avg_psnr = sum(sum_psnr) / len(sum_psnr)
avg_lpips = sum(sum_lpips) / len(sum_lpips)
print(f'cuhk avg val ssim:{toBlue(str(avg_ssim))} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.write(f'cuhk avg val ssim:{toBlue(str(avg_ssim))} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.close()

# lfdof
sum_psnr = []
sum_ssim = []
sum_lpips = []
save_out_dataset_dir=os.path.join(save_out_dir, 'lfdof')
os.makedirs(save_out_dataset_dir, exist_ok=True)
f = open(f'{save_out_dataset_dir}/metrics.txt', 'w')

for i, batch in enumerate(tqdm(test_loader_lfdof)):
    gt = batch['gt'].cuda()
    img = batch['img'].cuda()

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out_result = net(img)[0][-1]
            metrics = compute_metrics(out_result, gt)

    psnr = metrics['psnr']
    ssim = metrics['ssim']
    lpips_val = metrics['lpips']
    sum_ssim.append(metrics['ssim'])
    sum_psnr.append(metrics['psnr'])
    sum_lpips.append(metrics['lpips'])
    img_name = test_data_lfdof.img_names[i]
    save_p = os.path.join(save_out_dataset_dir, test_data_lfdof.img_names[i])
    if save_img:
        Image.fromarray((np.clip(metrics['out_numpy'], 0,1)*255).astype(np.uint8)).save(save_p)


    f.write(f'{img_name} psnr/ssim/lpips {psnr}/{ssim}/{lpips_val}\n')


avg_ssim = sum(sum_ssim) / len(sum_ssim)
avg_psnr = sum(sum_psnr) / len(sum_psnr)
avg_lpips = sum(sum_lpips) / len(sum_lpips)
print(f'lfdof avg val ssim:{toBlue(str(avg_ssim))} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.write(f'lfdof avg val ssim:{toBlue(str(avg_ssim))} psnr:{toGreen(str(avg_psnr))} lpips:{toRed(str(avg_lpips))}')
f.close()