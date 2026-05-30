import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch.nn.functional as F
from tqdm import tqdm

from data import KeyPatchDataset, Dataset, TestDataset, LFDOFDataset
# from log import TensorBoardX
from network import GKMNetNewModel
import torch
torch.cuda.set_device(0)
from utils import *
import torch.nn as nn
# from time import time
import random
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
from util import multi_scale_loss, curricular_ms_loss, curriculum_ms_loss
from util import toRed, toBlue, toCyan, toGreen, toYellow, compute_metrics
import cv2



log10 = np.log(10)
MAX_DIFF = 1
le = 1

mse=nn.MSELoss().cuda()
mae=nn.L1Loss().cuda()

def backward(loss, optimizer):
    optimizer.zero_grad()
    loss['mse'].backward()
    optimizer.step()
    return


def worker_init_fn_seed(worker_id):
    seed = 10
    seed += worker_id
    np.random.seed(seed)


def setup_seed(seed):
    seed = int(seed)
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True



def seed_everything(seed):
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_scheduler(optimizer, steps):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps, eta_min=1e-5)

seed = 777
seed_everything(seed)
crop_size=256
dataset_n='dpdd'
assert dataset_n in ['dpdd', 'lfdof']
data_dir = '/hdd/sda/yaoxin/data'
test_data_rtf=TestDataset(img_path=f'{data_dir}/RTFDataset/image/0',
                          gt_path=f'{data_dir}/RTFDataset/GT')
noisy_aug=None
train_data=KeyPatchDataset(img_path=f'{data_dir}/LFDOF_reformat/train_data/input',
                   gt_path=f'{data_dir}/LFDOF_reformat/train_data/ground_truth', 
                   entropy_img_path=f'{data_dir}/LFDOF_reformat_sobel_map/train_data/input',
                   entropy_gt_path=f'{data_dir}/LFDOF_reformat_sobel_map/train_data/ground_truth',
                   eps=0.1,
                   crop_size=(crop_size, crop_size))

test_data_dpdd=TestDataset(img_path=f'{data_dir}/LFDOF_reformat/test_data/input',
                           gt_path=f'{data_dir}/LFDOF_reformat/test_data/ground_truth')

test_data_realdof=TestDataset(img_path=f'{data_dir}/RealDOF/source',
                              gt_path=f'{data_dir}/RealDOF/target')

batch_size=4
acc_step=1
lr=2e-4
train_loader=DataLoader(train_data, shuffle=True, batch_size=batch_size//acc_step, num_workers=8)
test_loader_dpdd=DataLoader(test_data_dpdd, shuffle=False, batch_size=1)
test_loader_realdof=DataLoader(test_data_realdof, shuffle=False, batch_size=1)
test_loader_rtf=DataLoader(test_data_rtf, shuffle=False, batch_size=1)

num_res=10
base_ch=32
net = GKMNetNewModel(num_res=num_res, base_ch=base_ch).cuda()
net.train()

enable_swa=True
swa_steps=100

model_name=f'model'

exp_dir = f'./logs/{model_name}/{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}'

ckpt_dir = f'{exp_dir}/ckpt'
backup_code_dir = f'{exp_dir}/code'

os.makedirs(ckpt_dir)
os.makedirs(backup_code_dir)

os.system(f"""cp ./*.py '{backup_code_dir}'""")

start_epoch=0
sav_freq=20
eval_freq_dpdd=2
eval_freq_realdof=100
eval_freq_rtf=1


epochs_of_each_stage = [60, 60, 60]
total_stages=len(epochs_of_each_stage)

passed_epochs = 0

for s, epoch_n in enumerate(epochs_of_each_stage):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    opt = SWA(optimizer)
    scheduler = get_scheduler(optimizer, epoch_n*len(train_loader)//acc_step)
    # 添加自动混合精度相关的初始化
    scaler = torch.amp.GradScaler()

    for epoch in range(epoch_n):
        sum_los = []
        sum_psnr = []
        sum_freq_loss = []

        with tqdm(total=len(train_loader),
                  desc=f'epoch{epoch + 1}/{epoch_n}; stage{s + 1}/{total_stages} train', unit='it', ncols=150) as pbar:
            for i, batch in enumerate(train_loader):
                gt = batch['gt'].cuda()
                img = batch['img'].cuda()

                # 使用 autocast 进行 BF16 混合精度训练
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    dbs = net(img)[0]
                # 在FP32精度下计算损失，确保数值稳定性
                with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
                    los = curriculum_ms_loss(dbs, gt, mse_lambda=1., freq_lambda=0.1, lpips_lambda=0.1, ssim_lambda=0.,
                                            l1_lambda=0., char_lambda=0., vgg_lambda=0., stage=s+1, total_stages=total_stages)

                los["loss"].backward()

                pbar.set_postfix(
                    {
                        'bat_loss': toBlue(f'{los["loss"].item():.5f}'),
                        'learning rate': toYellow(f'{optimizer.param_groups[0]["lr"]}'),
                        # 'mse_loss': f'{los["mse"].item():.5f}',
                        # 'freq_loss': f'{los["freq loss"].item():.5f}',
                        'psnr': f'{los["psnr"].item():.5f}'
                    }
                )

                sum_los.append(los["loss"].item())
                sum_psnr.append(los['psnr'].item())
                # sum_freq_loss.append(los['freq loss'].item())

                if ((i + 1) % acc_step) == 0:
                    optimizer.step()  # 反向传播，更新网络参数
                    optimizer.zero_grad()
                    scheduler.step()

                pbar.update(1)

        
        passed_epochs+=1

        if passed_epochs>=epoch_n-swa_steps and enable_swa and s==len(epochs_of_each_stage)-1:
            print('updating swa...')
            opt.update_swa()

        epoch_avg_train_loss = sum(sum_los) / len(sum_los)
        print(f'epoch{epoch + 1}/{epoch_n}; stage{s + 1}/{total_stages} avg train loss:{epoch_avg_train_loss}')
        epoch_avg_train_psnr = sum(sum_psnr) / len(sum_psnr)
        print(f'epoch{epoch + 1}/{epoch_n}; stage{s + 1}/{total_stages} train psnr:{epoch_avg_train_psnr}')
        # epoch_avg_train_freq_loss = sum(sum_freq_loss) / len(sum_freq_loss)
        # print(f'epoch{start_epoch + epoch + 1}/{start_epoch + epoch_n} train freq loss:{epoch_avg_train_freq_loss}')

        f = open(f'{exp_dir}/train_log.txt', 'a')
        msg = f'epoch{epoch + 1}/{epoch_n}; stage{s + 1}/{total_stages} avg train loss:{epoch_avg_train_loss}'
        msg += f'train psnr:{epoch_avg_train_psnr}\n'
        # msg+=f'train freq loss:{epoch_avg_train_freq_loss}\n'
        f.write(msg)
        f.close()

        if (epoch + 1) % sav_freq == 0:
            torch.save(net.state_dict(),
                       f'{ckpt_dir}/epoch{epoch + 1}in{epoch_n}-stage{s + 1}in{total_stages}.pth')
            # torch.save(optimizer.state_dict(),
            #            f'checkpoints/Adam-{model_name}-epoch{start_epoch + epoch + 1}.pth')
            # torch.save(scheduler.state_dict(),
            #            f'checkpoints/scheduler-{model_name}-epoch{start_epoch + epoch + 1}.pth')

        else:
            torch.save(net.state_dict(),
                       f'{ckpt_dir}/{model_name}.pth')
            torch.save(optimizer.state_dict(),
                       f'{ckpt_dir}/Adam-{model_name}.pth')
            torch.save(scheduler.state_dict(),
                       f'{ckpt_dir}/scheduler-{model_name}.pth')


if enable_swa:
    opt.swap_swa_sgd()
    torch.save(net.state_dict(),
               f'{ckpt_dir}/swa_final.pth')
    sum_psnr = []
    sum_ssim = []
    for i, batch in enumerate(tqdm(test_loader_rtf)):
        # for k in batch:
        #     batch[k] = batch[k].to(device)
        gt = batch['gt'].cuda()
        img = batch['img'].cuda()

        with torch.no_grad():
            # 验证阶段也使用混合精度
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                out= net(img)[0]

            metrics = compute_metrics(out[-1], gt)

        sum_ssim.append(metrics['ssim'])
        sum_psnr.append(metrics['psnr'])

    epoch_avg_ssim = sum(sum_ssim) / len(sum_ssim)
    epoch_avg_psnr = sum(sum_psnr) / len(sum_psnr)
    print('rtf epoch{0}/{1} avg val ssim:{2} psnr:{3}'.format(
        start_epoch + epoch + 1, start_epoch + epoch_n, toBlue(str(epoch_avg_ssim)),
        toGreen(str(epoch_avg_psnr))
    ))

    f = open(f'{exp_dir}/train_log.txt', 'a')
    f.write('\nrtf epoch{0}/{1} avg val ssim:{2} psnr:{3}\n'.format(
        start_epoch + epoch + 1, start_epoch + epoch_n, str(epoch_avg_ssim), str(epoch_avg_psnr)
    ))
    f.close()
