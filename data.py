import os
import tabnanny

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import cv2
import time
import torch
import torchvision.transforms.functional as F
import random
from tqdm import tqdm

def normalize_depth_0_1(depth_array):
    """
    对深度图进行归一化处理: (d-median(d))/(mean(abs(d-median(d))))
    """
    # 计算中位数
    min_d = np.min(depth_array)
    
    # 计算 (d-median(d))
    diff = depth_array - min_d
    
    normalized_depth = diff / (np.max(depth_array) - np.min(depth_array))

    return normalized_depth

# 自定义随机缩放和裁剪类
class RandomRescaleCrop:
    def __init__(self, scale_range=(0.5, 1.5), crop_size=(224, 224)):
        self.scale_range = scale_range
        self.crop_size = crop_size

    def __call__(self, img1, img2):
        scale = random.uniform(*self.scale_range)
        width, height = img1.size
        new_width, new_height = int(width * scale), int(height * scale)

        img1 = img1.resize((new_width, new_height), Image.BILINEAR)
        img2 = img2.resize((new_width, new_height), Image.BILINEAR)

        left = random.randint(0, new_width - self.crop_size[0])
        top = random.randint(0, new_height - self.crop_size[1])
        right = left + self.crop_size[0]
        bottom = top + self.crop_size[1]

        img1 = img1.crop((left, top, right, bottom))
        img2 = img2.crop((left, top, right, bottom))

        return img1, img2


class RotateAndCrop:
    def __init__(self, crop_size, angle_range=(0, 360)):
        """
        :param crop_size: 最终裁剪的图像大小，例如 (224, 224)
        :param angle_range: 旋转角度范围，例如 (0, 360)
        """
        self.crop_size = crop_size
        self.angle_range = angle_range

    def rotate_and_crop(self, image, mask, angle):
        # 随机旋转图像和对应的掩码
        rotated_image = transforms.functional.rotate(image, angle, expand=True)
        rotated_mask = transforms.functional.rotate(mask, angle, expand=True)

        # 将mask转换为numpy数组
        rotated_mask_np = np.array(rotated_mask)

        # 黑色区域是mask中值为0的部分
        coords = np.column_stack(np.where(rotated_mask_np != 0))

        # 找到图像中非黑色区域的边界
        top_left = coords.min(axis=0)
        bottom_right = coords.max(axis=0)

        # 裁剪到非黑色区域
        cropped_image = rotated_image.crop((*top_left[::-1], *bottom_right[::-1]))

        return cropped_image

    def random_crop(self, image1, image2):
        """
        在裁剪后的图片上使用相同的参数随机裁剪出一个固定大小的区域
        """
        image_width, image_height = image1.size
        crop_width, crop_height = self.crop_size

        # 随机选择裁剪起点，确保裁剪区域在图像内
        left = np.random.randint(0, image_width - crop_width + 1)
        top = np.random.randint(0, image_height - crop_height + 1)

        # 使用相同的起点对两张图片进行裁剪
        cropped_image1 = image1.crop((left, top, left + crop_width, top + crop_height))
        cropped_image2 = image2.crop((left, top, left + crop_width, top + crop_height))

        return cropped_image1, cropped_image2

    def __call__(self, image1, image2):
        # 在指定范围内随机选择一个角度进行旋转
        angle = np.random.uniform(*self.angle_range)

        # 创建一个全白的mask图像，用于确定旋转后变黑的区域
        mask = Image.new("L", image1.size, color=255)

        # 对第一张图片旋转并裁剪掉因为旋转变黑的区域
        cropped_image1 = self.rotate_and_crop(image1, mask, angle)
        # 对第二张图片进行相同的旋转和裁剪操作
        cropped_image2 = self.rotate_and_crop(image2, mask, angle)

        # 从裁剪后的图像中使用相同的参数随机裁剪出固定大小
        final_cropped_image1, final_cropped_image2 = self.random_crop(cropped_image1, cropped_image2)

        return final_cropped_image1, final_cropped_image2



# 自定义随机翻转类
class RandomFlip:
    def __call__(self, img1, img2):
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_TOP_BOTTOM)
            img2 = img2.transpose(Image.FLIP_TOP_BOTTOM)
        return img1, img2


# 定义一个随机选择90度或-90度的函数
class Random90Rotation:
    def __call__(self, img1, img2):
        angle = random.choice([0, 90, 180, 270])  # 随机选择
        # angle = random.choice([-90, 90, 0])  # 随机选择
        return transforms.functional.rotate(img1, angle), transforms.functional.rotate(img2, angle)

class ToTensorTwice:
    def __call__(self, img1, img2):
        to_tensor = transforms.ToTensor()
        return to_tensor(img1), to_tensor(img2)

# 自定义数据增强类
class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img1, img2):
        for transform in self.transform:
            img1, img2 = transform(img1, img2)
        return img1, img2

class Dataset(data.Dataset):
    def __init__(self, img_path, gt_path, depth_path=None, crop_size=(256, 256), noise_aug=None):
        super().__init__()

        self.crop_size = crop_size
        st = time.time()
        print('loading data')
        img_names = sorted(os.listdir(img_path))
        gt_names = sorted(os.listdir(gt_path))
        self.train_img_list = []
        self.train_gt_list = []
        self.train_depth_list = []
        
        for img_name, gt_name in zip(img_names, gt_names):
            img_name = os.path.join(img_path, img_name)
            gt_name = os.path.join(gt_path, gt_name)
            self.train_img_list.append(img_name)
            self.train_gt_list.append(gt_name)
            
        # 如果提供了depth_path，则加载深度图
        if depth_path is not None:
            depth_names = sorted(os.listdir(depth_path))
            for depth_name in depth_names:
                # 修改为.npy文件扩展名
                depth_name_base = os.path.splitext(depth_name)[0]  # 去掉原始扩展名
                depth_name_npy = depth_name_base + '.npy'  # 添加.npy扩展名
                depth_name_npy_full = os.path.join(depth_path, depth_name_npy)
                self.train_depth_list.append(depth_name_npy_full)
        else:
            self.train_depth_list = None
            
        print('loading data finished', time.time() - st)
        self.noise_aug = noise_aug

    def __len__(self):
        return len(self.train_gt_list)

    def normalize_depth(self, depth_array):
        """
        对深度图进行归一化处理: (d-median(d))/(mean(abs(d-median(d))))
        """
        # 计算中位数
        median_d = np.median(depth_array)
        
        # 计算 (d-median(d))
        diff = depth_array - median_d
        
        # 计算 mean(abs(d-median(d)))
        mean_abs_diff = np.mean(np.abs(diff))
        
        # 避免除零错误
        if mean_abs_diff == 0:
            return depth_array
        else:
            # 归一化: (d-median(d))/(mean(abs(d-median(d))))
            normalized_depth = diff / mean_abs_diff
            return normalized_depth
        


    def random_rescale_crop(self, *arrays):
        """
        对多个数组应用相同的随机缩放和裁剪
        """
        # 随机缩放因子
        scale = random.uniform(1.0, 1.0)  # 根据原代码设置为1.0
        h, w = arrays[0].shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        
        # 缩放所有数组
        scaled_arrays = []
        for array in arrays:
            if len(array.shape) == 3:  # 彩色图像
                scaled_array = cv2.resize(array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:  # 深度图（单通道）
                scaled_array = cv2.resize(array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scaled_arrays.append(scaled_array)
        
        # 随机裁剪
        crop_h, crop_w = self.crop_size
        if new_h > crop_h and new_w > crop_w:
            top = random.randint(0, new_h - crop_h)
            left = random.randint(0, new_w - crop_w)
            cropped_arrays = []
            for scaled_array in scaled_arrays:
                cropped_array = scaled_array[top:top+crop_h, left:left+crop_w]
                cropped_arrays.append(cropped_array)
        else:
            # 如果缩放后的图像比裁剪尺寸小，则直接返回
            cropped_arrays = scaled_arrays
            
        return cropped_arrays

    def random_flip(self, *arrays):
        """
        对多个数组应用相同的随机翻转
        """
        flipped_arrays = list(arrays)
        
        # 随机水平翻转
        if random.random() > 0.5:
            flipped_arrays = [np.fliplr(arr) for arr in flipped_arrays]
            
        # 随机垂直翻转
        if random.random() > 0.5:
            flipped_arrays = [np.flipud(arr) for arr in flipped_arrays]
            
        return flipped_arrays

    def random_90_rotation(self, *arrays):
        """
        对多个数组应用相同的随机90度旋转
        """
        angle = random.choice([0, 1, 2, 3])  # 0, 90, 180, 270度对应的旋转次数
        if angle == 0:
            return arrays
            
        rotated_arrays = []
        for array in arrays:
            rotated_array = np.rot90(array, angle, axes=(0, 1))
            rotated_arrays.append(rotated_array)
            
        return rotated_arrays

    def to_tensor(self, array):
        """
        将numpy数组转换为tensor
        """
        if len(array.shape) == 3:  # 彩色图像 (H, W, C)
            tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).float() / 255.0
        else:  # 深度图 (H, W)
            tensor = torch.from_numpy(array.copy()).float().unsqueeze(0)
        return tensor

    def __getitem__(self, idx):
        if self.train_img_list[idx].endswith('0968.png'):
            print("skipping")
            return self[random.randint(0, len(self.train_gt_list)-1)]
        elif self.train_img_list[idx].endswith('1316.png'):
            print("skipping")
            return self[random.randint(0, len(self.train_gt_list)-1)]
        # 加载图像并转换为numpy数组
        clear_img = np.array(Image.open(self.train_gt_list[idx]))
        blurry_img = np.array(Image.open(self.train_img_list[idx]))
        
        # 确保图像为RGB格式
        if len(clear_img.shape) == 2:
            clear_img = np.stack([clear_img, clear_img, clear_img], axis=2)
        if len(blurry_img.shape) == 2:
            blurry_img = np.stack([blurry_img, blurry_img, blurry_img], axis=2)
            
        # 如果有深度图，也加载深度图
        if self.train_depth_list is not None:
            # 修改为加载.npy文件
            depth_img = np.load(self.train_depth_list[idx])
            # 对深度图进行归一化处理
            # depth_img = normalize_depth_0_1(depth_img)
            depth_img = self.normalize_depth(depth_img)
            
            # 应用相同的数据增强到所有图像
            augmented_arrays = self.random_rescale_crop(blurry_img, clear_img, depth_img)
            blurry_img, clear_img, depth_img = augmented_arrays
            
            augmented_arrays = self.random_90_rotation(blurry_img, clear_img, depth_img)
            blurry_img, clear_img, depth_img = augmented_arrays
            
            augmented_arrays = self.random_flip(blurry_img, clear_img, depth_img)
            blurry_img, clear_img, depth_img = augmented_arrays
            
            # 转换为tensor
            blurry_img = self.to_tensor(blurry_img)
            clear_img = self.to_tensor(clear_img)
            depth_img = self.to_tensor(depth_img)
            
            batch = {'img': blurry_img, 'gt': clear_img, 'depth': depth_img}
        else:
            # 应用相同的数据增强到图像
            augmented_arrays = self.random_rescale_crop(blurry_img, clear_img)
            blurry_img, clear_img = augmented_arrays
            
            augmented_arrays = self.random_90_rotation(blurry_img, clear_img)
            blurry_img, clear_img = augmented_arrays
            
            augmented_arrays = self.random_flip(blurry_img, clear_img)
            blurry_img, clear_img = augmented_arrays
            
            # 转换为tensor
            blurry_img = self.to_tensor(blurry_img)
            clear_img = self.to_tensor(clear_img)
            
            batch = {'img': blurry_img, 'gt': clear_img}
            
        return batch



class KeyPatchDataset(Dataset):
    def __init__(self, entropy_img_path, entropy_gt_path, proposal_sample_num = 4, proposal_crop_num_per_sample = 4, eps=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.entropy_img_path = entropy_img_path
        self.entropy_gt_path = entropy_gt_path
        self.proposal_sample_num = proposal_sample_num
        self.proposal_crop_num_per_sample = proposal_crop_num_per_sample

        self.entropy_img_names = sorted(os.listdir(entropy_img_path))
        self.entropy_gt_names = sorted(os.listdir(entropy_gt_path))
        self.train_entropy_img_list = []
        self.train_entropy_gt_list = []

        self.eps = eps
        
        for img_name, gt_name in zip(self.entropy_img_names, self.entropy_gt_names):
            img_name = os.path.join(entropy_img_path, img_name)
            gt_name = os.path.join(entropy_gt_path, gt_name)
            self.train_entropy_img_list.append(img_name)
            self.train_entropy_gt_list.append(gt_name)

        print("start loading")
        self.train_img_samples = []
        self.train_gt_samples = []
        self.train_entropy_img_samples = []
        self.train_entropy_gt_samples = []
        # for i, (img_p, gt_p, ent_img_p, ent_gt_p) in tqdm(enumerate(zip(self.train_img_list, 
        #                                             self.train_gt_list, 
        #                                             self.train_entropy_img_list, 
        #                                             self.train_entropy_gt_list))):
        #     # if not img_p.endswith('1316.png'):
        #     #     continue
        #     # if not img_p.endswith('0968.png'):
        #     #     continue
        #     # if i > 50:
        #     #     break
        #     img = np.array(Image.open(img_p))
        #     gt = np.array(Image.open(gt_p))
        #     ent_img = np.array(Image.open(ent_img_p))
        #     ent_gt = np.array(Image.open(ent_gt_p))
        #     self.train_img_samples.append(img)
        #     self.train_gt_samples.append(gt)
        #     self.train_entropy_img_samples.append(ent_img)
        #     self.train_entropy_gt_samples.append(ent_gt)
        print("loading finished")


    def __getitem__(self, idx):
        # 随机选择 proposal_sample_num 个样本索引
        sample_indices = random.sample(range(len(self.train_img_list)), min(self.proposal_sample_num, len(self.train_img_list)))
        
        patches = []
        entropy_weights = []
        
        # 对每个选中的样本进行处理
        for sample_idx in sample_indices:
            # img = self.train_img_samples[sample_idx]
            # gt = self.train_gt_samples[sample_idx]
            # ent_img = self.train_entropy_img_samples[sample_idx]
            # ent_gt = self.train_entropy_gt_samples[sample_idx]

            img_p = self.train_img_list[sample_idx]
            gt_p = self.train_gt_list[sample_idx]
            ent_img_p = self.train_entropy_img_list[sample_idx]
            ent_gt_p = self.train_entropy_gt_list[sample_idx]

            img = np.array(Image.open(img_p).convert('RGB'))
            gt = np.array(Image.open(gt_p).convert('RGB'))
            ent_img = np.array(Image.open(ent_img_p))
            ent_gt = np.array(Image.open(ent_gt_p))
            
            h, w = img.shape[:2]
            crop_h, crop_w = self.crop_size
            
            # 每个样本生成 proposal_crop_num_per_sample 个随机裁剪
            for _ in range(self.proposal_crop_num_per_sample):
                if h > crop_h and w > crop_w:
                    top = random.randint(0, h - crop_h)
                    left = random.randint(0, w - crop_w)
                else:
                    # 如果图像太小，直接使用整个图像
                    top, left = 0, 0
                    crop_h, crop_w = h, w
                
                # 裁剪所有相关图像
                img_patch = img[top:top+crop_h, left:left+crop_w]
                gt_patch = gt[top:top+crop_h, left:left+crop_w]
                ent_img_patch = ent_img[top:top+crop_h, left:left+crop_w]
                ent_gt_patch = ent_gt[top:top+crop_h, left:left+crop_w]
                # 计算熵图的和作为权重
                ent_gt_mask = np.where((ent_gt_patch/255 > 25/255), 1, 0)  
                ent_img_mask = np.where((ent_img_patch/255 > 15/255), 1, 0)    
                ent_union_mask = ent_img_mask | ent_gt_mask  
                ent_residual = np.abs(ent_img_patch/255 - ent_gt_patch/255)
                ent_mask = np.where(ent_residual > 10/255, 1, 0)     
                ent_img_sum_in_mask = np.sum(ent_mask*(ent_img_patch/255)) / (ent_mask.sum()+1e-8)
                ent_gt_sum_in_mask = np.sum(ent_mask*(ent_gt_patch/255)) / (ent_mask.sum()+1e-8)
                ent_sum_aggregate_in_mask = ent_gt_sum_in_mask - ent_img_sum_in_mask
                # ent_mask = np.where((ent_gt_patch/255 > 25/255) | (ent_img_patch/255 > 25/255), 1, 0)                
                # entropy_sum_aggregate = (np.sum(ent_mask*(ent_gt_patch/255)) - np.sum(ent_mask*(ent_img_patch/255))) / (ent_mask.sum()+1e-8)
                ent_img_sum = np.sum(ent_img_mask*(ent_img_patch/255)) / (ent_img_mask.sum()+1e-8)
                ent_gt_sum = np.sum(ent_gt_mask*(ent_gt_patch/255)) / (ent_gt_mask.sum()+1e-8)
                # entropy_sum_aggregate = ent_gt_sum - ent_img_sum
                entropy_sum_pixle_wise = np.sum(ent_union_mask*(ent_gt_patch/255) - ent_union_mask*(ent_img_patch/255)) / (ent_union_mask.sum()+1e-8)
                residual_patch = np.abs(gt_patch.astype(np.float32) - img_patch.astype(np.float32)).astype(np.uint8)
                # residual_sparsity = np.sum(ent_mask[:,:,np.newaxis].repeat(3,-1)*np.where(np.abs(img_patch/255 - gt_patch/255) < 0.05, 1, 0)) / ent_mask[:,:,np.newaxis].repeat(3,-1).sum()
                residual_mask = np.where((residual_patch/255 > 10/255), 1, 0)
                if residual_mask.sum() == 0:
                    residual_sparsity = 1
                else:
                    residual_sparsity = 1 - np.sum(residual_mask*residual_patch/255) / residual_mask.sum()
                    # residual_sparsity = 1 - np.sum(residual_mask*residual_patch/255) / max(residual_mask.sum(), 0.1*residual_mask.shape[0]*residual_mask.shape[1]*residual_mask.shape[2])
                residual_sparsity = np.clip(residual_sparsity, 0, 1)
                # entropy_metric = entropy_sum_aggregate / (entropy_sum_pixle_wise+1e-8)
                alpha = 1.0
                entropy_metric = alpha * ent_sum_aggregate_in_mask + (1-alpha) * residual_sparsity
                entropy_metric = max(0, entropy_metric)

                if residual_sparsity < 0.85 and entropy_metric < 0.2 and ent_img_sum_in_mask > 0.2 \
                or residual_sparsity < 0.8 and entropy_metric < 0.25 and ent_img_sum_in_mask > 0.15:
                    continue
                # entropy_metric = entropy_sum_aggregate
                entropy_weights.append(entropy_metric)
                # 存储裁剪后的图像对
                patches.append((img_patch, gt_patch, ent_img_patch, ent_gt_patch, residual_patch, residual_sparsity, entropy_metric))        
        
        patches = sorted(patches, key=lambda x: x[-1], reverse=True)
        entropy_weights = sorted(entropy_weights, reverse=True)
    
        # 归一化权重为概率分布
        entropy_weights = np.array(entropy_weights)
        # 添加一个小常数防止所有权重都为0的情况
        entropy_weights = entropy_weights + self.eps
        probability = entropy_weights / np.sum(entropy_weights)
        
        # 根据概率分布选择一个patch
        selected_idx = np.random.choice(len(patches), p=probability)
        selected_img_patch, selected_gt_patch, _, _, _, _, _ = patches[selected_idx]
        
        # 应用数据增强
        augmented_arrays = self.random_90_rotation(selected_img_patch, selected_gt_patch)
        selected_img_patch, selected_gt_patch = augmented_arrays
        
        augmented_arrays = self.random_flip(selected_img_patch, selected_gt_patch)
        selected_img_patch, selected_gt_patch = augmented_arrays
        
        # 转换为tensor
        img_tensor = self.to_tensor(selected_img_patch)
        gt_tensor = self.to_tensor(selected_gt_patch)
        
        batch = {'img': img_tensor, 'gt': gt_tensor}
        return batch
    


class TestDataset_CHUK(data.Dataset):
    def __init__(self,img_path):
        super(type(self), self).__init__()
        st = time.time()
        self.img_names = sorted(os.listdir(img_path))
        self.test_img_list = []
        for img_name in self.img_names:
            img_name = os.path.join(img_path, img_name)
            # self.test_img_list.append(torch.from_numpy(np.array(cv2.imread(img_name)).transpose((2, 0, 1))))
            self.test_img_list.append(img_name)
        print('loading data finished', time.time() - st)

    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, idx):
        # blurry_img = self.test_img_list[idx]
        blurry_img = torch.from_numpy(np.array(cv2.imread(self.test_img_list[idx])).transpose((2, 0, 1)))
        # print(blurry_img.shape)
        _, h, w = blurry_img.shape
        aim_h = int(np.floor(h / 16) * 16)
        aim_w = int(np.floor(w / 16) * 16)
        blurry_img = blurry_img[:, :aim_h, :aim_w] / 255.
        # print(blurry_img.shape)
        batch = {'img': blurry_img}
        return batch



class LFDOFDataset(data.Dataset):
    def __init__(self, img_path, crop_size=(256, 256)):
        super(LFDOFDataset, self).__init__()
        st = time.time()

        self.crop_size=crop_size

        self.img_names, self.gt_names = load_LFDOF_file_list(img_path)
        self.train_img_list = []
        self.train_gt_list = []
        for img_name, gt_name in zip(self.img_names, self.gt_names):
            # self.train_img_list.append(torch.from_numpy(np.array(cv2.imread(img_name)).transpose((2, 0, 1))))
            # self.train_gt_list.append(torch.from_numpy(np.array(cv2.imread(gt_name)).transpose((2, 0, 1))))
            self.train_img_list.append(img_name)
            self.train_gt_list.append(gt_name)
        print('loading data finished', time.time() - st)

    def __len__(self):
        return len(self.train_gt_list)

    def data_augmentation(self, img, crop_left, crop_top, hf, vf, rot):
        img = img[:, crop_top:crop_top + self.crop_size[1], crop_left:crop_left + self.crop_size[0]]
        if hf:
            img = F.hflip(img)
        if vf:
            img = F.vflip(img)
        img = torch.rot90(img, rot, [1, 2])
        return img

    def set_crop_size(self, x):
        self.crop_size=x

    def __getitem__(self, idx):

        # clear_img = self.train_gt_list[idx]
        # blurry_img = self.train_img_list[idx]
        clear_img = torch.from_numpy(np.array(cv2.imread(self.train_gt_list[idx])).transpose((2, 0, 1)))
        blurry_img = torch.from_numpy(np.array(cv2.imread(self.train_img_list[idx])).transpose((2, 0, 1)))
        _, h, w = clear_img.shape
        crop_left = int(np.floor(np.random.uniform(0, w - self.crop_size[0] + 1)))
        crop_top = int(np.floor(np.random.uniform(0, h - self.crop_size[1] + 1)))
        hf = np.random.randint(0, 2)
        vf = np.random.randint(0, 2)
        rot = np.random.randint(0, 4)
        blurry_img = self.data_augmentation(blurry_img, crop_left, crop_top, hf, vf, rot) / 255.
        clear_img = self.data_augmentation(clear_img, crop_left, crop_top, hf, vf, rot) / 255.
        batch = {'img': blurry_img, 'gt': clear_img}
        return batch


class TestDataset(data.Dataset):
    def __init__(self, img_path, gt_path, depth_path=None, noisy=None):
        super(type(self), self).__init__()
        st = time.time()
        self.img_names = sorted(os.listdir(img_path))
        self.gt_names = sorted(os.listdir(gt_path))
        self.test_img_list = []
        self.test_gt_list = []
        self.test_depth_list = []
        
        for img_name, gt_name in zip(self.img_names, self.gt_names):
            img_name = os.path.join(img_path, img_name)
            gt_name = os.path.join(gt_path, gt_name)
            # self.test_img_list.append(torch.from_numpy(np.array(cv2.imread(img_name)).transpose((2, 0, 1))))
            # self.test_gt_list.append(torch.from_numpy(np.array(cv2.imread(gt_name)).transpose((2, 0, 1))))
            self.test_img_list.append(img_name)
            self.test_gt_list.append(gt_name)
            
        # 如果提供了depth_path，则加载深度图
        if depth_path is not None:
            depth_names = sorted(os.listdir(depth_path))
            for depth_name in depth_names:
                depth_name = os.path.join(depth_path, depth_name)
                self.test_depth_list.append(depth_name)
        else:
            self.test_depth_list = None
            
        print('loading data finished', time.time() - st)

        self.noisy = noisy
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.test_img_list)

    def normalize_depth(self, depth_array):
        """
        对深度图进行归一化处理: (d-median(d))/(mean(abs(d-median(d))))
        """
        # 计算中位数
        median_d = np.median(depth_array)
        
        # 计算 (d-median(d))
        diff = depth_array - median_d
        
        # 计算 mean(abs(d-median(d)))
        mean_abs_diff = np.mean(np.abs(diff))
        
        # 避免除零错误
        if mean_abs_diff == 0:
            return depth_array
        else:
            # 归一化: (d-median(d))/(mean(abs(d-median(d))))
            normalized_depth = diff / mean_abs_diff
            return normalized_depth

    def __getitem__(self, idx):
        # clear_img = self.test_gt_list[idx]
        # blurry_img = self.test_img_list[idx]
        clear_img = Image.open(self.test_gt_list[idx]).convert('RGB')
        blurry_img = Image.open(self.test_img_list[idx]).convert('RGB')
        # print(blurry_img.shape)
        w, h = blurry_img.size
        aim_h = int(np.floor(h / 16) * 16)
        aim_w = int(np.floor(w / 16) * 16)
        clear_img = clear_img.crop((0, 0, aim_w, aim_h))
        blurry_img = blurry_img.crop((0, 0, aim_w, aim_h))
        
        if self.noisy:
            blurry_img = blurry_img + torch.randn_like(blurry_img) * self.noisy
        # print(blurry_img.shape)
        
        # 如果有深度图，也加载并处理深度图
        if self.test_depth_list is not None:
            
            # 转换为numpy数组进行归一化处理
            depth_array = np.load(self.test_depth_list[idx])
            depth_array = depth_array[:aim_h, :aim_w]
            depth_array = self.normalize_depth(depth_array)
            # depth_array = normalize_depth_0_1(depth_array)
            depth_img = Image.fromarray(depth_array.astype(np.float32))
            
            # 转换为tensor
            depth_tensor = transforms.ToTensor()(depth_img)
            
            batch = {
                'img': self.transform(blurry_img), 
                'gt': self.transform(clear_img),
                'depth': depth_tensor
            }
        else:
            batch = {
                'img': self.transform(blurry_img), 
                'gt': self.transform(clear_img)
            }
            
        return batch


def load_RTF_file_list(input_path):
    df_img_path_list = []
    gt_img_path_list = []

    df_dirs=sorted(os.listdir(os.path.join(input_path, 'image')))
    print(df_dirs)
    for di in df_dirs:
        df_imgs_name=sorted(os.listdir(os.path.join(input_path, 'image', di)))
        for name in df_imgs_name:
            df_img_path_list.append(os.path.join(input_path, 'image', di, name))
            gt_img_path_list.append(os.path.join(input_path, 'GT','sharp'+name[5:]))
    return df_img_path_list, gt_img_path_list


def load_LFDOF_file_list(path):
    df_img_path_list=[]
    gt_img_path_list=[]

    df_img_dirs=sorted(os.listdir(os.path.join(path, 'input')))

    # print(df_img_dirs)
    for di in df_img_dirs:
        gt_path=os.path.join(path, 'ground_truth', f'{di}.png')
        df_imgs_path_for_gt=os.listdir(os.path.join(path, 'input', di))
        for df_path in df_imgs_path_for_gt:
            df_img_path_list.append(os.path.join(path, 'input', di, df_path))
            gt_img_path_list.append(gt_path)
    return df_img_path_list, gt_img_path_list


class RTFTestDataset(data.Dataset):
    def __init__(self,path):
        super(type(self), self).__init__()
        st = time.time()

        self.img_names, self.gt_names=load_RTF_file_list(path)
        self.test_img_list=[]
        self.test_gt_list = []
        for img_name, gt_name in zip(self.img_names, self.gt_names):
            self.test_img_list.append(torch.from_numpy(np.array(cv2.imread(img_name)).transpose((2, 0, 1))))
            self.test_gt_list.append(torch.from_numpy(np.array(cv2.imread(gt_name)).transpose((2, 0, 1))))
            # self.test_img_list.append(img_name)
            # self.test_gt_list.append(gt_name)
        print('loading data finished', time.time() - st)

    def __len__(self):
        return len(self.test_img_list)

    def __getitem__(self, idx):
        clear_img = self.test_gt_list[idx]
        blurry_img = self.test_img_list[idx]
        # clear_img = torch.from_numpy(np.array(cv2.imread(self.test_gt_list[idx])).transpose((2, 0, 1)))
        # blurry_img = torch.from_numpy(np.array(cv2.imread(self.test_img_list[idx])).transpose((2, 0, 1)))
        # print(blurry_img.shape)
        _, h, w = blurry_img.shape
        aim_h = int(np.floor(h / 16) * 16)
        aim_w = int(np.floor(w / 16) * 16)
        clear_img = clear_img[:, :aim_h, :aim_w] / 255.
        blurry_img = blurry_img[:, :aim_h, :aim_w] / 255.
        # print(blurry_img.shape)
        batch = {'img': blurry_img, 'gt': clear_img}
        return batch




if __name__ == '__main__':
    imglist, gtlist=load_RTF_file_list('/home/lavie/zc/datasets/RTFDataset')