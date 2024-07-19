import os
import torch
import torchvision
import random
import numpy as np
import cv2

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.gif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img*(min_max[1] - min_max[0]) + min_max[0]
    return img


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
def transform_augment(img_list, split='val', min_max=(0, 1)):    
    imgs = [totensor(img) for img in img_list]
    # if split == 'train':
    #     imgs = torch.stack(imgs, 0)
    #     imgs = hflip(imgs)
    #     imgs = torch.unbind(imgs, dim=0)
    ret_img = [(img - img.min()) / (img.max() - img.min()+1e-4) for img in imgs]
    return ret_img

def rgb2ihs(rgb_torch):
    # 计算公式来自https://github.com/hanna-xu/DDcGAN/blob/master/PET-MRI%20image%20fusion/rgb2ihs.m
    r = rgb_torch[:,0,:,:]
    g = rgb_torch[:,1,:,:]
    b = rgb_torch[:,2,:,:]

    I = 1 / np.sqrt(3) * r + 1 / np.sqrt(3) * g + 1 / np.sqrt(3) * b
    H = 1 / np.sqrt(6) * r + 1 / np.sqrt(6) * g - 2 / np.sqrt(6) * b
    S = 1 / np.sqrt(2) * r - 1 / np.sqrt(2) * g
    ihs_torch = torch.stack([I, H, S],1)
    return ihs_torch

def ihs2rgb(ihs_torch):
    # 计算公式来自https://github.com/hanna-xu/DDcGAN/blob/master/PET-MRI%20image%20fusion/ihs2rgb.m
    I = ihs_torch[:,0,:,:]
    V1 = ihs_torch[:,1,:,:]
    V2 = ihs_torch[:,2,:,:]

    r = 1 / np.sqrt(3) * I + 1 / np.sqrt(6) * V1 + 1 / np.sqrt(2) * V2
    g = 1 / np.sqrt(3) * I + 1 / np.sqrt(6) * V1 - 1 / np.sqrt(2) * V2
    b = 1 / np.sqrt(3) * I - 2 / np.sqrt(6) * V1
    rgb_torch = torch.stack([r, g, b],1)
    return rgb_torch


def rgb2YCbCr(rgb_torch):
    # 计算公式来自https://github.com/hanna-xu/DDcGAN/blob/master/PET-MRI%20image%20fusion/rgb2ihs.m
    r = rgb_torch[:,0,:,:]
    g = rgb_torch[:,1,:,:]
    b = rgb_torch[:,2,:,:]

    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = -0.169 * r - 0.331 * g + 0.500 * b + 128
    Cr = 0.500 * r - 0.419 * g - 0.081 * b + 128
    ihs_torch = torch.stack([Y, Cb, Cr],1)
    return ihs_torch

def YCbCr2rgb(ihs_torch):
    # 计算公式来自https://github.com/hanna-xu/DDcGAN/blob/master/PET-MRI%20image%20fusion/ihs2rgb.m
    Y = ihs_torch[:,0,:,:]
    Cb = ihs_torch[:,1,:,:]
    Cr = ihs_torch[:,2,:,:]

    r = Y + 1.402 * (Cr-128)
    g= Y - 0.34414 * (Cb-128) - 0.71414 * (Cr-128)
    b= Y + 1.772 * (Cb-128)
    rgb_torch = torch.stack([r, g, b],1)
    return rgb_torch



def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    # tensor = (tensor - min_max[0]) / \
    #     (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.detach().numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)
    # cv2.imwrite(img_path, img)


import os
 
# PNG --> png
root_path = r"E:\newpaper_data\dataset\T2-T2Flair\1\T2"
files = os.listdir(root_path)
for filename in files:
    file_wo_ext, file_ext = os.path.splitext(filename) #
    if file_ext == ".PNG":
        newfile = os.path.join(root_path, file_wo_ext + ".png")
        os.rename(filename, newfile)
