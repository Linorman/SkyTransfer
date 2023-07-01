import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import compare_ssim as sk_cpt_ssim

import os
import glob
import random

import torch

if torch.cuda.is_available():
    torch.cuda.current_device()

import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, utils

import json


class PairedDataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot90=False,
            with_random_rot180=False,
            with_random_rot270=False,
            with_random_crop=False,
            with_random_brightness=False,
            with_random_gamma=False,
            with_random_saturation=False
    ):
        self.img_size = img_size
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot90 = with_random_rot90
        self.with_random_rot180 = with_random_rot180
        self.with_random_rot270 = with_random_rot270
        self.with_random_crop = with_random_crop
        self.with_random_brightness = with_random_brightness
        self.with_random_gamma = with_random_gamma
        self.with_random_saturation = with_random_saturation

    def transform(self, img1, img2):

        # resize image and covert to tensor
        img1 = TF.to_pil_image(img1)
        img1 = TF.resize(img1, [self.img_size, self.img_size], interpolation=3)
        img2 = TF.to_pil_image(img2)
        img2 = TF.resize(img2, [self.img_size, self.img_size], interpolation=3)

        if self.with_random_hflip and random.random() > 0.5:
            img1 = TF.hflip(img1)
            img2 = TF.hflip(img2)

        if self.with_random_vflip and random.random() > 0.5:
            img1 = TF.vflip(img1)
            img2 = TF.vflip(img2)

        if self.with_random_rot90 and random.random() > 0.5:
            img1 = TF.rotate(img1, 90)
            img2 = TF.rotate(img2, 90)

        if self.with_random_rot180 and random.random() > 0.5:
            img1 = TF.rotate(img1, 180)
            img2 = TF.rotate(img2, 180)

        if self.with_random_rot270 and random.random() > 0.5:
            img1 = TF.rotate(img1, 270)
            img2 = TF.rotate(img2, 270)

        if self.with_random_crop and random.random() > 0.5:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=img1, scale=(0.5, 1.0), ratio=(0.9, 1.1))
            img1 = TF.resized_crop(
                img1, i, j, h, w, size=(self.img_size, self.img_size))
            img2 = TF.resized_crop(
                img2, i, j, h, w, size=(self.img_size, self.img_size))

        if self.with_random_brightness and random.random() > 0.5:
            # multiply a random number within a - b
            img1 = TF.adjust_brightness(img1, brightness_factor=random.uniform(0.5, 1.5))

        if self.with_random_gamma and random.random() > 0.5:
            # img**gamma
            img1 = TF.adjust_gamma(img1, gamma=random.uniform(0.5, 1.5))

        if self.with_random_saturation and random.random() > 0.5:
            # saturation_factor, 0: grayscale image, 1: unchanged, 2: increae saturation by 2
            img1 = TF.adjust_saturation(img1, saturation_factor=random.uniform(0.5, 1.5))

        # to tensor
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)

        return img1, img2


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


def parse_config(path_to_json=r'./config.json'):
    with open(path_to_json) as f:
        data = json.load(f)
    args = Struct(**data)

    return args


def clip_01(x):
    x[x > 1.0] = 1.0
    x[x < 0] = 0
    return x


def cpt_pxl_cls_acc(pred_idx, target):
    pred_idx = torch.reshape(pred_idx, [-1])
    target = torch.reshape(target, [-1])
    return torch.mean((pred_idx.int() == target.int()).float())


def cpt_batch_psnr(img, img_gt, PIXEL_MAX):
    mse = torch.mean((img - img_gt) ** 2, dim=[1, 2, 3])
    psnr = 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
    return torch.mean(psnr)


def cpt_psnr(img, img_gt, PIXEL_MAX):
    mse = np.mean((img - img_gt) ** 2)
    psnr = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return psnr


def cpt_rgb_ssim(img, img_gt):
    img = clip_01(img)
    img_gt = clip_01(img_gt)
    SSIM = 0
    for i in range(3):
        tmp = img[:, :, i]
        tmp_gt = img_gt[:, :, i]
        ssim = sk_cpt_ssim(tmp, tmp_gt)
        SSIM = SSIM + ssim
    return SSIM / 3.0


def cpt_ssim(img, img_gt):
    img = clip_01(img)
    img_gt = clip_01(img_gt)
    return sk_cpt_ssim(img, img_gt)
