import torch
import torchvision

import math
import cv2
import numpy as np
from scipy.ndimage import rotate


class RandCrop(object):
    def __init__(self, crop_size, scale):
        # if output size is tuple -> (height, width)
        assert isinstance(crop_size, (int, tuple))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            assert len(crop_size) == 2
            self.crop_size = crop_size
        
        self.scale = scale

    def __call__(self, sample):
        # img_LR: H x W x C (numpy array)
        img_LR, img_HR = sample['img_LR'], sample['img_HR']

        h, w, c = img_LR.shape
        new_h, new_w = self.crop_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        img_LR_crop = img_LR[top: top+new_h, left: left+new_w, :]

        h, w, c = img_HR.shape
        top = np.random.randint(0, h - self.scale*new_h)
        left = np.random.randint(0, w - self.scale*new_w)
        img_HR_crop = img_HR[top: top + self.scale*new_h, left: left + self.scale*new_w, :]

        sample = {'img_LR': img_LR_crop, 'img_HR': img_HR_crop}
        return sample


class RandRotate(object):
    def __call__(self, sample):
        # img_LR: H x W x C (numpy array)
        img_LR, img_HR = sample['img_LR'], sample['img_HR']

        prob_rotate = np.random.random()
        if prob_rotate < 0.25:
            img_LR = rotate(img_LR, 90).copy()
            img_HR = rotate(img_HR, 90).copy()
        elif prob_rotate < 0.5:
            img_LR = rotate(img_LR, 90).copy()
            img_HR = rotate(img_HR, 90).copy()
        elif prob_rotate < 0.75:
            img_LR = rotate(img_LR, 90).copy()
            img_HR = rotate(img_HR, 90).copy()
        
        sample = {'img_LR': img_LR, 'img_HR': img_HR}
        return sample


class RandHorizontalFlip(object):
    def __call__(self, sample):
        # img_LR: H x W x C (numpy array)
        img_LR, img_HR = sample['img_LR'], sample['img_HR']

        prob_lr = np.random.random()
        if prob_lr < 0.5:
            img_LR = np.fliplr(img_LR).copy()
            img_HR = np.fliplr(img_HR).copy()
        
        sample = {'img_LR': img_LR, 'img_HR': img_HR}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        # img_LR : H x W x C (numpy array) -> C x H x W (torch tensor)
        img_LR, img_HR = sample['img_LR'], sample['img_HR']

        img_LR = img_LR.transpose((2, 0, 1))
        img_HR = img_HR.transpose((2, 0, 1))

        img_LR = torch.from_numpy(img_LR)
        img_HR = torch.from_numpy(img_HR)

        sample = {'img_LR': img_LR, 'img_HR': img_HR}
        return sample


class VGG19PerceptualLoss(torch.nn.Module):
    def __init__(self, feature_layer=35):
        super(VGG19PerceptualLoss, self).__init__()
        model = torchvision.models.vgg19(pretrained=True)
        self.features = torch.nn.Sequential(*list(model.features.children())[:feature_layer]).eval()
        # Freeze parameters
        for name, param in self.features.named_parameters():
            param.requires_grad = False
    
    def forward(self, source, target):
        vgg_loss = torch.nn.functional.l1_loss(self.features(source), self.features(target))

        return vgg_loss
        

