import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm import tqdm

from opt.option import args
from data.DF2K_dataset import DF2KDataset
from util.utils import RandCrop, RandHorizontalFlip, RandRotate, ToTensor
from model.rcan import RCAN
from model.discriminator import UNetDiscriminator
from loss.loss import VGG19PerceptualLoss, calc_psnr, calc_ssim

# device setting
if args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('using GPU %s' % args.gpu_id)
else:
    print('use --gpu_id to specify GPU ID to use')
    exit()

# make directory for saving weights
if not os.path.exists(args.snap_path):
    os.mkdir(args.snap_path)


# load training dataset
train_dataset = DF2KDataset(
    db_path=args.dir_data,
    transform = transforms.Compose([RandCrop(args.patch_size, args.scale), RandHorizontalFlip(), RandRotate(), ToTensor()])
)
train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    drop_last=True,
    shuffle=True
)


# define model (Generator)
model_G = RCAN(args).cuda()


# loss & optimizer & scheduler
loss_L1 = nn.L1Loss().cuda()
optimizer_G = torch.optim.Adam(
    params=model_G.parameters(),
    lr=args.lr_G,
    weight_decay=args.weight_decay
)
scheduler_G = torch.optim.lr_scheduler.StepLR(
    optimizer=optimizer_G,
    step_size=args.step_size_G,
    gamma=args.gamma_G
)


# load weights & optimizer
if args.checkpoint is not None:
    checkpoint = torch.load(args.checkpoint)
    model_G.load_state_dict(checkpoint['model_G'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G'])
    scheduler_G.load_state_dict(checkpoint['scheduler_G'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
else:
    start_epoch = 0


# make directory for saving weights
if not os.path.exists(args.snap_path):
    os.mkdir(args.snap_path)


# training
for epoch in range(start_epoch, args.epochs):
    model_G.train()

    running_loss = 0.0
    iter = 0

    psnr_total = 0.0
    ssim_total = 0.0

    for data in tqdm(train_loader):
        iter += 1

        img_LR, img_HR = data['img_LR'].cuda(), data['img_HR'].cuda()
        img_SR = model_G(img_LR)

        optimizer_G.zero_grad()
        loss = loss_L1(img_SR, img_HR)
        loss.backward()
        
        optimizer_G.step()
        scheduler_G.step()

        psnr = calc_psnr(img_SR, img_HR, args.scale, args.rgb_range)
        ssim = calc_ssim(img_SR, img_HR)

        running_loss += loss.item()
        psnr_total += psnr.item()
        ssim_total += ssim.item()
    
    print('[train] epoch: %d, lr: %f, loss: %f, psnr: %f, ssim: %f' % (epoch, optimizer_G.param_groups[0]['lr'], running_loss/iter, psnr_total/iter, ssim_total/iter))

    if (epoch +1) % args.save_freq == 0:
        weights_file_name = 'epoch%d.pth' % (epoch+1)
        weights_file = os.path.join(args.snap_path, weights_file_name)
        torch.save({
            'epoch': epoch,
            'model_G': model_G.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'scheduler_G': scheduler_G.state_dict(),
            'loss': loss
        }, weights_file)
        print('save weights of epoch %d' % (epoch+1))