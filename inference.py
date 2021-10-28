import os
import numpy as np
import cv2
from tqdm import tqdm

import torch
import torchvision

from model.rcan import RCAN
from opt.option import args


# device setting
if args.gpu_id is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('using GPU %s' % args.gpu_id)
else:
    print('use --gpu_id to specify GPU ID to use')
    exit()


# make directory for saving weights
if not os.path.exists(args.results):
    os.mkdir(args.results)


# numpy array -> torch tensor
class ToTensor(object):
    def __call__(self, sample):
        sample = np.transpose(sample, (2, 0, 1))
        sample = torch.from_numpy(sample)
        return sample


# create model & load weights
model_G = RCAN(args).cuda()
checkpoint = torch.load(args.weights)
model_G.load_state_dict[checkpoint['model_G']]
model_G.eval()


# input transform
transforms = torchvision.transforms.Compose([ToTensor()])


filenames = os.listdir(args.dir_test)
filenames.sort()
for filename in tqdm(filenames):
    img_name = os.path.join(args.dir_test, filename)
    ext = os.path.splitext(img_name)[-1]
    if ext == '.png':
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img).astype('float32') / 255
        
        img = transforms(img)
        img = torch.tensor(img.cuda()).unsqueeze(0)

        # inference output
        out = model_G(img)

        # result image save (b x c x h x w (torch tensor) -> h x w x c (numpy array))
        out = out.data.cpu().squeeze().numpy()
        out = np.clip(out, 0, 1)
        out = np.transpose(out, (1, 2, 0))
        cv2.imwrite('%s_out.png' % (os.path.join(args.results, filename)[:-4]), out)
