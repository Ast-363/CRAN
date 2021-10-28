import os
import numpy as np
import cv2

from torch.utils.data import Dataset


class DF2KDataset(Dataset):
    def __init__(self, db_path, transform):
        super(DF2KDataset, self).__init__()
        self.db_path = db_path
        self.transform = transform

        self.dirname_LR = os.path.join(self.db_path, 'DF2K_LR_bicubic/X4')
        self.dirname_HR = os.path.join(self.db_path, 'DF2K_HR')

        self.filelist_LR = os.listdir(self.dirname_LR)
        self.filelist_LR.sort()
        self.filelist_HR = os.listdir(self.dirname_HR)
        self.filelist_HR.sort()

    def __len__(self):
        return len(self.filelist_LR)
    
    def __getitem__(self, idx):

        img_name_LR = self.filelist_LR[idx]
        img_LR = cv2.imread(os.path.join(self.dirname_LR, img_name_LR), cv2.IMREAD_COLOR)
        img_LR = cv2.cvtColor(img_LR, cv2.COLOR_BGR2RGB)
        img_LR = np.array(img_LR).astype('float32') / 255

        img_name_HR = self.filelist_HR[idx]
        img_HR = cv2.imread(os.path.join(self.dirname_HR, img_name_HR), cv2.IMREAD_COLOR)
        img_HR = cv2.cvtColor(img_HR, cv2.COLOR_BGR2RGB)
        img_HR = np.array(img_HR).astype('float32') / 255

        sample = {'img_LR': img_LR, 'img_HR': img_HR}

        if self.transform:
            sample = self.transform(sample)
        
        return sample