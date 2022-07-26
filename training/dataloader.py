import torch
from torchvision import datasets, transforms

from pathlib import Path
import json
import os 
from PIL import Image
import numpy as np

import nibabel as nib
import h5py

from util.visualization import get_labeled_image 

import random 
from skimage.io import imread
from util.extra import crop_sample, pad_sample, resize_sample, normalize_volume
import cv2 as cv 


class BraTSDataLoader(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        super().__init__() 
        assert subset in ["train", "valid"]
        
        self.subset = subset
        self.transform = transform

        self.num_channels = 4
        self.patch_dim = (160,160,16)
        self.num_classes = 3
    
        self.dataset_path = Path("data/BraTS-Data/processed/")
        with open(os.path.join(self.dataset_path, "config.json")) as json_file:
            data_config = json.load(json_file)

        self.items = data_config[subset]        
        # print(self.items)


    def __getitem__(self, index):        
        data_file = os.path.join(self.dataset_path, self.subset, self.items[index])
        image, label = self._load_data(data_file)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image" : image,
            "label" : label,
        }

    def __len__(self):
        return len(self.items) 


    def _load_data(self, data_file):
        X = np.zeros((1, self.num_channels, *self.patch_dim), dtype=np.float64)
        y = np.zeros((1, self.num_classes, *self.patch_dim), dtype=np.float64)
        with h5py.File(data_file, "r") as f:
            X[0,:] = np.array(f.get("x"))
            y[0,:] = np.moveaxis(np.array(f.get("y")), 3, 0)[1:]
        return X, y        


    def _load_data2(self, list_IDs_temp):
        # 1 in the beginning is for the batch 
        X = np.zeros((1, self.num_channels, *self.patch_dim), dtype=np.float64)
        y = np.zeros((1, self.num_classes, *self.patch_dim), dtype=np.float64)
        print("list of IDs:", list_IDs_temp)

        for i, ID in enumerate(list_IDs_temp):
            if self.verbose:
                print(f"Training on {self.dataset_path + ID}")
            with h5py.File(self.dataset_path + ID, "r") as f:
                X[i] = np.array(f.get("x"))
                # remove the background class
                y[i] = np.moveaxis(np.array(f.get("y")), 3, 0)[1:]
        return X, y


    def _load_case(image_nifty_file, label_nifty_file):
        image = np.array(nib.load(image_nifty_file).get_fdata())
        label = np.array(nib.load(label_nifty_file).get_fdata())
        return image, label



class MRIDataLoader(torch.utils.data.Dataset):

    def __init__(self, data_dir, subset="train", transform=None, image_dim=None, mask_dim=None):
        assert subset in ["train", "valid"]

        self.data_dir = data_dir + subset
        for data in os.walk(self.data_dir):
            self.patients = data[1]
            break
        self.transform = transform
        self.image_dim = image_dim
        self.mask_dim = mask_dim

    
    def __getitem__(self, idx):   
        image, mask = self._readimage(idx)
        
        # additional transformation for training?!  
        if self.transform is not None:
            image = self.transform(image)
            mask  = self.transform(mask)

        return {
            "image" : image,
            "mask" : mask
        }

    
    def __len__(self):
        return len(self.patients)


    def _readimage(self, idx, standardize=True):
        image = self.data_dir + "/" + self.patients[idx] + "/" + self.patients[idx] + ".tif"
        mask = self.data_dir +  "/" + self.patients[idx] + "/" + self.patients[idx] + "_mask.tif"
        img = cv.imread(image, cv.IMREAD_COLOR)
        msk = cv.imread(mask, cv.IMREAD_GRAYSCALE)

        if self.image_dim:
            img = cv.resize(img, self.image_dim, cv.INTER_AREA)
        if self.mask_dim:
            msk = cv.resize(msk, self.mask_dim, cv.INTER_AREA)
        
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)    
        msk_tensor = torch.from_numpy(msk).type(torch.FloatTensor) 
        img_tensor = torch.permute(img_tensor, (2,0,1))
        msk_tensor = msk_tensor.unsqueeze(0)
        
        return self._adjust(img_tensor, msk_tensor)

    
    def _adjust(self, img, mask):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        return img, mask
