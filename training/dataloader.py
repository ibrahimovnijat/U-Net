from html.entities import html5
import torch
from torchvision import datasets, transforms

from pathlib import Path
import json
import os 
import numpy as np
import nibabel as nib
import h5py

from util.visualization import get_labeled_image 


class BraTSDataLoader(torch.utils.data.Dataset):
    def __init__(self, split, transform=None):
        super().__init__() 
        assert split in ["train", "valid"]
        
        self.split = split
        self.transform = transform

        self.num_channels = 4
        self.patch_dim = (160,160,16)
        self.num_classes = 3
    
        self.dataset_path = Path("data/BraTS-Data/processed/")
        with open(os.path.join(self.dataset_path, "config.json")) as json_file:
            data_config = json.load(json_file)

        self.items = data_config[split]        
        # print(self.items)


    def __getitem__(self, index):        
        data_file = os.path.join(self.dataset_path, self.split, self.items[index])
        image, label = self.__load_data(data_file)
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image" : image,
            "label" : label,
        }


    def __len__(self):
        return len(self.items) 


    def __load_data(self, data_file):
        X = np.zeros((1, self.num_channels, *self.patch_dim), dtype=np.float64)
        y = np.zeros((1, self.num_classes, *self.patch_dim), dtype=np.float64)
        with h5py.File(data_file, "r") as f:
            X[0,:] = np.array(f.get("x"))
            y[0,:] = np.moveaxis(np.array(f.get("y")), 3, 0)[1:]
        return X, y        


    def __load_data2(self, list_IDs_temp):
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


    def load_case(image_nifty_file, label_nifty_file):
        image = np.array(nib.load(image_nifty_file).get_fdata())
        label = np.array(nib.load(label_nifty_file).get_fdata())
        return image, label




class KaggleDataLoader(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()
        pass 

    def __getitem__(self, index):
        pass 

    
