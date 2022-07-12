import torch
from torchvision import datasets, transforms

from pathlib import Path
import json
import numpy as np
import nibabel as nib

from util.visualization import get_labeled_image 


class BraTSDataLoader(torch.utils.data.Dataset):

    def __init__(self, split):
        super().__init__() 

        print("__init__ called")
        assert split in ["train", "valid"]
        print("test passed")
        
        self.num_channels = 4
        self.dim = (160,160,16)
        self.num_classes = 3
        self.config_path = "data/BraTS-Data/processed/config.json"        
        with open(self.config_path) as json_file:
            data_config = json.load(json_file)
        self.items = data_config[split]
        
        print(self.items)    
        print("end of __init__")


    def __getitem__(self, index):        
        image_data = self.items[index]
        return {
            "image" : None,
            "label" : None,
        }
        

    def __len__(self):
        return len(self.items) 


    @staticmethod
    def load_case(image_nifty_file, label_nifty_file):
        image = np.array(nib.load(image_nifty_file).get_fdata())
        label = np.array(nib.load(label_nifty_file).get_fdata())
        return image, label


    @staticmethod
    def preprocess(self, image):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomRotation(10),
            # transforms.RandomHorizontalFlip(),
        ])
        return self.transforms(image)


    def get_patches(self):
        pass






class KaggleDataLoader(torch.utils.data.Dataset):

    def __init__(self):
        super().__init__()
        pass 

    def __getitem__(self, index):
        pass 

    
