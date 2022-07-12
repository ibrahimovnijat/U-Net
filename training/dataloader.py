import torch
from torchvision import datasets, transforms

# from torch.utils.data.sampler import SubsetRandomSampler

from pathlib import Path
import numpy as np
import nibabel as nib

from util.visualization import get_labeled_image 


class BraTS_Loader(torch.utils.data.Dataset):

    dataset_path = Path("data/BraTS-Data/")

    def __init__(self, split):
        super().__init__() 
        assert split in ["train", "val"]

    
        
    def __getitem__(self):
        pass 


    def __len__(self):
        pass 

    def load_case(image_nifty_file, label_nifty_file):
        image = np.array(nib.load(image_nifty_file).get_fdata())
        label = np.array(nib.load(label_nifty_file).get_fdata())
        return image, label


    def preprocess(self, image):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomRotation(10),
            # transforms.RandomHorizontalFlip(),
        ])
        return self.transforms(image)



    def get_patches(self):
        pass
