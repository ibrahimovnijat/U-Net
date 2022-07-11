import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

from pathlib import Path



class DataLoader(torch.utils.data.Dataset):

    num_classes = 10
    dataset_path = Path("data/")


    def __init__(self):
        super().__init__() 
        
    def __getitem__(self):
        pass 


    def __len__(self):
        pass 


    def preprocess(self, image):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomRotation(10),
            # transforms.RandomHorizontalFlip(),
        ])
        return self.transforms(image)


    @staticmethod
    def move_batch_to_device(batch, device):
        pass

