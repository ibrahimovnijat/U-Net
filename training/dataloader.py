from html.entities import html5
from tkinter import Image
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
    """Brain MRI dataset for FLAIR abnormality segmentation"""

    in_channels = 3
    out_channels = 1

    def __init__(
        self,
        images_dir,
        transform=None,
        image_size=256,
        subset="train",
        random_sampling=True,
        validation_cases=10,
        seed=42,
    ):
        assert subset in ["all", "train", "validation"]

        # read images
        volumes = {}
        masks = {}
        print("reading {} images...".format(subset))
        for (dirpath, dirnames, filenames) in os.walk(images_dir):
            image_slices = []
            mask_slices = []
            for filename in sorted(
                filter(lambda f: ".tif" in f, filenames),
                key=lambda x: int(x.split(".")[-2].split("_")[4]),
            ):
                filepath = os.path.join(dirpath, filename)
                if "mask" in filename:
                    mask_slices.append(imread(filepath, as_gray=True))
                else:
                    image_slices.append(imread(filepath))
            if len(image_slices) > 0:
                patient_id = dirpath.split("/")[-1]
                volumes[patient_id] = np.array(image_slices[1:-1])
                masks[patient_id] = np.array(mask_slices[1:-1])

        self.patients = sorted(volumes)

        # select cases to subset
        if not subset == "all":
            random.seed(seed)
            validation_patients = random.sample(self.patients, k=validation_cases)
            if subset == "validation":
                self.patients = validation_patients
            else:
                self.patients = sorted(
                    list(set(self.patients).difference(validation_patients))
                )

        print("preprocessing {} volumes...".format(subset))
        # create list of tuples (volume, mask)
        self.volumes = [(volumes[k], masks[k]) for k in self.patients]

        print("cropping {} volumes...".format(subset))
        # crop to smallest enclosing volume
        self.volumes = [crop_sample(v) for v in self.volumes]

        print("padding {} volumes...".format(subset))
        # pad to square
        self.volumes = [pad_sample(v) for v in self.volumes]

        print("resizing {} volumes...".format(subset))
        # resize
        self.volumes = [resize_sample(v, size=image_size) for v in self.volumes]

        print("normalizing {} volumes...".format(subset))
        # normalize channel-wise
        self.volumes = [(normalize_volume(v), m) for v, m in self.volumes]

        # probabilities for sampling slices based on masks
        self.slice_weights = [m.sum(axis=-1).sum(axis=-1) for v, m in self.volumes]
        self.slice_weights = [
            (s + (s.sum() * 0.1 / len(s))) / (s.sum() * 1.1) for s in self.slice_weights
        ]

        # add channel dimension to masks
        self.volumes = [(v, m[..., np.newaxis]) for (v, m) in self.volumes]

        print("done creating {} dataset".format(subset))

        # create global index for patient and slice (idx -> (p_idx, s_idx))
        num_slices = [v.shape[0] for v, m in self.volumes]
        self.patient_slice_index = list(
            zip(
                sum([[i] * num_slices[i] for i in range(len(num_slices))], []),
                sum([list(range(x)) for x in num_slices], []),
            )
        )

        self.random_sampling = random_sampling

        self.transform = transform

    def __len__(self):
        return len(self.patient_slice_index)

    def __getitem__(self, idx):
        patient = self.patient_slice_index[idx][0]
        slice_n = self.patient_slice_index[idx][1]

        if self.random_sampling:
            patient = np.random.randint(len(self.volumes))
            slice_n = np.random.choice(
                range(self.volumes[patient][0].shape[0]), p=self.slice_weights[patient]
            )

        v, m = self.volumes[patient]
        image = v[slice_n]
        mask = m[slice_n]

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        # fix dimensions (C, H, W)
        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)

        image_tensor = torch.from_numpy(image.astype(np.float32))
        mask_tensor = torch.from_numpy(mask.astype(np.float32))

        # return tensors
        return image_tensor, mask_tensor