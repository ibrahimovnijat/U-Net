import os 
from pathlib import Path
import shutil
import numpy as np
import random 


# Copying all images/slices (for all patients) into images subfolder in mri_data folder
# Copying all masks into masks subfolder in mri_data folder
def sortData(data_dir):
    import os 
    import shutil
    
    images_dir = "data/mri_data/images/"
    masks_dir =  "data/mri_data/masks/"

    for folders in os.walk(data_dir):
        direct, subfolders, files = folders[0], folders[1], folders[2]
        print(f"Saving images in {direct} folder..")
        if len(files) > 2:
            for file in files:
                file_dir = direct + "/" + file
                if "mask" in file:
                    destination = masks_dir + file
                    shutil.copy(file_dir, destination) 
                else:
                    destination = images_dir + file 
                    shutil.copy(file_dir, destination)             
        print(f"Finished {direct} folder")
    print("Done.")




#Copying all image/mask pairs into their respective subfolders(with names) in mri_data/data 
def sortData2(data_dir):
    skip = True
    for folders in os.walk(data_dir):
        if skip:  # skip the original directory
            skip = False
            continue        
        direct, subfolders, files = folders[0], folders[1], folders[2]

        img_pairs, images, masks = [], [], []
        for file in files:
            if "mask" in file:
                masks.append(file)
            else:
                images.append(file)

        for image in images:
            image_name = image.split(".")[0]
            for mask in masks:
                mask_name = mask.split(".")[0]     
                if str(image_name + "_mask") == mask_name:
                    data = {}
                    data["image"] = image
                    data["mask"] = mask
                    data["folder_name"] = image_name
                    img_pairs.append(data) 
        
        for i in range(len(img_pairs)):
            # save image/mask to their respective folder
            new_dir = os.path.join("data/mri_data/data/", img_pairs[i]["folder_name"])
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            dest_image = new_dir + "/" + img_pairs[i]["image"]
            dest_mask  = new_dir + "/" + img_pairs[i]["mask"]
            src_image  = direct  + "/" + img_pairs[i]["image"]
            src_mask   = direct  + "/" + img_pairs[i]["mask"]

            shutil.copy(src_image, dest_image)
            shutil.copy(src_mask, dest_mask)
        print(f"Saved images/masks in {direct} folder..")    
    print("Done.")
    
    
    
def train_valid_split(data_dir: Path, valid_size=0.25, train_size=0.75, random_sampling=True):
    
    if (valid_size + train_size) > 1.0:
        train_size = 1. - valid_size
    
    data = os.listdir(data_dir)
    print("total data len:", len(data))
    
    num_valid = int(np.floor(len(data) * valid_size))
    num_train = int(np.floor(len(data) * train_size))
    
    if (num_valid + num_train) < len(data):
        num_train += len(data) - (num_valid + num_train)
    
    if random_sampling:
        valid_data = random.sample(data, int(num_valid))
        for val in valid_data:
            data.remove(val)    
        train_data = data
    else:
        valid_data = data[:num_valid]
        train_data = data[num_valid:]
    
    print("valid data len:", len(valid_data))
    print("train data len:", len(train_data))
    
    valid_path = os.path.join(data_dir.parents[0], "valid") 
    if not os.path.exists(valid_path):
        print(f"{valid_path} does not exist.. Creating directory..")
        os.mkdir(valid_path)
    elif os.path.exists(valid_path):
        print(f"{valid_path} exists.. Cleaning directory..")
        
        for folder in os.listdir(valid_path):
            shutil.rmtree(os.path.join(valid_path, folder))
        if len(os.listdir(valid_path)) == 0:
            print(f"{valid_path} cleared..")
        else:
            print(f"Something went wrong while clearing {valid_path} directory..")
            return 

    print(f"Saving validation data to {valid_path} folder..")
    # saving validation data into directory
    for patient_id in valid_data:
        # saving images/masks into valid directory/patient_folder
        patient_folder = os.path.join(valid_path, patient_id)
        if not os.path.exists(patient_folder):
            os.mkdir(patient_folder)
        
        dest_image = os.path.join(patient_folder, f"{patient_id}.tif")
        dest_mask = os.path.join(patient_folder, f"{patient_id}_mask.tif")

        src_image = os.path.join(data_dir, patient_id, f"{patient_id}.tif")
        src_mask = os.path.join(data_dir, patient_id, f"{patient_id}_mask.tif")     
        
        shutil.copy(src_image, dest_image)
        shutil.copy(src_mask, dest_mask)
    
    print("Done..")
    
    train_path = os.path.join(data_dir.parents[0], "train")
    if not os.path.exists(train_path):
        print(f"{train_path} does not exist.. Creating directory..")
        os.mkdir(train_path)
    elif os.path.exists(train_path):
        print(f"{train_path} exists.. Cleaning directory..")

        for folder in os.listdir(train_path):
            shutil.rmtree(os.path.join(train_path, folder))
        if len(os.listdir(train_path)) == 0:
            print(f"{train_path} cleared..")
        else:
            print(f"Something went wrong while cleaning {train_path} directory.")
            return 
        
    print(f"Saving training data to {train_path} folder..")
    # saving training data into directory
    for patient_id in train_data:
        # saving images/masks into train directory/patient_folder
        patient_folder = os.path.join(train_path, patient_id)
        if not os.path.exists(patient_folder):
            os.mkdir(patient_folder)

        dest_image = os.path.join(patient_folder, f"{patient_id}.tif")
        dest_mask = os.path.join(patient_folder, f"{patient_id}_mask.tif")

        src_image = os.path.join(data_dir, patient_id, f"{patient_id}.tif")
        src_mask = os.path.join(data_dir, patient_id, f"{patient_id}_mask.tif")     

        shutil.copy(src_image, dest_image)
        shutil.copy(src_mask, dest_mask)

    print("Done.")
    
