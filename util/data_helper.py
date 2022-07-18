import os 
from pathlib import Path
import shutil
import numpy as np


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
