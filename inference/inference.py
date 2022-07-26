import torch
from model.unet import UNet2D
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
import random


class UNet2DInference:
    """Utility for inference using trained UNet model"""

    def __init__(self, ckpt):
        self.model = UNet2D(3,1)
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()


    def inference(self, image: torch.tensor)->torch.tensor:
        # input_tensor = torch.from_numpy(image).float().unsqueeze(0)
        pred =  self.model(image.float().unsqueeze(0))
        pred = (pred > 0.5).float()
        return pred.squeeze(0)
    
    
    def plot_results(self, dataset, indexes):
        num_rows = len(indexes)        
        for i, index in enumerate(indexes):
            # load image/mask
            image = dataset[index]["image"]
            mask = dataset[index]["mask"]
            # inference 
            prediction = self.inference(image)
            
            image = torch.permute(image, (1,2,0)).cpu().detach().numpy()
            mask = torch.permute(mask, (1,2,0)).cpu().detach().numpy()
            prediction = torch.permute(prediction, (1,2,0)).cpu().detach().numpy()
            
            # plot results
            figure(figsize=(9,9), dpi=80)
            plt.subplot(1, 3, 1)
            plt.imshow(image, cmap="gray")
            plt.title("Original image")
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Label")
            plt.subplot(1, 3, 3)
            plt.imshow(prediction, cmap="gray")
            plt.title("Prediction")
            plt.show()
        
        
    def random_inference(self, dataset, plot=True):
        rand_idx = random.randint(0, len(dataset))
        print("data index:", rand_idx)
        image = dataset[rand_idx]["image"]
        mask = dataset[rand_idx]["mask"]
        
        prediction = self.inference(image)
        
        image = torch.permute(image, (1,2,0)).cpu().detach().numpy()
        mask = torch.permute(mask, (1,2,0)).cpu().detach().numpy()
        prediction = torch.permute(prediction, (1,2,0)).cpu().detach().numpy()
        
        if plot:
            figure(figsize=(9,9), dpi=80)
            plt.subplot(1, 3, 1)
            plt.imshow(image, cmap="gray")
            plt.title("Original image")
            plt.subplot(1, 3, 2)
            plt.imshow(mask, cmap="gray")
            plt.title("Label")
            plt.subplot(1, 3, 3)
            plt.imshow(prediction, cmap="gray")
            plt.title("Prediction")
            plt.show()
