
import numpy as np
import torch

# unique cell identifier: dice coefficient for a single output class 
def single_class_dice_coefficient(y_true, y_pred, axis=(0,1,2), espilon=0.00001):
    dice_num = 2 * torch.sum(y_true* y_pred, axis=axis) + epsilon
    dice_denom = torch.sum(y_true, axis=axis) + torch.sum(y_pred, axis=axis) + epsilon
    dice_coeff = dice_num / dice_denom
    return dice_coeff


# dice coefficient for multiple classes
def dice_coefficient(y_true, y_pred, axis=(1,2,3), epsilon=0.00001):
    dice_num = 2 * torch.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denum = torch.sum(y_true, axis=axis) + torch.sum(y_pred, axis=axis) + epsilon
    dice_coeff = torch.mean(dice_num / dice_denom)
    return dice_coeff


# soft dice loss suited much better for training
def soft_dice_loss(y_true, y_pred, axis=(1,2,3), epsilon=0.00001):
    dice_num = 2 * torch.sum(y_true*y_pred, axis=axis) +epsilon
    dice_denom = torch.sum(y_true*y_true, axis=axis) + torch.sum(y_pred*y_pred, axis=axis) + epsilon
    dice_loss = 1 - torch.mean(dice_num/dice_denom)    
    return dice_loss

