import torch

# unique cell identifier: dice coefficient for a single output class 
def single_class_dice_coefficient(y_pred, y_true, axis=(0,1,2), epsilon=1e-8):
    dice_num = 2 * torch.sum(y_true* y_pred, axis=axis) + epsilon
    dice_denum = torch.sum(y_true, axis=axis) + torch.sum(y_pred, axis=axis) + epsilon
    dice_coeff = dice_num / dice_denum
    # print("dice_coeff:", dice_coeff)
    return dice_coeff


# dice coefficient for multiple classes
def dice_coefficient(y_pred, y_true, axis=(0,1,2), epsilon=1e-8):
    dice_num = 2 * torch.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denum = torch.sum(y_true, axis=axis) + torch.sum(y_pred, axis=axis) + epsilon
    dice_coeff = torch.mean(dice_num / dice_denum)
    return dice_coeff


# soft dice loss suited much better for training
def soft_dice_loss(y_pred, y_true, axis=(1,2,3), epsilon=1e-8):
    dice_num = 2 * torch.sum(y_true*y_pred, axis=axis) + epsilon
    dice_denum = torch.sum(y_true*y_true, axis=axis) + torch.sum(y_pred*y_pred, axis=axis) + epsilon
    dice_loss = 1 - torch.mean(dice_num/dice_denum)
    return dice_loss