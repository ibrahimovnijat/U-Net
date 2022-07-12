import torch
from pathlib import Path
import nibabel as nib

from model.unet import UNet
from model.losses import dice_coefficient, single_class_dice_coefficient, soft_dice_loss



def train(model, trainloader, valloader, config):

    # declare loss and move to specified device
    loss_criterion = soft_dice_loss()
    loss_criterion.to(config["device"])

    # declare optimizer
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], betas=config["betas"], 
                                     weight_decay=config["weight_decay"], eps=config["eps"])
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"])
    else:
        print("Wrong optimizer chosen..")
        return 

    # set model to train, important if your network has e.g. dropout or batchnorm layers
    model.train()
    
    # implement the training loop. It looks very much the same as in the previous exercise part, except that you are now using points instead of voxel grids
    best_accuracy = 0.
    train_loss_running = 0.
    
    for epoch in range(config["max_epochs"]):
        for i, batch in enumerate(trainloader):
            # move batch to device 
            ShapeNetPoints.move_batch_to_device(batch, config["device"])
            
            optimizer.zero_grad()
            
            # forward path
            prediction = model(batch["points"])
            
            # calculate batch loss 
            batch_train_loss = loss_criterion(prediction, batch["label"])
            
            batch_train_loss.backward()            
            optimizer.step()
    
            train_loss_running += batch_train_loss.item()
        
            iteration = epoch * len(trainloader) + i            
            if iteration % config["print_every_n"] == (config["print_every_n"] - 1):
                print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running /  config["print_every_n"]:.3f}')
                train_loss_running = 0.

            # validation every n 
            if iteration % config["validate_every_n"] == (config["validate_every_n"] - 1):
                model.eval()
                
                batch_val_loss = 0.
                correct = 0.
                total = 0.
                
                for batch in valloader:
                    ShapeNetPoints.move_batch_to_device(batch, device)
                    
                    with torch.no_grad():
                        prediction = model(batch["points"])
                        _, predicted_label = torch.max(prediction, dim=1)

                        correct += (predicted_label == batch["label"]).sum().item()
                        batch_val_loss += loss_criterion(prediction, batch["label"]).item()
                        
                        total += predicted_label.shape[0]

                accuracy = 100 * correct / total       
                print(f'[{epoch:03d}/{i:05d}] val_loss: {batch_val_loss / len(valloader):.3f}, val_accuracy: {accuracy:.3f}%')
                    
                if accuracy > best_accuracy:
                    torch.save(model.state_dict(), f'runs/{config["experiment_name"]}/model_best.ckpt')
                    best_accuracy = accuracy
                    print("Validation accuracy increased.. Saving model..")
                model.train()
            
                
def main(config):
    """
    Function for training U-Net on BraTD Data

    :param config: configuration for training - has the following keys
                   'experiment_name': name of the experiment, checkpoint will be saved to folder "exercise_2/runs/<experiment_name>"
                   'device': device on which model is trained, e.g. 'cpu' or 'cuda:0'
                   'batch_size': batch size for training and validation dataloaders
                   'resume_ckpt': None if training from scratch, otherwise path to checkpoint (saved weights)
                   'learning_rate': learning rate for optimizer
                   'max_epochs': total number of epochs after which training should stop
                   'print_every_n': print train loss every n iterations
                   'validate_every_n': print validation loss and validation accuracy every n iterations
                   'is_overfit': if the training is done on a small subset of data specified in exercise_2/split/overfit.txt,
                                 train and validation done on the same set, so error close to 0 means a good overfit. Useful for debugging.
    """

    # Declare device
    device = torch.device("cpu")
    if torch.cuda.is_available() and config["device"].startswith("cuda"):
        device = torch.device(config["device"])
        print("Using device:", config["device"])
    else:
        print("Using CPU")
        
    # Create Dataloaders
    train_dataset = ShapeNetPoints('train' if not config['is_overfit'] else 'overfit')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,   # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=True,    # Shuffling the order of samples is useful during training to prevent that the network learns to depend on the order of the input data
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    val_dataset = ShapeNetPoints('val' if not config['is_overfit'] else 'overfit')


    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,     # Datasets return data one sample at a time; Dataloaders use them and aggregate samples into batches
        batch_size=config['batch_size'],   # The size of batches is defined here
        shuffle=False,   # During validation, shuffling is not necessary anymore
        num_workers=4,   # Data is usually loaded in parallel by num_workers
        pin_memory=True  # This is an implementation detail to speed up data uploading to the GPU
    )

    # Instantiate model
    # model = PointNetClassification(ShapeNetPoints.num_classes)
    model = UNet(num_classes=4)

    # Load model if resuming from checkpoint
    if config["resume_ckpt"] is not None:
        model.load_state_dict(torch.load(config["resume_ckpt"], map_location="cpu"))

    # Move model to specified device
    model.to(config["device"])

    # Create folder for saving checkpoints
    Path(f"runs/{config['experiment_name']}").mkdir(exist_ok=True, parents=True)

    # Start training
    # train(model, train_dataloader, val_dataloader, config)

    
    