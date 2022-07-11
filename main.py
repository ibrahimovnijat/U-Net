import argparse
import os 
import sys 

from training import unet_training

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():

    #TODO add default configs here...
    default_config = {}
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--max-epochs", type=int, default=50, help="Num epochs for training [default is 50]")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size during training [default is 32]")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for training [default is 0.001]")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer [default is 0.9]")
    parser.add_argument("--optimizer", default="adam", help="Optimizer: adam, sgd [default is adam]")
    parser.add_argument("--decay-rate", type=float, default=0.7, help="Decay rate for learning rate [default is 0.7]")
    parser.add_argument("--device", default="cuda", help="Training device: cuda or cpu [default is cuda]")
    parser.add_argument("--experiment-name", default="test_experiment", help="Enter an experiment name [default is test_experiment]")
    parser.add_argument("--pin-memory", default=False, help="Pin memory to fast data loading [default is True]")
    parser.add_argument("--log-dir", default="logs", help="Log directory [default is 'log'] ")
    parser.add_argument("--num-workers", type=int, default=4, help="Num workers for dataloader [default is 4]")
    parser.add_argument("--plot-results", default=False, help="Plot training/validation accuracy and error [default is False]")
    parser.add_argument("--save-losses-to-txt", default=True, help="Save training/validation accuracy/error to txt file in log [default is true]")
    
    flags = parser.parse_args()
    # print(flags)

    print("pin_memory:", flags.pin_memory)

    train_config = {
        "experiment_name" : flags.experiment_name,
        "device" : flags.device,
        "batch_size" :   flags.batch_size,
        "learning_rate" : flags.learning_rate,
        "max_epochs" : flags.max_epochs,
        "optimizer" : flags.optimizer,
        "decay_rate" : flags.decay_rate,
        "momentum" : flags.momentum,

        "resume_ckpt" : None,
        "num_workers" : flags.num_workers,
        "pin_memory" : flags.pin_memory,

        "print_every_n" : 100,
        "validate_every_n": 100,
        "save_losses_to_txt" : flags.save_losses_to_txt,   
        "plot_results" : flags.plot_results,
        "log_dir" : flags.log_dir,
    }

    print(train_config)

    # unet_training.main(train_config)
    


if __name__ == "__main__": main()
    
