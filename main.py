import argparse
import os 
import sys 

from training import unet_training

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--max-epoch", type=int, default=50, help="Num epochs for training [default is 50]")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size during training [default is 32]")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for training [default is 0.001]")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for optimizer [default is 0.9]")
    parser.add_argument("--optimizer", default="adam", help="Optimizer: adam, sgd [default is adam]")
    parser.add_argument("--decay-rate", type=float, default=0.7, help="Decay rate for learning rate [default is 0.7]")
    parser.add_argument("--device", default="cuda", help="Training device: cuda or cpu [default is cuda]")
    parser.add_argument("--experiment-name", default="test_experiment", help="Enter an experiment name [default is test_experiment]")
    parser.add_argument("--pin-memory", type=bool, default=True, help="Pin memory to fast data loading [default is True]")



    config_flags = parser.parse_args()
    print(config_flags)

    config = {
        "experiment_name" : "test",
        "device" : "cuda",
        "batch_size" : 32,
        "learning_rate" : 1e-3,
        "max_epochs" : 50,
        "optimizer" : "adam",
        "decay_rate" : 0.7,
        "momentum" : 0.9,

        "resume_ckpt" : None,
        "num_workers" : 4,
        "pin_memory" : True,

        "print_every_n" : 100,
        "validate_every_n": 100,
        "save_losses_to_txt" : False,   
        "plot_results" : False,

    }

    # unet_training.main(config)



if __name__ == "__main__": main()

