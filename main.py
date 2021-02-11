# Standard libraries
import argparse
import pickle

# External Imports
import torch

from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

# Internal Imports
from src.models.models import SO3Classifier
from src.utils.evaluation import test_epoch
from src.utils.training import train_epoch, val_epoch

# Interesting feature I didn't know about.
# Helps debug autograd issues.
# torch.autograd.set_detect_anomaly(True)

# Optimizer options
optimizers = {"SGD": torch.optim.SGD,
              "Adadelta": torch.optim.Adadelta,
              "Adam": torch.optim.Adam}

# Image classifcation sets
data_sources = {"mnist": {"dataset": datasets.MNIST,
                          "transforms": transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                          "local_path": "data/mnist",
                          "in_channels": 1,
                          "size": 28,
                          "num_classes": 10},
                "cifar10": {"dataset":datasets.CIFAR10,
                          "transforms": transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                          "local_path": "data/cifar10",
                          "in_channels": 3,
                          "size": 32,
                          "num_classes": 10},
                "cifar100": {"dataset":datasets.CIFAR100,
                          "transforms": transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                          "local_path": "data/cifar100",
                          "in_channels": 3,
                          "size": 32,
                          "num_classes": 100}
                }


def main(epochs, lr, optimizer_type, es_param, data_source):
    """Train the SO3 model on basic datasets for fun, report performance metrics.
    """

    # Load data
    cfg = data_sources[data_source]

    # Train
    source_train = cfg["dataset"](cfg["local_path"], train=True, download=True, transform=cfg["transforms"])
    # split into train val
    ratio = 1/10
    num_samples = len(source_train)
    num_val = int(ratio*num_samples)
    train, val = torch.utils.data.random_split(source_train, [num_samples-num_val, num_val])
    train_loader = torch.utils.data.DataLoader(train, batch_size=1)
    val_loader = torch.utils.data.DataLoader(val, batch_size=1)

    # No validation yet, cause lazy and just getting something running

    # Test
    source_test = cfg["dataset"](cfg["local_path"], train=False, download=True, transform=cfg["transforms"])

    test_loader = torch.utils.data.DataLoader(source_test, batch_size=1)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize network
    in_channels = cfg["in_channels"]
    in_size = cfg["size"]
    num_classes = cfg["num_classes"]
    model = SO3Classifier(in_channels, in_size, num_classes).to(device)

    # Training utilities
    optimizer = optimizers[optimizer_type](model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=1)

    # Train loop
    train_losses = []
    val_losses = []
    for epoch in range(1, epochs + 1):
        # train
        epoch_train_losses = train_epoch(model, device, train_loader, optimizer, epoch)
        scheduler.step()
        train_losses.extend(epoch_train_losses)
        # val
        epoch_val_losses = val_epoch(model, device, val_loader, epoch)
        val_losses.extend(epoch_val_losses)

        # Basic early stopping
        increasing_loss = [(val_losses[-(i + 1)] > val_losses[-(es_param + 1)]) for i in range(es_param)]

        # If the mean loss from patience (*100) batches ago is smaller
        # then every mean loss since (e.g. if val loss starts monotonically increasing from overfitting)
        if False not in increasing_loss:
            print("Early stopping activated.")
            break

    # Save mean train losses
    with open(f'data/training_logs/train_losses.pkl', 'wb') as file:
        pickle.dump(train_losses, file)
    # Save mean val losses
    with open(f'data/training_logs/val_losses.pkl', 'wb') as file:
        pickle.dump(val_losses, file)

    # One test loop on test data
    test_epoch(model, test_loader, device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of training loops.')
    parser.add_argument('--lr',
                        type=float,
                        default=.001,
                        help='Learning rate to use in SGD.')
    parser.add_argument('--optimizer_type',
                        type=str,
                        default="SGD",
                        help="Optimizer type. Currently, one of: SGD, Adadelta, Adam")
    parser.add_argument('--es_param',
                        type=int,
                        default=5,
                        help="Param that controls this stupid version of \
                        early stopping.")
    parser.add_argument('--data_source',
                        type=str,
                        default="cifar100",
                        help="Which torchvision.datasets dataset to train on. \
                        Only mnist or cifar10 /cifar100 right now.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args_dict = vars(args)
    main(**args_dict)
