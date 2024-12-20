"""
Script for training a single model for OOD detection.
"""

import os
import torch
from torch import optim

# Import dataloaders
import data.ood_detection.cifar10 as cifar10
import data.ood_detection.cifar100 as cifar100
import data.ood_detection.svhn as svhn
import data.dirty_mnist as dirty_mnist

# Import network models
from net.lenet import lenet
from net.resnet import resnet18, resnet50
from net.wide_resnet import wrn
from net.vgg import vgg16

# Import train and validation utilities
from utils.args import training_args
from utils.train_utils import train_single_epoch


dataset_num_classes = {"cifar10": 10, "cifar100": 100, "svhn": 10, "dirty_mnist": 10}

dataset_loader = {
    "cifar10": cifar10,
    "cifar100": cifar100,
    "svhn": svhn,
    "dirty_mnist": dirty_mnist,
}

models = {
    "lenet": lenet,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "wide_resnet": wrn,
    "vgg16": vgg16,
}


if __name__ == "__main__":

    args = training_args().parse_args()

    print("Parsed args", args)
    print("Seed: ", args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    print("Using device:", device)

    num_classes = dataset_num_classes[args.dataset]

    # Choosing the model to train
    net = models[args.model](
        spectral_normalization=args.sn,
        mod=args.mod,
        coeff=args.coeff,
        num_classes=num_classes,
        mnist="mnist" in args.dataset,
    ).to(device)

    opt_params = net.parameters()
    if args.optimiser == "sgd":
        optimizer = optim.SGD(
            opt_params,
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimiser == "adam":
        optimizer = optim.Adam(opt_params, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[args.first_milestone, args.second_milestone], gamma=0.1
    )

    train_loader, _ = dataset_loader[args.dataset].get_train_valid_loader(
        root=args.dataset_root,
        batch_size=args.train_batch_size,
        augment=args.data_aug,
        val_size=0.1,
        val_seed=args.seed,
        pin_memory=args.gpu,
    )

    training_set_loss = {}

    for epoch in range(0, args.epoch):
        train_loss = train_single_epoch(epoch, net, train_loader, optimizer, device, loss_function=args.loss_function)

        scheduler.step()

        if not os.path.exists(args.save_loc):
            os.makedirs(args.save_loc)
        if (epoch+1) == 300:
            saved_name = os.path.join(args.save_loc, args.model + "_300.pt")
            torch.save(net.state_dict(), saved_name)

    saved_name = os.path.join(args.save_loc, args.model + "_" + str(epoch + 1) + ".pt")
    torch.save(net.state_dict(), saved_name)
    print("Model saved to ", saved_name)
