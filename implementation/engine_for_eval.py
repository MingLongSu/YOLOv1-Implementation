import torch.nn as nn

from tqdm import tqdm
from vis import view_image
from argparse import ArgumentParser
from torch.utils.data import DataLoader

def test_one_epoch(model: nn.Module, loss_fn: nn.Module, 
                    test_loader: DataLoader, args: ArgumentParser):
    """
    Trains the model on a single epoch

    Parameters
    -------
    model : nn.Module
        Passed deep learning model for training on single epoch
    loss_fn : nn.Module 
        Loss function to compute the loss from outputs and ground truths
    test_loader : DataLoader
        Data loader for training
    args : ArgumentParser
        Stores the arguments initially passed on running
    """
    # Evaluate test set by mean loss on one epoch
    model.eval()
    mean_loss = []
    for X, y in tqdm(test_loader, leave=True):
        X, y = X.to(args.device), y.to(args.device)
        hat_y = model(X)
        loss = loss_fn(hat_y, y)
        mean_loss.append(loss.item())

    print(f'Mean loss {sum(mean_loss)/len(mean_loss)}')

    # Outputting first n images for visualization
    counter = 0 
    n_to_show = 2
    for X, y in test_loader:
        if (counter == n_to_show):
            break
        
        X, y = X.to(args.device), y.to(args.device)
        hat_y = model(X)
        view_image()

        counter += 1

    model.train()
