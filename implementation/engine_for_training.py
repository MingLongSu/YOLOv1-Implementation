import torch.nn as nn
import torch.optim as optim
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train_one_epoch(model: nn.Module, loss_fn: nn.Module, optimizer: optim.Optimizer, 
                    lr_scheduler: optim.lr_scheduler.LRScheduler, writer: SummaryWriter, 
                    global_step: int, train_loader: DataLoader, args: argparse.ArgumentParser):
    """
    Trains the model on a single epoch

    Parameters
    -------
    model : nn.Module
        Passed deep learning model for training on single epoch
    loss_fn : nn.Module 
        Loss function to compute the loss from outputs and ground truths
    optimizer : optim.Optimizer
        Optimizer for training
    lr_scheduler : optim.lr_schedulers
        Updates learning rate based on learning progress
    train_loader : DataLoader
        Data loader for training
    args : ArgumentParser
        Stores the arguments initially passed on running
    """ 
    model.train()
    mean_loss = []
    for X, y in tqdm(train_loader, leave=True):
        X, y = X.to(args.device), y.to(args.device)
        # print('X', X)
        hat_y = model(X)
        loss = loss_fn(hat_y, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
    mean_loss_scalar = sum(mean_loss)/len(mean_loss)
    lr_scheduler.step(mean_loss_scalar)
    writer.add_scalar("YoloV1 Training Loss", mean_loss_scalar, global_step=global_step)
    print(f'Mean loss: {mean_loss_scalar}')
    print(mean_loss)
    return global_step
