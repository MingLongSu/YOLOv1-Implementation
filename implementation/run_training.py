import os
import torch
import torch.optim as optim
import torchvision.transforms as transforms
import argparse

from tqdm import tqdm
from glob import glob
from utils import PascalVOCClasses, Compose
from dataset import PascalVOCDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from modelling import YOLOv1
from loss import YOLOv1Loss 
from engine_for_training import train_one_epoch

def get_args():
    arg_parser = argparse.ArgumentParser(
        prog='YoloV1 Training Script',
        description='Running this script will train the YoloV1 object detection model',
        add_help=False
    )
    # Defaults
    arg_parser.add_argument('--batch_size', default=4, type=int)
    arg_parser.add_argument('--epochs', default=135, type=int)
    arg_parser.add_argument('--save_epoch_freq', default=1, type=int)
    arg_parser.add_argument('--start_epoch', default=0, type=str) # IMPLEMENT THIS

    # Model parameters
    arg_parser.add_argument('--input_depth', default=3, type=int)
    arg_parser.add_argument('--input_size', default=448, type=int)
    arg_parser.add_argument('--grid_size', default=7, type=int)
    arg_parser.add_argument('--n_bounding_boxes', default=2, type=int)
    arg_parser.add_argument('--n_classes', default=20, type=int)

    # Optimizer parameters
    arg_parser.add_argument('--optimizer', default='sgd', type=str)
    arg_parser.add_argument('--momentum', default=0.9, type=int)
    arg_parser.add_argument('--weight_decay', default=5e-4, type=float) 
    arg_parser.add_argument('--lr', default=1e-3, type=float)
    arg_parser.add_argument('--warm_up_lr', default=1e-2, type=int)
    arg_parser.add_argument('--warm_up_epochs', default=30, type=int)

    # Dataset parameters
    arg_parser.add_argument('--metaset_path', default='path/to/meta_data/csv_file', type=str)
    arg_parser.add_argument('--imgs_path', default='path/to/imgs', type=str)
    arg_parser.add_argument('--lbls_path', default='path/to/lbls', type=str)
    arg_parser.add_argument('--output_dir', default='path/to/checkpoints', type=str)
    arg_parser.add_argument('--log_dir', default='path/to/log_dir', type=str)
    arg_parser.add_argument('--device', default='cuda', type=str)
    arg_parser.add_argument('--seed', default=31, type=int)
    arg_parser.add_argument('--resume', default=False, type=bool)
    arg_parser.add_argument('--n_workers', default=6, type=int)
    arg_parser.add_argument('--pin_memory', default=True, type=bool)
    arg_parser.add_argument('--shuffle', default=True, type=bool)
    arg_parser.add_argument('--drop_last', default=True, type=bool)
    arg_parser.add_argument('--dataset_mean', default=(0.485, 0.456, 0.406), type=tuple)
    arg_parser.add_argument('--dataset_std', default=(0.229, 0.224, 0.225), type=tuple)

    return arg_parser.parse_args()

def main(args: argparse.Namespace):
    """
    loss_fn = YOLOv1Loss()
    test_preds = torch.rand(())
    """
    
    # Init definitions
    classes = PascalVOCClasses()

    # Desired transformation
    transform = Compose([transforms.Resize((args.input_size, args.input_size)),])

    # Get model and loss function
    model = YOLOv1(args).to(args.device)
    loss_fn = YOLOv1Loss()
    if (args.resume):
        # Search for most recently created checkpoint and load weights
        ckpts = glob(os.path.join(args.output_dir, '*.ckpt'))
        ckpts = sorted(
            ckpts, 
            key=lambda ckpt: int(ckpt.split('/')[-1].split('.')[0]), 
            reverse=True
        )
        last_ckpt = ckpts[-1]
        model.load_state_dict(os.path.join(args.output_dir, last_ckpt))

    # Get optimizer and learning rate scheduler
    if (args.optimizer == 'sgd'):
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            momentum=args.momentum, 
            weight_decay=args.weight_decay
        )
    elif (args.optimizer == 'adam'):
        optimizer = optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5
    )

    # Get summary writer
    writer = SummaryWriter(args.log_dir)

    # Get training dataset
    trainset = PascalVOCDataset(
        args,
        transform=transform
    )
    trainset_loader = DataLoader(
        trainset,
        batch_size=args.batch_size, 
        shuffle=True,
        pin_memory=True, 
        num_workers=args.n_workers,
        drop_last=args.drop_last
    )

    # Iterate over epochs
    global_step = 0
    for epoch in tqdm(range(args.epochs), leave=True):
        global_step = train_one_epoch(
            model, loss_fn, optimizer, lr_scheduler,
            writer, global_step, trainset_loader, args
        )

        # Saving progress of training
        if (epoch % args.save_epoch_freq == 0):
            torch.save(model.state_dict, os.path.join(args.output_dir, f'yolov1_epoch_{epoch}.ckpt'))
    
    # Clean up
    writer.close()
    

if __name__ == '__main__':
    args = get_args()
    main(args)
