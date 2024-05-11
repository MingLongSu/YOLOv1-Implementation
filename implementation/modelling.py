import torch
import torch.nn as nn
import argparse

class CNN_Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, 
                 kernel_size: int, stride: int, padding: int): 
        super(CNN_Block, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batchnorm2d = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm2d(self.conv2d(x)))
    
class YOLOv1(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(YOLOv1, self).__init__()
        self.in_channels = args.input_depth
        self.grid_size = args.grid_size
        self.n_bounding_boxes = args.n_bounding_boxes
        self.n_classes = args.n_classes
        self.darknet_architecture = [
            (7, 64, 2, 3), 
            'm',
            (3, 192, 1, 1),
            'm',
            (1, 128, 1, 0),
            (3, 256, 1, 1), 
            (1, 256, 1, 0),
            (3, 512, 1, 1),
            'm',
            [(1, 256, 1, 0), (3, 512, 1, 1), 4],
            (1, 512, 1, 0),
            (3, 1024, 1, 1),
            'm',
            [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
            (3, 1024, 1, 1), 
            (3, 1024, 2, 1),
            (3, 1024, 1, 1),
            (3, 1024, 1, 1),
        ]
        self.darknet, self.last_out_channels = self.build_darknet_()
        self.connected = self.build_connected_()

    def forward(self, x):
        return self.connected(torch.flatten(self.darknet(x), start_dim=1))
    
    def build_darknet_(self):
        # Make copy of architecture and initialize number of input channels
        architecture_ = self.darknet_architecture
        in_channels_ = self.in_channels

        # Iterate over layers of architecture and build sequentially
        sequential = []
        for layer in architecture_:
            # If is tuple -> is conv2d layer, so unpack the layer contents
            if (isinstance(layer, tuple)):
                kernel_size, out_channels, stride, padding = layer
                conv2d = CNN_Block(
                    in_channels_, 
                    out_channels,
                    kernel_size,
                    stride, 
                    padding
                )
                sequential += [conv2d]
                in_channels_ = out_channels
            # If is str -> is maxpool2d layer, so append maxpool2d layer
            elif (isinstance(layer, str)):
                maxpool2d = nn.MaxPool2d(2, 2)
                sequential += [maxpool2d]
            # If is list -> is repeating sequence of convolutional layers
            elif (isinstance(layer, list)):
                repetitions_layers = []
                n_repetitions = layer[-1]
                for layer_ in layer[:-1]:
                    kernel_size, out_channels, stride, padding = layer_
                    conv2d = CNN_Block(
                        in_channels_, 
                        out_channels,
                        kernel_size,
                        stride, 
                        padding
                    )
                    repetitions_layers.append(conv2d)
                    in_channels_ = out_channels
                sequential += repetitions_layers * n_repetitions
                    
        return nn.Sequential(*sequential), in_channels_
    
    def build_connected_(self):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.last_out_channels * self.grid_size ** 2, 496), 
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1), 
            nn.Linear(496, (self.n_bounding_boxes * 5 + self.n_classes) * self.grid_size ** 2),
        )
