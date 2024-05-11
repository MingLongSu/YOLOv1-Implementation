import os
import torch
import argparse 
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

class PascalVOCDataset(Dataset):
    """Class defining dataset for the Pascal VOC 2014 challenge"""

    def __init__(self, args: argparse.Namespace, transform=None):
        self.metaset = pd.read_csv(args.metaset_path, names=['imgs', 'lbls'])
        self.path_to_imgs = args.imgs_path
        self.path_to_lbls = args.lbls_path
        self.grid_size = args.grid_size
        self.n_bounding_boxes = args.n_bounding_boxes
        self.n_classes = args.n_classes
        self.mean = args.dataset_mean
        self.std = args.dataset_std
        self.transform = transform

    def __len__(self):
        return self.metaset.shape[0]

    def __getitem__(self, index: int):
        # Getting path to label data and reading label
        col_names = ['class', 'norm_x', 'norm_y', 'norm_w', 'norm_h']
        lbls_path = os.path.join(self.path_to_lbls, self.metaset.iloc[index]['lbls'])
        lbls_data = pd.read_csv(lbls_path, names=col_names, delim_whitespace=True)

        # Creating target tensor using data from labels
        y = torch.zeros(
            (self.grid_size, self.grid_size, self.n_classes + 5 * self.n_bounding_boxes),
            dtype=torch.float32
        )

        # Iterate over labels and add to target tensor
        for row_index in range(lbls_data.shape[0]):
            # Define unnormalized x and y coordinates with respect to grid
            unnorm_x = int(self.grid_size * lbls_data.iloc[row_index]['norm_x'])   
            unnorm_y  = int(self.grid_size * lbls_data.iloc[row_index]['norm_y'])   

            # Define class for the bounding box for label at currently row index
            lbl_class = lbls_data.iloc[row_index]['class']
            
            # Add new bounding box label if not currently occupied at grid location
            if (y[unnorm_y, unnorm_x, 24] == 0):
                y[unnorm_y, unnorm_x, 24] = 1
                y[unnorm_y, unnorm_x, 29] = 1
                lbl_coords = torch.tensor(
                    lbls_data.iloc[row_index][col_names[1:]].values
                )
                y[unnorm_y, unnorm_x, 20:24] = lbl_coords
                y[unnorm_y, unnorm_x, 25:29] = lbl_coords
                y[unnorm_y, unnorm_x, int(lbl_class)] = 1
        
        # Getting path to image data and reading image
        img_path = os.path.join(self.path_to_imgs, self.metaset.iloc[index]['imgs'])
        X = plt.imread(img_path)
        X = torch.tensor(X, dtype=torch.float32).permute(2, 0, 1)

        # Applying transformations
        if (self.transform != None):
            X, y = self.transform(X, y)
            X = X / 255
            X = (X - torch.tensor(self.mean)[:, None, None]) / torch.tensor(self.std)[:, None, None]
        return X, y
