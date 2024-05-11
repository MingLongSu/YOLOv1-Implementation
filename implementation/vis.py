import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd

from utils import PascalVOCClasses, non_max_suppression

def view_image(img_name: str, lbl_name: str, classes: PascalVOCClasses, hat_y: torch.Tensor=None):
    '''
    Presents image with bounding boxes and labels

    Parameters
    -------
    img_name : str 
        Name used for image file (including file type)
    lbl_name : str
        Name used for label file (including file type)
    '''

    # Init setup
    pwd = os.getcwd()

    # Reading sample image
    img_file_path = os.path.join(pwd, 'data', 'images', img_name)
    img = plt.imread(img_file_path)

    # Getting dimensions of image
    img_height, img_width, _ = img.shape

    # Reading label data of image
    lbl_file_path = os.path.join(pwd, 'data', 'labels', lbl_name)
    df_label = pd.read_csv(
        lbl_file_path, 
        delim_whitespace=True, 
        names=['class', 'x-coord', 'y-coord', 'width', 'height']
    )
    
    # Configuring plot to include bounding boxes and class names
    fig, axes = plt.subplots()
    axes.imshow(img)
    for _, row in df_label.iterrows():
        # Setting aside coord data
        obj_class_name = classes()[row['class']]
        coord_x = row['x-coord']*img_width 
        coord_y = row['y-coord']*img_height
        bb_width = row['width']*img_width
        bb_height = row['height']*img_height

        # Adding bounding box to shown image
        rect = patches.Rectangle(
            (coord_x - bb_width / 2, coord_y - bb_height / 2),
            bb_width, bb_height,
            linewidth=1, edgecolor='r', facecolor='none'
        )
        axes.add_patch(rect)

        # Adding text label to shown image
        plt.text(
            coord_x - bb_width / 2, coord_y - bb_height / 2,
            obj_class_name, bbox=dict(facecolor='red')
        )
        
    plt.show()



def compare_image(hat_y: torch.Tensor, y: torch.Tensor, classes: PascalVOCClasses):
    pass
