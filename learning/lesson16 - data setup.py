"""
Dataset structure example

imgs -> folder with an assload of images
labels.csv - file that contains teh image labels

labels.csv: 
img1.jpg, 0
img2.jpg, 1
img3.jpg, 0
img4.jpg, 2

labels:
0 = dog
1 = cat
2 = yo momma
"""

import os
import pandas as pd
from torchvision import read_image
from torch.utils.data import Dataset


class OurDataset(Dataset):
    def __init__(self, labels_path, imgs_dir, transform=None):
        super().__init__()
        self.imgs_dir = imgs_dir
        self.transform = transform
        self.labels = pd.read_csv(labels_path)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_path = os.join(self.imgs_dir, self.labels.iloc[index, 0])
        # iloc=index location
        image = read_image(img_path)
        label = self.labels.iloc[index, 1]

        # apply transformations
        if self.transform:
            image = self.transform(image)

        return image, label
