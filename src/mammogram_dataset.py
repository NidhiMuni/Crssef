import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import sampler
import torchvision.datasets as dset
import torchvision.transforms as T
import numpy as np
import os
import PIL
import pandas as pd
from skimage import io

NUM_CLASSES = 2

class MammogramDataset(Dataset):
    def __init__(self, root_dir, dataset, transform = None):
        self.root_dir = root_dir
        self.dataset = dataset
        self.transform = transform
        csv_file = os.path.join(root_dir, dataset+".csv")
        self.datapoints = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, index):
        image_file = self.datapoints.iloc[index, 0]
        label = str(self.datapoints.iloc[index, 1])
        image_path = os.path.join(self.root_dir, self.dataset, image_file)
        image = io.imread(image_path)
        # image = np.expand_dims(image, 2)
                
        if self.transform:
            image = self.transform(image)
            
        # image = np.squeeze(image, 2)
            
        sample = {'image': image, 'label': int(label)}
        return sample
    
    def print_summary(self):
        print('Root dir:    ', self.root_dir)
        print('Dataset:     ', self.dataset)
        print('Datapoints:  ', self.__len__())

    def print_datapoint(self, index):
        print(self.__getitem__(index))
