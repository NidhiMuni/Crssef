import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import argparse
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython import embed
from torch.utils.data import Dataset
import pandas as pd
from skimage import io, transform
import model
import torchvision.transforms as T
from mammogram_dataset import MammogramDataset
import PIL
import torch.optim as optim

class Rescale(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, image):
        return transform.resize(image, (self.output_size, self.output_size))