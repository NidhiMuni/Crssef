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
from utils import Rescale

######################################
############## Method ################
######################################

def train(epoch):
    running_loss = 0.0
    correct = 0
    for iteration, data in enumerate(loader_train):
        # get the inputs; data is a list of [inputs, labels]
        image = data['image']
        label = data['label']
        image = image.to(device=device, dtype=dtype)
        label = label.to(device=device, dtype=torch.long)

        # zero the parameter gradients
        optimizer.zero_grad()

        scores = neuralNet.forward(image)
        scores = torch.squeeze(scores)
        loss = criterion(scores, label)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        
        # compute the accuracy
        pred = scores.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(label.data).cpu().sum()

        if iteration % batch_size == 0:
            print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (iteration+1) * len(data), len(loader_train.dataset),
                100. * (iteration+1)*len(data) / len(loader_train.dataset), loss.item()))

    train_accuracy = correct / float(len(loader_train.dataset))
    print("Training accuracy ({:.2f}%)".format(100*train_accuracy))
    return (train_accuracy*100.0)

def validate(epoch):
    correct = 0
    neuralNet.eval()
    global best_accuracy
    for iteration, data in enumerate(loader_val):
        # get the inputs; data is a list of [inputs, labels]
        image = data['image']
        label = data['label']
        image = image.to(device=device, dtype=dtype)
        label = label.to(device=device, dtype=torch.long)
        
        # do the forward pass
        scores = neuralNet.forward(image)
        scores = torch.squeeze(scores)
        pred = scores.data.max(1)[1] # got the indices of the maximum, match them
        correct += pred.eq(label.data).cpu().sum()

    print("Predicted {} out of {}".format(correct, iteration * batch_size))
    epoch_accuracy = correct / (iteration * batch_size) * 100
    print("Accuracy = {:.2f}".format(epoch_accuracy))

    # now save the model if it has better accuracy than the best model seen so forward
    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        # save the model
        torch.save(neuralNet.state_dict(),'saved_model.pth')
    return epoch_accuracy

######################################
############### Main #################
######################################
# default settings
learning_rate = 0.001
batch_size = 4
num_classes = 2
log_schedule = 10
epochCount = 5
num_workers = 1
print_every = batch_size

transform = T.Compose([
                Rescale(32),
                T.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

train_data = MammogramDataset("Mini_DDSM_Upload", "train", transform=transform)
test_data = MammogramDataset("Mini_DDSM_Upload", "test")

VAL_RATIO = 0.2
NUM_VAL = int(len(train_data)*VAL_RATIO)
NUM_TRAIN = len(train_data) - NUM_VAL
NUM_TEST = len(test_data)
BATCH_SIZE = batch_size


loader_train = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)), drop_last=True)
loader_val = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, NUM_TRAIN + NUM_VAL)))
loader_test = DataLoader(test_data, batch_size=BATCH_SIZE)

dtype = torch.float32
device = torch.device('cpu')

epoch = 0
loss_list = []
val_acc_list = []
best_accuracy = 0.0

neuralNet = model.SqueezeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(neuralNet.parameters(), lr=0.001, momentum=0.9)


for epoch in range(epochCount):
    train(epoch)
    validate(epoch)

print(loss_list)

