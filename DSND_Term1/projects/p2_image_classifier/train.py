# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 23:57:32 2018

@author: Narendra.Sahu
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import vgg19, densenet121, vgg16
from torch import nn, optim
import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from collections import OrderedDict
import torchvision

import time
import json
import copy
import seaborn as sns
import numpy as np
from PIL import Image

from torch.autograd import Variable
from torchvision import datasets, models, transforms
from PIL import Image

arch = 'vgg13' alternatively
learning_rate = 0.0001
epochs = 10
GPU = True
data_dir = 'flowers'

if GPU == True:
    device = 'cuda'
else:
    device = 'cpu'

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

data_transforms = {'train' : transforms.Compose([transforms.RandomRotation(30),
                                                 transforms.RandomResizedCrop(224),                                                 
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225]),
                                                 ]),
                   
                   'valid' : transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225])
                                                ]),
                   
                   'test' : transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                      [0.229, 0.224, 0.225]),
                                               ])
                  }
                                                

# TODO: Load the datasets with ImageFolder
image_datasets = {'train' : datasets.ImageFolder(train_dir, transform=data_transforms['train']),
                  'valid' : datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
                  'test' : datasets.ImageFolder(test_dir, transform=data_transforms['test'])
                 }

# TODO: Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'train' : DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
               'valid' : DataLoader(image_datasets['valid'], batch_size=32),
               'test' : DataLoader(image_datasets['test'], batch_size=32)
              }

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    


def validation(model, testloader, criterion):
    test_loss = 0
    accuracy = 0
    model.to(device)
    model.eval()
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)

        output = model(inputs)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss/len(testloader), accuracy/len(testloader)

classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear((25088 if arch == 'vgg19' else 1024),4096)),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(0.2)),
    ('fc2', nn.Linear(4096,1024)),
    ('relu2', nn. ReLU()),
    ('dropout2', nn.Dropout(0.2)),
    ('fc3', nn.Linear(1024,102)),
    ('output', nn.LogSoftmax(dim=1))]))

model = getattr(models, arch)(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
model.classifier = classifier
if GPU == True:
    model.cuda()
else:
    model.cpu()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
model.train()

print_every = 40
steps = 0

for epoch in range(epochs):
    accuracy = 0
    running_loss = 0.0
    for inputs, labels in dataloaders['train']:

        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        output = model(inputs)
        _, preds = torch.max(output.data, 1)
        loss = criterion(output, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
        if steps % print_every == 0:
            test_loss, test_accuracy = validation(model, dataloaders['valid'], criterion)
            print("Epoch: {}/{}".format(epoch+1, epochs),
                  "Train Loss: {:.4f}".format(running_loss/print_every),
                  "Train Accuracy : {:.4f}".format(accuracy/print_every),
                  "Validation Loss : {:.4f}".format(test_loss),
                  "Validation Accuracy : {:.4f}".format(test_accuracy))
            model.train()
            accuracy = 0
            running_loss = 0
            
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'arch' : arch,
              'classifier' : model.classifier,
              'state_dict' : model.state_dict(),
              'optimizer' : optimizer,
              'optimizer_dict' : optimizer.state_dict(),
              'epochs' : epochs,
              'class_to_idx' : model.class_to_idx}

torch.save(checkpoint, 'checkpoint.pth')

def load_checkpoint(checkpoint):
    checkpoint = torch.load(checkpoint)
    model = getattr(torchvision.models, checkpoint['arch'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    for param in model.parameters():
        param.requires_grad = False
    model.load_state_dict(checkpoint['state_dict'])
    model.optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    return model

filepath = 'checkpoint.pth'
model = load_checkpoint(filepath)