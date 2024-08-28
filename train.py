#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import GaussianBlur
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.nn as nn
import time
from torch.optim.lr_scheduler import StepLR
from torchvision import models
import warnings
import random


# In[2]:

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore", category=UserWarning)

class CNN_Improved(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Improved, self).__init__()
        # Load a pre-trained ResNet-18 model
        resnet18 = models.resnet18(pretrained=True)
        
        # Extract the feature extraction layers from the ResNet-18 model
        self.features = nn.Sequential(*list(resnet18.children())[:-2])
        
        # Add an adaptive average pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d(1)

         # Add a fully connected layer for classification with 'num_classes' output units
        self.fc = nn.Linear(512, num_classes)  

    def forward(self, x):
        # Forward pass through the model
        x = self.features(x)        # Extract features using pre-trained layers
        x = self.avgpool(x)         # Apply adaptive average pooling
        x = x.view(x.size(0), -1)   # Flatten the feature map
        x = self.fc(x)               # Classify using fully connected layer
        return x


# In[3]:


# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Load dataset
dataset = datasets.ImageFolder(root=r'D:\Users\shiya\done\COMP309\project\code\traindata', transform=transform)

# Split the dataset into training and test sets
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders for training and testing
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[4]:


def train(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    since = time.time()
    # Create lists to store loss and accuracy values
    loss_history = []
    accuracy_history = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_corrects = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            total_corrects += torch.sum(preds == labels)
            running_loss += loss.item() * inputs.size(0)
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = total_corrects.double() / total

        print(f'Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

        # Append loss and accuracy to the history lists
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_acc)

        scheduler.step()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    # save model
    torch.save(model.state_dict(), 'model.pth')

    # Return loss and accuracy history
    return time_elapsed, loss_history, accuracy_history




# In[5]:


# Initialize the model, loss function, and optimizer
num_classes = 3
model = CNN_Improved(num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
num_epochs = 10


# In[6]:


cnn_training_time, cnn_loss_history, cnn_accuracy_history = train(model, train_loader, criterion, optimizer, scheduler, num_epochs)

