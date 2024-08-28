#!/usr/bin/env python
# coding: utf-8

# In[13]:


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


if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Disable UserWarnings for clarity
warnings.filterwarnings("ignore", category=UserWarning)

# Define your CNN_Improved class here
class CNN_Improved(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Improved, self).__init__()
        # 
        resnet18 = models.resnet18(pretrained=True)
        
        # 
        self.features = nn.Sequential(*list(resnet18.children())[:-2])
        
        # 
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)  # 

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# Define data transformations and load test dataset
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
batch_size = 32
test_dataset = datasets.ImageFolder(root=r'D:\Users\shiya\done\COMP309\project\code\testdata', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Initialize your CNN_Improved model
num_classes = 3
model = CNN_Improved(num_classes)  # Make sure to define the number of classes correctly

# Load the trained model weights
model.load_state_dict(torch.load(r'D:\Users\shiya\done\COMP309\project\code\model.pth'))
model.eval()

# Function to test the model
def test_model(model, test_loader):
    model.eval()
    all_predictions = []
    all_labels = []

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy

# Evaluation on the test set
test_accuracy = test_model(model, test_loader)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

