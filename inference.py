import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Define the same model architecture as used during training
class CNN_Improved(nn.Module):
    def __init__(self, num_classes):
        super(CNN_Improved, self).__init__()
        # Load a pre-trained ResNet-18 model
        resnet18 = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(resnet18.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load the trained model
num_classes = 3
model = CNN_Improved(num_classes)

# Load the saved model parameters
model.load_state_dict(torch.load('model.pth'))

# Set the model to evaluation mode
model.eval()

# Define the same data preprocessing as used during training
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the test dataset
test_dataset = ImageFolder(root=r'D:\Users\shiya\done\COMP309\project\code\testdata', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Perform inference and calculate accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')
