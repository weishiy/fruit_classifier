# Image Classification Project

## Overview
This project involves building, training, and evaluating a Convolutional Neural Network (CNN) for image classification. The dataset contains 6,000 RGB images across 3 classes: tomato, cherry, and strawberry. The main objectives are to understand the image classification pipeline, implement a CNN, and explore various techniques to improve model performance.

## Project Structure
- **train.py**: Script to build and train the CNN model. The trained model is saved as `model.pth`.
- **test.py**: Script to load the trained model from `model.pth` and evaluate its performance on unseen test data.
- **model.pth**: The trained CNN model.
- **testdata/**: Directory where the test images are placed for evaluation by `test.py`.
- **report.pdf**: A detailed report describing the methodology, experiments, and results.

## Important Notes
- **Do not move or rename files or directories after the initial setup**: Moving files or directories to another location after setting up the project may cause errors. It is recommended to keep the project structure intact as provided.
- Ensure all code runs smoothly on ECS School machines.
- Report should not exceed 10 pages, including figures and tables.


## Important Links
- [PyTorch CIFAR10 Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Torchvision Image Loading](https://pytorch.org/vision/stable/io.html#image)
- [Saving and Loading Models in PyTorch](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
