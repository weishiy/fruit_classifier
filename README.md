# Image Classification Project

## Overview
This project involves building, training, and evaluating a Convolutional Neural Network (CNN) for image classification. The dataset contains 6,000 RGB images across 3 classes: tomato, cherry, and strawberry. The main objectives are to understand the image classification pipeline, implement a CNN, and explore various techniques to improve model performance.(This is for course COMP309).

## Project Structure
- **train.py**: Script to build and train the CNN model. The trained model is saved as `model.pth`.
- **test.py**: Script to load the trained model from `model.pth` and evaluate its performance on unseen test data.
- **model.pth**: The trained CNN model.
- **testdata/**: Directory where the test images are placed for evaluation by `test.py`.
- **report.pdf**: A detailed report describing the methodology, experiments, and results.

## Important Notes
- **Do not plagiarize**: Please do not copy this project. I encountered an issue where I moved the project from the original C: drive to the D: drive, and the test accuracy became significantly lower than before. The accuracy is now inconsistent with the original results, and I haven't had the time to figure out the cause of this error yet.


## Important Links
- [PyTorch CIFAR10 Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [Torchvision Image Loading](https://pytorch.org/vision/stable/io.html#image)
- [Saving and Loading Models in PyTorch](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
