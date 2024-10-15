# CIFAR-10 Image Classification using CNNs

This repository demonstrates how to build and train a Convolutional Neural Network (CNN) to classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html). The CIFAR-10 dataset consists of 60,000 32x32 color images across 10 different classes (airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks).

## Table of Contents

- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## Dataset

The CIFAR-10 dataset is composed of:
- 60,000 32x32 color images
- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training images and 10,000 test images

You can download the CIFAR-10 dataset directly from [Keras](https://keras.io/api/datasets/cifar10/) or from [this link](https://www.cs.toronto.edu/~kriz/cifar.html).

## Model Architecture

The Convolutional Neural Network (CNN) used in this project follows a typical architecture:

1. **Input Layer**: 32x32 RGB images.
2. **Convolutional Layers**: Multiple layers with filters, followed by activation (ReLU) and pooling (MaxPooling).
3. **Dropout**: Used for regularization to avoid overfitting.
4. **Flatten Layer**: Converts 2D data to 1D for fully connected layers.
5. **Fully Connected Layers (Dense)**: One or more dense layers with activation (ReLU).
6. **Output Layer**: 10 units (one for each class) with softmax activation for classification.

### Model Summary

| Layer Type       | Details                              |
|------------------|--------------------------------------|
| Conv2D           | 32 filters, 3x3 kernel, ReLU         |
| MaxPooling2D     | 2x2 pool size                        |
| Conv2D           | 64 filters, 3x3 kernel, ReLU         |
| MaxPooling2D     | 2x2 pool size                        |
| Flatten          | Flattening layer                     |
| Dense            | 512 units, ReLU                      |
| Dropout          | 0.5 dropout rate                     |
| Dense            | 10 units (softmax activation)        |

## Installation

To get started with this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/cifar-10-using-cnns.git
cd cifar-10-using-cnns
pip install -r requirements.txt
```

## Dependencies

The project requires the following dependencies:

- **TensorFlow/Keras**: Used for building and training the CNN.
- **NumPy**: For numerical operations and array handling.
- **Matplotlib**: For plotting and visualizing the model's predictions.

Make sure to install these libraries by running:

```bash
pip install -r requirements.txt
```
## Usage (Continued)

### Training Parameters

You can customize several training parameters to optimize the model performance:

- **Batch Size**: Set the number of samples processed before the model is updated. Common values are 32, 64, or 128.
- **Epochs**: Define how many times the learning algorithm will work through the entire training dataset. Increasing the number of epochs can improve accuracy but may also lead to overfitting.
- **Learning Rate**: Adjust the rate at which the model learns. You can use learning rate schedulers to adapt the learning rate during training.

To modify these parameters, edit the corresponding values in the Jupyter notebook or in the `train_cifar10.py` script.

### Evaluating the Model

After training the model, you can evaluate its performance on the test dataset. The notebook provides a section to calculate and display the model's accuracy on unseen data. You can also view a confusion matrix to better understand the model's performance across different classes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
