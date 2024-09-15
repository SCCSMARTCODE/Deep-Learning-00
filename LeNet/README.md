---

# LeNet Architecture for MNIST Classification

This repository contains an implementation of the LeNet architecture, designed for the classification of the MNIST dataset. The LeNet model here has been modified to accept 28x28 grayscale images and is trained to classify digits (0-9).

## Project Overview

LeNet, one of the earliest convolutional neural networks, was developed for document recognition and is particularly effective at image classification tasks. This implementation uses PyTorch and includes techniques such as dropout and weight decay to improve generalization and prevent overfitting.

### Model Architecture

The model follows a sequential design:

- **Convolutional Layers**:
  - Conv1: 1x32x32 input -> 32 feature maps (3x3 kernel)
  - Conv2: 32 feature maps -> 64 feature maps (3x3 kernel)
  - Conv3: 64 feature maps -> 128 feature maps (3x3 kernel)
  - Each conv layer is followed by ReLU activation, batch normalization, and max pooling.

- **Fully Connected Layers**:
  - Flatten the output of the convolutional layers and pass it through:
    - A 512x64 layer with a dropout of 0.3.
    - A 64x10 output layer for final classification.

### Results

After training for 10 epochs, the model achieved **99.16% accuracy** on the test set, with a **loss of 0.0253**. The model is saved in a `.pth` format for future inference.

## Training Setup

Here are the key parameters used during training:

- **Epochs**: 10
- **Batch Size**: 128
- **Learning Rate**: 1e-4 (max_lr = 1e-2)
- **Optimizer**: Adam with weight decay = 0.002
- **Learning Rate Scheduler**: OneCycleLR

## Visualizations and Logs

To monitor the model's training, I utilized Weights and Biases (W&B) for real-time logging and visualization.

You can view the full training and validation logs with the following link:  
[W&B Experiment Logs](https://wandb.ai/sccsmartcode-prometheus-/LeNet/runs/9m3w7fyi?nw=nwusersccsmartcode)

## Explanation of Regularization Techniques

I have incorporated two key regularization techniques in this project to avoid overfitting:

1. **Dropout**: Dropout was used during training to randomly deactivate neurons and prevent the network from becoming too reliant on specific neurons.  
   For more details, check out my blog post on **Dropout**:  
   [Understanding Dropout in Neural Networks](https://medium.com/@sccsmart247/understanding-dropout-in-deep-learning-intuition-theory-and-practicality-61d407f14282)

2. **Weight Decay**: To penalize large weights and further improve generalization, weight decay was applied during training.  
   You can read about weight decay and its importance in my Medium post:  
   [A Comprehensive Guide to Weight Decay in Neural Networks](https://medium.com/@sccsmart247/understanding-weight-decay-in-deep-learning-the-why-the-how-and-the-impact-2f0c88ce69da)

## How to Use

### Prerequisites

- PyTorch
- torchvision
- torchsummary
- Weights and Biases (optional, for experiment tracking)

### Installation

Clone the repository and install the required libraries:
```bash
git clone https://github.com/SCCSMARTCODE/Deep-Learning-00
cd LeNet
```


### Saving and Loading the Model

To save the trained model:
```python
torch.save(network.state_dict(), 'parameter.pth')
```

To load the model for inference:
```python
network = LeNet()
network.load_state_dict(torch.load('parameter.pth'))
network.eval()
```


## References

- **LeNet Paper**: [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

---
