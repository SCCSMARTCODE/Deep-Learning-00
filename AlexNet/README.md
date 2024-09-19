
---

# AlexNet for Cat and Dog Classification

This repository contains an implementation of the AlexNet architecture, designed for the binary classification of the **Cat and Dog** dataset from Kaggle. The AlexNet model has been customized to work with 227x227 RGB images and uses various data augmentation techniques to improve generalization. This implementation also includes optimizations like dropout and weight decay to minimize overfitting and enhance performance.

## Project Overview

**AlexNet** is a convolutional neural network architecture that became famous for its performance in the 2012 ImageNet competition. With a deep design and a large number of parameters, AlexNet is highly effective for image classification tasks. In this project, I used the **Cat and Dog** dataset to fine-tune and optimize AlexNet for distinguishing between these two categories of images.

This project leverages PyTorch, with regularization methods like **data augmentation**, **dropout**, and **weight decay** to maximize accuracy and prevent overfitting.

## Model Architecture

The AlexNet architecture follows a five-convolutional layer design with fully connected layers at the end:

### Convolutional Layers:
- **Conv1**: 3x227x227 input -> 96 feature maps (11x11 kernel, stride=4, padding=2)
- **Conv2**: 96 feature maps -> 256 feature maps (5x5 kernel, stride=1, padding=2)
- **Conv3**: 256 feature maps -> 384 feature maps (3x3 kernel, stride=1, padding=1)
- **Conv4**: 384 feature maps -> 384 feature maps (3x3 kernel, stride=1, padding=1)
- **Conv5**: 384 feature maps -> 256 feature maps (3x3 kernel, stride=1, padding=1)
  
Each convolutional layer is followed by **ReLU activation**, **batch normalization**, and **max pooling** layers to reduce spatial dimensions.

### Fully Connected Layers:
- **Flattened Output** from Conv layers.
- **FC1**: 4096 neurons with dropout (0.5 probability).
- **FC2**: 4096 neurons with dropout (0.5 probability).
- **FC3**: 1 neurons (final output layer for binary classification).

## Results

After training the model for 30 epochs, it achieved:
- **Validation Accuracy**: 82.52%
- **Test Accuracy**: 83.14%
- **Test Loss**: 0.3608

The model is saved in `.pth` format for future inference and experimentation.

## Training Setup

The following parameters were used during training:

- **Total Training Epochs**: 30
- **Batch Size**: 128
- **Learning Rate**: 1e-5 (using OneCycleLR scheduler with max_lr=1e-3)
- **Optimizer**: Adam with **weight decay** = 0.008
- **Loss Function**: Binary Cross Entropy (BCE)

## Regularization and Augmentation

To prevent overfitting, I employed several regularization techniques:

1. **Dropout**: Applied with a 50% probability in fully connected layers to avoid reliance on specific neurons.
2. **Weight Decay**: Applied during training to regularize the model and improve generalization.
3. **Data Augmentation**: I used augmentations like random flips, blurring, brightness adjustments, etc., to increase dataset variety and improve model generalization.

## Visualizations and Logs

I used **Weights and Biases (W&B)** to track the training process, including real-time metrics like loss, accuracy, and learning rate trends.

You can check out the complete training logs and visualizations through this [W&B Experiment Logs](https://wandb.ai/sccsmartcode-prometheus-/AlexNet).

## How to Use

### Prerequisites

- PyTorch
- torchvision
- torchsummary
- Weights and Biases (optional for experiment tracking)

### Installation

1. Clone the repository and install the required libraries:

    ```bash
    git clone https://github.com/YourUsername/AlexNet-CatsDogs
    cd AlexNet-CatsDogs
    pip install -r requirements.txt
    ```

### Saving and Loading the Model

To save the trained model:
[download parameter](https://drive.google.com/file/d/1bG6YLLMaiFKZYXPUHxg_2WmcdSz7-wqm/view?usp=sharing)

```python
torch.save(model.state_dict(), 'parameter.pth')
```

To load the model for inference:

```python
model = AlexNet()
model.load_state_dict(torch.load('alexnet_catsdogs.pth'))
model.eval()
```

## Inference

For running inference on test data:

```python
run_inference(network, test_loader, criterion)
```

## References

- **AlexNet Paper**: [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- **Kaggle Dataset**: [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)

---
