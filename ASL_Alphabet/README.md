# American Sign Language Alphabet Recognition

This project aims to develop a model that can recognize and classify images of the American Sign Language (ASL) alphabet with high accuracy. The dataset used for this task includes a comprehensive collection of images representing the 26 letters of the alphabet, as well as symbols for SPACE, DELETE, and NOTHING.

## Table of Contents
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training and Validation](#training-and-validation)
- [Results](#results)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## Dataset

The dataset contains the following details:

- **Number of training images:** 87,000
- **Image size:** 200x200 pixels
- **Number of classes:** 29 (26 letters A-Z, SPACE, DELETE, and NOTHING)
- **Number of test images:** 29 (to encourage real-world test scenarios)

## Model Architecture

The model architecture is designed to effectively capture the features of ASL images and perform classification with high accuracy. The architecture includes:

- **Input Layer:** Accepts 200x200 pixel images.
- **Convolutional Layers:** Multiple layers to extract spatial features, followed by ReLU activation and Batch Normalization.
- **Pooling Layers:** MaxPooling layers to reduce dimensionality while preserving important features.
- **Residual Blocks:** Added for better gradient flow and to prevent vanishing gradient problems.
- **Fully Connected Layers:** To learn non-linear combinations of the features.
- **Output Layer:** Softmax activation to classify the images into one of the 29 classes.

## Training and Validation

### Data Pipeline

1. **Data Augmentation:** Techniques such as horizontal flipping and rotation were applied to enhance the robustness of the model.
2. **Normalization:** Image pixel values were normalized to have a mean of 0 and a standard deviation of 1.

### Training Parameters

- **Optimizer:** Adam with weight decay
- **Learning Rate:** 0.001
- **Weight Decay:** 0.85
- **Batch Size:** 32
- **Epochs:** 50
- **Loss Function:** Cross-Entropy Loss
- **Mixed Precision Training:** Using `torch.cuda.amp` for faster and more efficient training

### Training Script

```python
import torch
from torch.cuda.amp import autocast, GradScaler

def fit(epochs, train_dl, val_dl, optimizer, criterion, scheduler, model):
    train_losses = []
    val_losses = []
    val_accuracies = []
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for inputs, labels in train_dl:
            inputs = inputs.to(device).half()
            labels = labels.to(device)

            optimizer.zero_grad()

            with autocast():
                pred = model(inputs)
                loss = criterion(pred, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        train_loss /= len(train_dl)
        train_losses.append(train_loss)

        val_loss = evaluate(val_dl, model, criterion)
        val_accuracy = accuracy(val_dl, model)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
    
    return train_losses, val_losses, val_accuracies
```

## Results

After 8 epochs of training, the model achieved an impressively low training loss and high validation accuracy, indicating that the model is performing exceptionally well on the dataset.

### Training Loss and Validation Loss

  - **Epoch 8/8:**
  - **Train Loss:** 0.0014
  - **Val Loss:** 0.0002
  - **Val Accuracy:** 100.00%

## Usage

To use the model, follow these steps:

1. **Clone the repository:**

2. **Install dependencies:**
   
3. **Download Pre-trained Model:**
   You can download the pre-trained model parameters from [https://drive.google.com/file/d/1mkNPKMAe23mBXM7HYj-u_H7U7QVCzt8e/view?usp=sharing](URL to the model file).

4. **Run the model:**
   
## Conclusion

This project demonstrates the potential of deep learning in recognizing the ASL alphabet with high accuracy. The model can be further improved and fine-tuned with more data and advanced techniques.

## Contact

For any questions or suggestions, please feel free to contact me at sccsmart247@gmail.com.

## Acknowledgments

Special thanks to [FOOTPRINTWORLD AI].
