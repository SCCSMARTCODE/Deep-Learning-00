# LoadCifar10

This project provides a Python class to load the CIFAR-10 dataset. The class `LoadCifar10` can download, extract, and load the dataset for use in machine learning projects. It supports custom transformations on the images and can differentiate between training and test datasets.

## Features

- Download the CIFAR-10 dataset.
- Extract the dataset from a tar file.
- Load and transform images for training or testing.
- Support for custom image transformations.

## Requirements

- Python 3.6 or higher
- `requests` library
- `tarfile` library
- `os` library

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/yourrepository.git
    ```
2. Navigate to the project directory:
    ```sh
    cd yourrepository
    ```

## Usage

Here's an example of how to use the `LoadCifar10` class:

```python
import os
from torchvision.transforms import ToTensor

class LoadCifar10:
    """
    This class loads our Cifar 10 dataset.
    """
    file_path = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_name = "zip_cifar_10.tar.gz"

    def __init__(self, root, download=False, transforms=None, train=True):
        self.root = root
        self.download = download
        self.transforms = []
        if transforms:
            try:
                self.transforms.extend(transforms)
            except TypeError:
                self.transforms.append(transforms)
        self.train = train
        self.dataset_folder = os.path.join(self.root, "cifar-10-batches-py")
        self.dataset = []

        if not root:
            return

        # Download if self.download is True
        if self.download:
            self.download_dataset()

        # Loading dataset
        if not os.path.exists(self.dataset_folder):
            print("File doesn't exist\nSet download to True to download it")
            return

        # Locate the necessary file based on requirement
        if self.train:
            file_names = [x for x in os.listdir(self.dataset_folder) if "data_batch" in x]
        else:
            file_names = [x for x in os.listdir(self.dataset_folder) if "test_batch" in x]

        # Convert the files to usable data
        for file_name in file_names:
            out = self.unpickle(os.path.join(self.dataset_folder, file_name))
            batch_images = list(out.get(b'data'))
            batch_images = map(self.format_image, batch_images)

            batch_dataset = list(zip(batch_images, out.get(b'labels')))

            self.dataset.extend(batch_dataset)

    def download_dataset(self):
        """
        This is the function that downloads and extracts the CIFAR-10 dataset from the main website.
        """
        if os.path.exists(self.dataset_folder):
            print("File exists...")
            return
        response = requests.get(self.file_path)
        if response.status_code != 404:
            downloaded_file_path = os.path.join(self.root, self.file_name)
            with open(downloaded_file_path, 'wb') as f:
                f.write(response.content)
        else:
            print("File Not Found")
            exit(-1)
        print("CIFAR-10 Downloaded Successfully...")

        # Extract the tar file
        with tarfile.open(downloaded_file_path, "r:gz") as f:
            f.extractall(self.root)

        # Delete the tar file
        os.remove(downloaded_file_path)

        print("CIFAR-10 Extracted Successfully...")

    def unpickle(self, file):
        """
        This function helps in converting compressed file into usable dictionary.
        """
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def format_image(self, image, size=(3, 32, 32)):
        """
        This function helps us with transforming our image.
        """
        image = image.reshape(size)

        if self.transforms:
            for transform in self.transforms:
                image = transform(image)
        return image

# Usage example:
load_cifar10 = LoadCifar10(root="/content/drive/MyDrive/Deep Learning/cifar_10", train=True, download=True)
dataset = load_cifar10.dataset
```


## Example

```python
from torchvision.transforms import ToTensor

# Create an instance of the LoadCifar10 class
load_cifar10 = LoadCifar10(root="/path/to/dataset", download=True, transforms=[ToTensor()], train=True)

# Access the dataset
dataset = load_cifar10.dataset

# Display the first image and its label
image, label = dataset[0]
print(f"Label: {label}")
print(f"Image shape: {image.shape}")
```
