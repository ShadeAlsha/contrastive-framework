import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, idx  # Return the original index along with data

    def __len__(self):
        return len(self.dataset)

def get_mnist_dataloaders(digits=(0, 1, 2, 3, 4), batch_size=5000, data_dir='data'):
    # Define a filter function to keep only the desired digits
    def filter_mnist(dataset, digits):
        indices = [i for i, (x, y) in enumerate(dataset) if y in digits]
        return Subset(IndexedDataset(dataset), indices)  # Wrap dataset to include indices

    # Define a transformation that includes flattening
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: torch.flatten(x))  # Flatten the tensor
    ])

    # Load the MNIST dataset and apply the transformation
    mnist_train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # Filter the datasets
    mnist_train_filtered = filter_mnist(mnist_train, digits)
    mnist_test_filtered = filter_mnist(mnist_test, digits)

    # Create DataLoaders
    train_loader = DataLoader(mnist_train_filtered, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_test_filtered, batch_size=len(mnist_test_filtered), shuffle=True)

    return train_loader, test_loader
