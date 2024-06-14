import idx2numpy
import numpy as np
import pandas as pd
import os
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import softmax
import torch.utils.data
from torchvision import transforms
import torch.utils
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics 

import matplotlib.pyplot as plt

class MNISTDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        image = self.transform(np.array(self.images[idx]))
        label = self.labels[idx]
        return image, label

class MNISTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Architecture
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32*7*7,10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through network
        h1 = F.relu(self.conv1(x))
        h1 = self.max_pool(h1)
        h2 = F.relu(self.conv2(h1))
        h2 = self.max_pool(h2)
        h2 = h2.view(-1, 32 * 7 * 7)
        z3 = self.fc1(h2)

        # Returns predictions
        return z3

def load_data(x_filename: str, y_filename: str, batch_size: int = 32, train: bool = False) -> Union[DataLoader, tuple[DataLoader, DataLoader]]:
    x_arr = idx2numpy.convert_from_file("./data/" + x_filename)
    y_arr = idx2numpy.convert_from_file("./data/" + y_filename)
    dataset = MNISTDataset(x_arr, y_arr)
    if train:
        gen = torch.Generator().manual_seed(42)
        train_data, val_data = torch.utils.data.random_split(dataset, lengths=[0.8, 0.2], generator=gen)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def save_checkpoint(model: torch.nn.Module, epoch: int) -> None:
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
    }

    filename = os.path.join("./checkpoints/", "epoch={}.checkpoint.pth.tar".format(epoch))
    torch.save(state, filename)

def get_stats(
    train_data: DataLoader,
    val_data: DataLoader,
    model: nn.Module,
    criterion,
    stats):

    def _metrics(data: DataLoader) -> tuple[float, float, float]:
        model.eval()
        y_pred, y_true, y_score, loss = [], [], [], []
        total = 0
        correct = 0
        for image, label in data:
            with torch.no_grad(): # don't need to save gradients for stats
                output = model(image) # tensor with probabilities
                predicted_y = torch.argmax(output, dim=1) #returns 0-9 prediction
                y_pred.append(predicted_y)
                y_true.append(label)
                y_score.append(softmax(output.data, dim=1)) #ensures distr sums to 1
                total += label.size(0) # adds batch size
                correct += (predicted_y == label).sum().item() # adds however many correct from batch
                loss.append(criterion(output, label).item())

        acc = correct / total
        loss = np.mean(loss) #compute average loss over all datapoints
        y_true = torch.cat(y_true)
        y_score = torch.cat(y_score)
        auroc = metrics.roc_auc_score(y_true, y_score, multi_class='ovo')
        return acc, loss, auroc

    train_accuracy, train_loss, train_auroc = _metrics(train_data)
    val_accuracy, val_loss, val_auroc = _metrics(val_data)

    print("Training Accuracy: {:.4f}\nTraining Loss: {:.4f}\nTraining AUROC: {:.4f}\n".format(train_accuracy, train_loss, train_auroc))
    print("Validation Accuracy: {:.4f}\nValidation Loss: {:.4f}\nValidation AUROC: {:.4f}\n".format(val_accuracy, val_loss, val_auroc))
    
    epoch_stats = [
        train_accuracy,
        train_loss,
        train_auroc,
        val_accuracy,
        val_loss,
        val_auroc,
    ]

    stats.append(epoch_stats)


def main():
    batch_size = 32
    # Load dataset
    print("Loading training and test data...")
    train_data, val_data = load_data("train-images-idx3-ubyte", "train-labels-idx1-ubyte", batch_size, train=True)
    test_data = load_data("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", batch_size)


    print("Success!")
    
    # Train model
    print("Training model...")

    model = MNISTModel()
    criterion = torch.nn.CrossEntropyLoss() # best type of loss for multiclass
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # could use typical SGD, but more efficient

    num_epochs = 2

    #Create plot for visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    plt.suptitle("MNIST Training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUROC")

    stats = [] # Array to store info from each epoch to plot

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch}\n")
        # Train the model through forward prop and back prop
        model.train()
        for image, label in train_data:
            optimizer.zero_grad()
            predicted = model(image)
            loss = criterion(predicted, label)
            loss.backward()
            optimizer.step()

        # Get stats from current epoch
        get_stats(train_data, val_data, model, criterion, stats)

        # Update the graph
        splits = ["Train", "Validation"]
        metrics = ["Accuracy", "Loss", "AUROC"]
        colors = ['c', 'm', 'y']
        for i, _ in enumerate(metrics):
            for j, _ in enumerate(splits):
                idx = len(metrics) * j + i
                axes[i].plot(
                    range(epoch - len(stats) + 1, epoch + 1),
                    [stat[idx] for stat in stats],
                    linestyle="-",
                    marker="o",
                    color=colors[j]
                )
            axes[i].legend(splits)


        # Save current epoch params in case it's the one we want
        save_checkpoint(model, epoch)

    plt.savefig("training.png", dpi=200)

    print("Finished Training!\n")

    test_epoch = int(input(f"Enter an epoch to run the test set on (0-{num_epochs-1}): "))

    # restore the checkpoint

    # forward prop through the test dataset

    # get stats and update plot


if __name__ == "__main__":
    main()
