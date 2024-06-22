import idx2numpy
import numpy as np
import pandas as pd
import os
from typing import Union, Tuple
import random

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
from PIL import Image
from collections import Counter

class MNISTDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        image = self.transform(np.array(self.images[idx]))
        label = self.labels[idx]
        return image, label

class MNISTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Architecture
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32*6*6,10) # (W−K+2P)/S]+1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Forward pass through network
        h1 = F.relu(self.conv1(x))
        h1 = self.max_pool(h1)
        h1 = h1.view(-1, 32 * 6 * 6)
        z2 = self.fc1(h1)

        # Returns predictions
        return z2

def load_data(x_filename: str, y_filename: str, batch_size: int = 32, train: bool = False) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    x_arr = idx2numpy.convert_from_file("./data/" + x_filename)
    y_arr = idx2numpy.convert_from_file("./data/" + y_filename)
    print("Distribution: ", Counter(y_arr))
    f, axarr = plt.subplots(2,5)
    for i in range(10):
        ind = random.randint(0, len(x_arr))
        img = Image.fromarray(x_arr[ind])
        r = int(i/5)
        axarr[r,i%5].imshow(img)
    plt.savefig('images.png')

    dataset = MNISTDataset(x_arr, y_arr)
    if train:
        gen = torch.Generator().manual_seed(42)
        train_data, val_data = torch.utils.data.random_split(dataset, lengths=[0.8, 0.2], generator=gen)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def save_checkpoint(model: torch.nn.Module, epoch: int, stats: list, filename="") -> None:
    state = {
        "stats": stats,
        "state_dict": model.state_dict(),
    }

    if filename == "":
        filename = os.path.join("./checkpoints/", "epoch={}.checkpoint.pth.tar".format(epoch))
    torch.save(state, filename)

def get_checkpoint(epoch: int) -> Tuple[nn.Module, list]:
    try:
        filename = os.path.join("./checkpoints/", "epoch={}.checkpoint.pth.tar".format(epoch))
        checkpoint = torch.load(filename)
        model = MNISTModel()
        model.load_state_dict(checkpoint["state_dict"])
        return model, checkpoint["stats"]
    except Exception:
        print("Did you enter the wrong epoch?")

def calc_metrics(data: DataLoader, model: nn.Module, criterion: nn.CrossEntropyLoss) -> Tuple[float, float, float]:
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
    y_pred = torch.cat(y_pred)
    auroc = metrics.roc_auc_score(y_true, y_score, multi_class='ovo')
    f1 = metrics.f1_score(y_true, y_pred, average="weighted")

    print("Accuracy: {:.4f}\nLoss: {:.4f}\nAUROC: {:.4f}\nF1 Score: {:.4f}\n".format(acc, loss, auroc, f1))

    return acc, loss, auroc

def get_train_val_stats(
    train_data: DataLoader,
    val_data: DataLoader,
    model: nn.Module,
    criterion,
    stats,
    ):

    print("Training")
    train_accuracy, train_loss, train_auroc = calc_metrics(train_data, model, criterion)
    print("Validation")
    val_accuracy, val_loss, val_auroc = calc_metrics(val_data, model, criterion)

    epoch_stats = [
        train_accuracy,
        train_loss,
        train_auroc,
        val_accuracy,
        val_loss,
        val_auroc,
    ]

    stats.append(epoch_stats)


def update_graph(axes: plt.Axes, epoch: int, stats: list) -> None:
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


