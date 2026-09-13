"""
Small helper functions for looking at data and training results.
"""
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def walk_thru_dir(dir_path: str) -> None:
    """Prints how many subfolders/images are in each folder of dir_path."""
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"{len(dirnames)} directories, {len(filenames)} images in '{dirpath}'")


def plot_loss_curves(results: Dict[str, List[float]]) -> None:
    """Plots train/test loss and accuracy curves from engine.train()'s output."""
    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="train_loss")
    plt.plot(epochs, results["test_loss"], label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label="train_acc")
    plt.plot(epochs, results["test_acc"], label="test_acc")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
