"""
Functions for getting image data ready to train on.

Expects data laid out like:
    data/
      train/
        class_a/*.jpg
        class_b/*.jpg
      test/
        class_a/*.jpg
        class_b/*.jpg
"""
import os
from pathlib import Path
import zipfile

import requests
from torch.utils.data import DataLoader
from torchvision import datasets

NUM_WORKERS = os.cpu_count()


def download_data(url: str, destination: str, data_dir: str = "data") -> Path:
    """Downloads a zip file from `url` and unzips it into data_dir/destination
    (skips if that folder already exists)."""
    image_path = Path(data_dir) / destination

    if image_path.is_dir():
        print(f"{image_path} already exists, skipping download.")
        return image_path

    image_path.mkdir(parents=True, exist_ok=True)
    zip_path = Path(data_dir) / f"{destination}.zip"

    print(f"Downloading {destination}...")
    request = requests.get(url)
    request.raise_for_status()
    with open(zip_path, "wb") as f:
        f.write(request.content)

    print(f"Unzipping {destination}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(image_path)

    zip_path.unlink()  # remove the zip once extracted
    return image_path


def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform,
    batch_size: int,
    num_workers: int = NUM_WORKERS,
):
    """Turns train/test folders into DataLoaders + returns the class names."""
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)
    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names
