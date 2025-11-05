import os
from pathlib import Path
import torch
import csv
from enum import Enum


class FruitDisease(Enum):
    APPLE = ['Black rot', 'Healthy', 'Rust', 'Scab']
    GRAPE = ['Black rot', 'Esca', 'Healthy', 'Spot']


def get_fruit_diseases(fruit: str) -> FruitDisease:
    match(fruit):
        case 'Apple':
            return FruitDisease.APPLE
        case 'Grape':
            return FruitDisease.GRAPE


# Generic utils functions #

def get_path(relative_path: str) -> str:
    """
    Return the absolute path to a file or directory, given a path relative
    to this utils.py file.

    Parameters:
        relative_path (str): Path to file or directory,
        relative to this script.

    Returns:
        str: Normalized absolute path.
    """

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, "../../", relative_path)
    return path


# Functions to go through the datas #

def count_images(dataset: str) -> dict:
    subfolders = {}
    for subfolder in os.listdir(dataset):
        path = os.path.join(dataset, subfolder)
        if os.path.isdir(path):
            files = [f for f in os.listdir(path)]
            count = len(files)
            subfolders[subfolder] = count

    return subfolders


def iter_images(dirpath, recursive=False):
    p = Path(dirpath)
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    it = p.rglob("*") if recursive else p.iterdir()
    for image_path in it:
        if image_path.is_file() and image_path.suffix.lower() in IMG_EXTS:
            yield image_path


# Mean and standard deviation for images normalization #

def compute_mean_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    if not os.path.exists('./models'):
        os.makedirs('./models')
    with open('./models/mean_std.csv', 'w') as csvfile:
        fieldnames = ['Image', 'Mean', 'Std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'Image': 'R',
                         'Mean': mean.tolist()[0],
                         'Std': std.tolist()[0]})
        writer.writerow({'Image': 'G',
                         'Mean': mean.tolist()[1],
                         'Std': std.tolist()[1]})
        writer.writerow({'Image': 'B',
                         'Mean': mean.tolist()[2],
                         'Std': std.tolist()[2]})


def read_mean_std(path: str):
    mean = []
    std = []
    with open(path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            mean += [float(row['Mean'])]
            std += [float(row['Std'])]

    return mean, std
