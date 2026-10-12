import os
import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from classes.Network import Network
from classes.leaf_transformations import Transformation, Options
from utils.utils import read_mean_std, get_fruit_diseases

import argparse


BG_COLOR = (0.1, 0.1, 0.1)


def _show_or_save(fig, output: str | None):
    if output is None:
        plt.show()
        return
    fig.savefig(output, facecolor=fig.get_facecolor(), dpi=150)
    plt.close(fig)
    print(f"Saved: {output}")


def _display_result(img,
                    transformed,
                    classes: list,
                    probabilities: list,
                    output: str | None = None):
    predicted = int(np.argmax(probabilities))

    fig = plt.figure(figsize=(8, 7))
    fig.set_facecolor(BG_COLOR)

    # Images : original and transformed side by side
    for i, (title, image) in enumerate([("Original", img),
                                        ("Transformed", transformed)]):
        ax_img = fig.add_axes((0.04 + i * 0.48, 0.42, 0.44, 0.52))
        ax_img.set_axis_off()
        ax_img.imshow(image)
        ax_img.set_title(title, color=(1, 1, 1), fontsize=12)

    # Text DL Classification
    ax_txt = fig.add_axes((0.05, 0.27, 0.9, 0.1), facecolor=BG_COLOR)
    ax_txt.set_axis_off()
    ax_txt.text(0.5, 0.6,
                '===    DL Classification    ===',
                fontsize=15,
                color=(1, 1, 1),
                ha='center',
                va='center')
    ax_txt.text(0.5, 0.0,
                'Class predicted : ',
                fontsize=12,
                color=(1, 1, 1),
                ha='right',
                va='center')
    ax_txt.text(0.5, 0.0,
                classes[predicted],
                fontsize=12,
                color='lime',
                fontweight='bold',
                ha='left',
                va='center')

    # Confidence of the model for each class
    ax_bars = fig.add_axes((0.25, 0.04, 0.6, 0.18), facecolor=BG_COLOR)
    colors = ['lime' if i == predicted else (0.45, 0.45, 0.45)
              for i in range(len(classes))]
    bars = ax_bars.barh(classes, probabilities, color=colors)
    ax_bars.bar_label(bars,
                      labels=[f"{p:.1%}" for p in probabilities],
                      padding=4,
                      color=(1, 1, 1))
    ax_bars.set_xlim(0, 1.15)
    ax_bars.invert_yaxis()
    ax_bars.tick_params(colors=(1, 1, 1), length=0)
    ax_bars.set_xticks([])
    for spine in ax_bars.spines.values():
        spine.set_visible(False)

    _show_or_save(fig, output)


def _display_confusion_matrix(matrix: np.ndarray,
                              classes: list,
                              accuracy: float,
                              output: str | None = None):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.set_facecolor(BG_COLOR)

    ax.imshow(matrix, cmap='Greens')
    ax.set_xticks(range(len(classes)), labels=classes, rotation=30,
                  ha='right', color=(1, 1, 1))
    ax.set_yticks(range(len(classes)), labels=classes, color=(1, 1, 1))
    ax.set_xlabel('Predicted', color=(1, 1, 1))
    ax.set_ylabel('Actual', color=(1, 1, 1))
    ax.set_title(f'Confusion matrix - accuracy {accuracy:.2f} %',
                 color=(1, 1, 1))

    threshold = matrix.max() / 2
    for y in range(len(classes)):
        for x in range(len(classes)):
            ax.text(x, y, matrix[y, x],
                    ha='center',
                    va='center',
                    color='white' if matrix[y, x] > threshold else 'black')

    fig.tight_layout()
    _show_or_save(fig, output)


def _prepare_testing_dataset(dataset_path: str, model_path: str, fruit: str):
    mean, std = read_mean_std(model_path + 'mean_std' + fruit + '.csv')
    print(f"the mean of the dataset is {mean}, and the std is {std}")

    # Define transformations for the training and test sets
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Create a dataset for training and a dataset for testing
    dataset = ImageFolder(root=dataset_path, transform=transformations)

    # Create a loader for the testing
    # Note that each shuffle is set to false for the test loader.
    test_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=0)
    print("The number of images in a test set is: ",
          len(test_loader))

    return test_loader


def predict_data(model: Network,
                 img_path: str,
                 model_path: str,
                 fruit: str,
                 output: str | None = None):

    # Define tranformations
    mean, std = read_mean_std(model_path + 'mean_std' + fruit + '.csv')
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Compute image
    img = Image.open(img_path)
    input = transformations(img)
    input = torch.unsqueeze(input, 0)
    model.eval()
    with torch.no_grad():
        output_tensor = model(input)
    probabilities = torch.softmax(output_tensor, dim=1)[0].tolist()

    transformed = Transformation().mask(
        img_path,
        Options(outdir='', debug=None, writeimg=False)
    )["display"][:, :, ::-1]   # plantcv works in BGR, matplotlib in RGB

    fruit_diseases = get_fruit_diseases(fruit)
    classes = fruit_diseases.value
    _display_result(img, transformed, classes, probabilities, output)


# Function to test the model with the test dataset
def predict_dataset(model: Network,
                    dataset_path: str,
                    model_path: str,
                    fruit: str,
                    output: str | None = None):

    model.eval()
    accuracy = 0.0
    total = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    classes = get_fruit_diseases(fruit).value
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    test_loader = _prepare_testing_dataset(dataset_path, model_path, fruit)

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing", leave=False):
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images.to(device))
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels.to(device)).sum().item()
            for label, pred in zip(labels.tolist(), predicted.tolist()):
                matrix[label, pred] += 1

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    print(f"Accuracy for the given dataset : {accuracy} %")
    _display_confusion_matrix(matrix, classes, accuracy, output)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s", "--src",
        type=str,
        default="Apple/Apple_rust/image (1).JPG",
        help="The image or folder that you want to predict."
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default="models/",
        help="Path to the pytorch model (.pth)"
    )

    parser.add_argument(
        "-f", "--fruit",
        type=str,
        choices={"Apple", "Grape"},
        default="Apple",
        help="The fruit leaf you want to predict"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Save the result figure to this file instead of displaying it."
    )

    return parser


def main():
    args = build_parser().parse_args()
    model = Network()
    model.load_state_dict(torch.load(args.model + 'bestModel' +
                                     args.fruit + '.pth',
                                     map_location='cpu'))

    if os.path.isdir(args.src):
        predict_dataset(model, args.src, args.model, args.fruit, args.output)
    else:
        predict_data(model, args.src, args.model, args.fruit, args.output)


if __name__ == "__main__":
    main()
