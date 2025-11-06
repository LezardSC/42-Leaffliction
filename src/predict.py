import os
import torch
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from classes.Network import Network
from utils.utils import read_mean_std, get_fruit_diseases

import argparse


def _display_result(img, classification: str):
    fig = plt.figure(figsize=(5, 7))
    fig.set_facecolor((0.1, 0.1, 0.1))

    # Images
    ax_img = fig.add_axes((0.05, 0.4, 0.9, 0.55))
    ax_img.set_axis_off()
    ax_img.imshow(img)

    # Text DL Classification
    ax_txt_classif = fig.add_axes((0.05, 0.2, 0.9, 0.1),
                                  facecolor=(0.1, 0.1, 0.1))
    ax_txt_classif.set_axis_off()
    ax_txt_classif.text(0.5, 0.02,  # Position (x, y) - centered at bottom
                        '===    DL Classification    ===',
                        backgroundcolor=(0.1, 0.1, 0.1),
                        fontsize=15,
                        color=(1, 1, 1),
                        ha='center',  # Horizontal alignment
                        zorder=2)

    # Text Result
    ax_txt_result = fig.add_axes((0.05, 0.1, 0.9, 0.1),
                                 facecolor=(0.1, 0.1, 0.1))
    ax_txt_result.set_axis_off()
    ax_txt_result.text(0.5, 0.02,  # Position (x, y) - centered at bottom
                       'Class predicted : ',
                       backgroundcolor=(0.1, 0.1, 0.1),
                       fontsize=12,
                       color=(1, 1, 1),
                       ha='right',
                       va='center',
                       zorder=2)

    ax_txt_result.text(0.5, 0.02,
                       classification,
                       fontsize=12,
                       color='lime',
                       fontweight='bold',
                       ha='left',
                       va='center',
                       zorder=2)

    plt.show()


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
                 fruit: str):

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
    output = model(input)
    _, predicted = torch.max(output.data, 1)

    fruit_diseases = get_fruit_diseases(fruit)
    classes = fruit_diseases.value
    _display_result(img, classes[predicted])


# Function to test the model with the test dataset
def predict_dataset(model: Network,
                    dataset_path: str,
                    model_path: str,
                    fruit: str):

    model.eval()
    accuracy = 0.0
    total = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    print(f"Accuracy for the given dataset : {accuracy} %")


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

    return parser


def main():
    args = build_parser().parse_args()
    model = Network()
    model.load_state_dict(torch.load(args.model + 'bestModel' +
                                     args.fruit + '.pth'))
    

    if os.path.isdir(args.src):
        predict_dataset(model, args.src, args.model, args.fruit)
    else:
        predict_data(model, args.src, args.model, args.fruit)


if __name__ == "__main__":
    main()
