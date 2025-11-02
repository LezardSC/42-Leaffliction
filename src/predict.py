import torch
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt

from classes.Network import Network
from utils.utils import read_mean_std

import argparse

classes = ('Black Rot', 'Healthy', 'Rust', 'Scab')


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
                       'Class predicted : ' + classification,
                       backgroundcolor=(0.1, 0.1, 0.1),
                       fontsize=12,
                       color=(1, 1, 1),
                       ha='center',  # Horizontal alignment
                       zorder=2)

    plt.show()


def predict(img_path: str, model_path: str, stats_path: str):
    # Load model
    model = Network(1)
    model.load_state_dict(torch.load(model_path))

    # Define tranformations
    mean, std = read_mean_std(stats_path)
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

    _display_result(img, classes[predicted])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--img",
        type=str,
        default="./Apple/Apple_rust/image (1).JPG",
        help="The image or folder that you want to predict."
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default="models/bestModel.pth",
        help="Path to the pytorch model (.pth)"
    )

    parser.add_argument(
        "-s", "--stats",
        type=str,
        default="models/mean_std.csv",
        help="Mean and Standard Deviation used to normalize data (.csv)"
    )

    return parser


def main():
    args = build_parser().parse_args()
    predict(args.img, args.model, args.stats)


if __name__ == "__main__":
    main()
