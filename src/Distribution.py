import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from utils import get_path, count_images


def draw_distribution(subfolders):
	labels = list(subfolders.keys())
	numbers = list(subfolders.values())

	fig, axes = plt.subplots(1, 2, figsize=(12, 6))

	axes[0].pie(numbers, labels=labels, autopct="%1.1f%%", startangle=90)
	axes[0].set_title("Distribution (pie)")


	bars = axes[1].bar(labels, numbers, color="blue")
	axes[1].set_title("Distribution (bars)")
	axes[1].set_ylabel("Number of images")
	axes[1].set_xticks(range(len(labels)))
	axes[1].set_xticklabels(labels, rotation=45, ha="right")
	axes[1].bar_label(bars, labels=[str(s) for s in numbers], padding=3)

	plt.tight_layout()
	plt.show()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--data",
        type=str,
        default=get_path("Apple/"),
        help="The path to the dataset."
    )

    return parser


def main():
	args = build_parser().parse_args()
	subfolders = count_images(args.data)

	draw_distribution(subfolders)


if __name__ == "__main__":
	main()