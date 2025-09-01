import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from utils import get_path


def draw_pie(subfolders):
	labels = list(subfolders.keys())
	numbers = list(subfolders.values())

	plt.pie(numbers, labels=labels, autopct="%1.1f%%", startangle=90)
	plt.title("Distribution of images of the dataset")
	plt.show()


def count_images(dataset: str):
	subfolders = {}
	for subfolder in os.listdir(dataset):
		path = os.path.join(dataset, subfolder)
		if os.path.isdir(path):
			files = [f for f in os.listdir(path)]
			count = len(files)
			subfolders[subfolder] = count

	return subfolders


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d", "--data",
        type=str,
        default=get_path("images/"),
        help="The path to the dataset."
    )

    return parser


def main():
	# OK take as argument a directory
	# fetches images in the directory
	# should work with every directories

	parser = build_parser()
	args = parser.parse_args()

	subfolders = count_images(args.data)
	
	draw_pie(subfolders)

if __name__ == "__main__":
	main()