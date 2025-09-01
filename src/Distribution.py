import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from utils import get_path


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
	# Va check chaque nom de dossier, prendre tout avant le premier _ et en faire un executable.
	# each executable display a pie graph of the number of images inside each subdirectories.
	# should work with every directories

	parser = build_parser()
	args = parser.parse_args()

	subfolders = count_images(args.data)
	print(subfolders)

if __name__ == "__main__":
	main()