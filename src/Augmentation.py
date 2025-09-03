import torch
import os
from PIL import Image
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.utils import save_image
import argparse

from utils import get_path, count_images


#TODO dict of augmentation and name so we apply in a loop and give the right name
#TODO highest - nb of images in the folder then we do the for loop in the range.
#TODO construct train and test folders before doing the augmentation.


AUGMENTATIONS = {
	"flip": T.RandomHorizontalFlip(p=1),
	"rotate": T.RandomChoice([T.RandomRotation([45, 45]), T.RandomRotation([-45, -45])], p=[0.5, 0.5]),
	"illumination": T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
	"blur": T.GaussianBlur(kernel_size=5, sigma=(1.0, 3.0)),
	"crop": T.RandomResizedCrop(size=224, scale=(0.5, 2), ratio=(1.0, 1.0)),
	"distortion": T.RandomPerspective(distortion_scale=0.3, p=1.0),
}


def _get_highest_nb_of_images(subfolders) -> int:
	return subfolders[max(subfolders, key=subfolders.get)]


def augment_images_until_equal(main_dir, subfolders):
	highest = _get_highest_nb_of_images(subfolders)

	transforms = T.Compose([
		T.RandomPerspective(distortion_scale=0.3, p=1.0),
		T.ToTensor(),
	])


	for subfolder in subfolders.keys():
		path = os.path.join(main_dir, subfolder)

		files = [f for f in os.listdir(path)
				 if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"))]
		if not files:
			continue

		filename = sorted(files)[0]
		filepath = os.path.join(path, filename)

		img = Image.open(filepath).convert("RGB")
		augmentation = transforms(img)

		name, ext = os.path.splitext(filename)
		new_name = f"{name}_aug_{ext}"
		new_path = os.path.join(path, new_name)

		save_image(augmentation, new_path)

		return new_path
		




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

	augment_images_until_equal(args.data, subfolders)

if __name__ == "__main__":
	main()