import torch
import os
from PIL import Image
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.utils import save_image
import argparse
import random

from utils import get_path, count_images
from delete_augmented_images import delete_augmented_images


AUGMENTATIONS = {
	"flip": T.RandomHorizontalFlip(p=1),
	"rotate": T.RandomChoice([T.RandomRotation([45, 45]), T.RandomRotation([-45, -45])], p=[0.5, 0.5]),
	"illumination": T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
	"blur": T.GaussianBlur(kernel_size=5, sigma=(1.0, 3.0)),
	# "crop": T.RandomResizedCrop(size=224, scale=(0.08, 1.0), ratio=(0.9, 1.1)),
	"distortion": T.RandomPerspective(distortion_scale=0.3, p=1.0),
}


def _get_highest_nb_of_images(subfolders: dict) -> int:
	return subfolders[max(subfolders, key=subfolders.get)]


def augment_images_until_equal(main_dir: str, subfolders: dict):
	highest = _get_highest_nb_of_images(subfolders)

	for subfolder, nb_of_images in subfolders.items():
		print(f"starting augmentation for folder {subfolder}...")

		path = os.path.join(main_dir, subfolder)
		files = [f for f in os.listdir(path)
				if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"))]
		if not files:
			continue
		existing = {fname.lower() for fname in os.listdir(path)}
		
		needed = highest - nb_of_images
		created = 0
		while created < needed:
			transformation_name, transformation = random.choice(list(AUGMENTATIONS.items()))

			transforms = T.Compose([
				transformation,
				T.ToTensor(),
			])

			filename = random.choice(files)
			filepath = os.path.join(path, filename)

			img = Image.open(filepath).convert("RGB")
			augmentation = transforms(img)

			name, ext = os.path.splitext(filename)

			new_name = f"{name}_" + transformation_name + f"{ext}"
			if new_name.lower() in existing:
				continue

			new_path = os.path.join(path, new_name)

			save_image(augmentation, new_path)
			existing.add(new_name.lower())
			created += 1

	print(f"Augmentation successfull. All folders now contains {highest} images.")


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
	subfolders: dict = count_images(args.data)

	augment_images_until_equal(args.data, subfolders)


if __name__ == "__main__":
	delete_augmented_images()
	main()