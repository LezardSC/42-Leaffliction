import os
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import argparse
import random

from utils.utils import get_path, count_images
from utils.delete_augmented_image import delete_augmented_images


AUGMENTATIONS = {
    "flip": T.RandomHorizontalFlip(p=1),
    "rotate": T.RandomChoice([T.RandomRotation([45, 45]),
                              T.RandomRotation([-45, -45])],
                             p=[0.5, 0.5]),
    "illumination": T.ColorJitter(brightness=0.4,
                                  contrast=0.4,
                                  saturation=0.4,
                                  hue=0.2),
    "blur": T.GaussianBlur(kernel_size=5, sigma=(1.0, 3.0)),
    "crop": T.RandomResizedCrop(size=256, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    "distortion": T.RandomPerspective(distortion_scale=0.3, p=1.0),
}


def _show_augmentations_grid(augmentations: dict, nrows, ncols):
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7, 6))

    for i in range(nrows * ncols):
        axs[i // ncols][i % ncols].set_axis_off()

    i = 0
    for t, img in augmentations.items():
        ax = axs[i // ncols][i % ncols]

        ax.imshow(img)
        ax.set_title(t)
        i += 1
    plt.tight_layout()
    plt.show()


def augment_single_file(path: str):
    dirname, basename = os.path.split(path)
    name, ext = os.path.splitext(basename)

    img = Image.open(path).convert("RGB")

    augmentations_to_draw = {}

    for transformation_name, transformation in AUGMENTATIONS.items():
        transforms = T.Compose([
            transformation,
            T.ToTensor(),
        ])

        augmentation_img = transforms(img)

        new_name = f"{name}_" + transformation_name + f"{ext}"
        new_path = os.path.join(dirname, new_name)
        save_image(augmentation_img, new_path)
        print(f"Saved: {new_path}")

        augmentations_to_draw[transformation_name] = \
            augmentation_img.permute(1, 2, 0).detach().numpy()

    _show_augmentations_grid(
        augmentations_to_draw,
        nrows=2,
        ncols=3,
    )


def _get_highest_nb_of_images(subfolders: dict) -> int:
    return subfolders[max(subfolders, key=subfolders.get)]


def augment_images_until_equal(main_dir: str, subfolders: dict):
    highest = _get_highest_nb_of_images(subfolders)

    for subfolder, nb_of_images in subfolders.items():
        print(f"starting augmentation for folder {subfolder}...")

        path = os.path.join(main_dir, subfolder)
        files = [f for f in os.listdir(path)
                 if f.lower().endswith((".jpg",
                                        ".jpeg",
                                        ".png",
                                        ".bmp",
                                        ".webp",
                                        ".tiff"))]
        if not files:
            continue
        existing = {fname.lower() for fname in os.listdir(path)}

        needed = highest - nb_of_images
        created = 0
        while created < needed:
            transformation_name, transformation = random.choice(
                list(AUGMENTATIONS.items())
                )

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

    print("Augmentation successfull.")
    print(f"All folders now contains {highest} images.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s", "--src",
        type=str,
        default=get_path("Apple/"),
        help="The path to the dataset."
    )

    return parser


def augmentation(data_path=None):
    args = build_parser().parse_args()
    data = data_path if data_path is not None else args.src
    print('src: ' + args.src)

    src = args.src

    if not os.path.exists(src):
        print(f"Error: The following path doesn't exist: {src}")

    is_dir = os.path.isdir(src)

    if is_dir:
        subfolders: dict = count_images(data)
        augment_images_until_equal(data, subfolders)
    elif os.path.isfile(src):
        augment_single_file(src)


if __name__ == "__main__":
    augmentation()
