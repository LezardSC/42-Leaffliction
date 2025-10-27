import os
from pathlib import Path


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
    path = os.path.join(script_dir, "../", relative_path)
    return path


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
    IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    it = p.rglob("*") if recursive else p.iterdir()
    for image_path in it:
        if image_path.is_file() and image_path.suffix.lower() in IMG_EXTS:
            yield image_path
