import os


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
