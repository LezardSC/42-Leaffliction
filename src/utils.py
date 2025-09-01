import os

def get_path(relative_path: str) -> str:
	"""
	Return the absolute path to a file or directory, given a path relative
	to this utils.py file.

	Parameters:
		relative_path (str): Path to file or directory, relative to this script.

	Returns:
		str: Normalized absolute path.
	"""

	script_dir = os.path.dirname(os.path.abspath(__file__))
	path = os.path.join(script_dir, "../", relative_path)
	return path