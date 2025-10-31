import argparse
from pathlib import Path

AUGMENTATIONS = {
	"flip",
	"rotate",
	"illumination",
	"blur",
	"crop",
	"distortion",
}


def _is_aug_suffix(suffixes):
	if not suffixes:
		return False
	return all(suffix.lower() in AUGMENTATIONS for suffix in suffixes)


def find_original_candidate(p: Path):
	stem_tokens = p.stem.split("_")
	if len(stem_tokens) <= 1:
		return None
	
	for i in range(1, len(stem_tokens)):
		base_tokens = stem_tokens[:-i]
		suffix_tokens = stem_tokens[-i:]

		if not _is_aug_suffix(suffix_tokens):
			continue

		base = "_".join(base_tokens)
		candidate = p.with_name(base + p.suffix)
		if candidate.exists():
			return candidate

	return None
		

def parse_args():
	p = argparse.ArgumentParser(
		description="Delete augmented images."
	)
	p.add_argument(
		"-r", "--root", default="./Apple", help="Root Folder"
	)
		
	return p.parse_args()


def delete_augmented_images(root_path=None):
	print("Deleting all augmented images...")

	args = parse_args()
	root = Path(root_path if root_path is not None else args.root).resolve()
	deleted = 0
	  
	for p in root.rglob("*"):
		if not p.is_file():
			continue

		original = find_original_candidate(p)
		if original is None:
			continue

		try:
			p.unlink()
			deleted += 1
		except Exception as e:
			print(f"Failed to delete {p}: {e}")



	print(f"Done. Deleted files: {deleted}")


if __name__ == "__main__":
	delete_augmented_images()