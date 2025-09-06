import argparse
from pathlib import Path

def parse_args():
	p = argparse.ArgumentParser(
		description="Delete augmented images."
	)
	p.add_argument(
		"-r", "--root", default="../Apple", help="Root Folder"
	)
		
	return p.parse_args()


def main():
	args = parse_args()
	root = Path(args.root)
	deleted = 0
	  
	for p in root.rglob("*"):
		if not p.is_file():
			continue
		if p.suffix:
			ext = p.suffix.lower().lstrip(".")
		else:
			continue

		stem = p.stem
		if "_" not in stem:
			continue

		base, _suffix = stem.rsplit("_", 1)
		original = p.with_name(base + p.suffix)

		if not original.exists():
			continue

		try:
			p.unlink()
			deleted += 1
		except Exception as e:
			print(f"Failed to delete {p}: {e}")



	print(f"Done. Deleted files: {deleted}")


if __name__ == "__main__":
	main()