from pathlib import Path
import shutil
import random
import os

SOURCE_ROOT = Path("./transformations")
DEST_ROOT = Path("./dataset_split")
VAL_RATIO = 0.2
SEED = 42


def sym_link_file(files, split_name, class_dir):
	for src in files:
		dst = DEST_ROOT / split_name / class_dir.name / src.name
		if dst.exists(): continue
		try:
			os.symlink(src.resolve(), dst)
		except FileExistsError:
			pass


def make_split():
	random.seed(SEED)
	if DEST_ROOT.exists():
		print(f"{DEST_ROOT} already exists, delete it if you want to recreate it.")
		return
	
	# creating the folders
	(DEST_ROOT / "train").mkdir(parents=True, exist_ok=True)
	(DEST_ROOT / "val").mkdir(parents=True, exist_ok=True)

	classes = [d for d in SOURCE_ROOT.iterdir() if d.is_dir()]
	if not classes:
		raise SystemExit(f"No class has been found in {SOURCE_ROOT}")
	
	for class_dir in classes:
		imgs = [img for img in class_dir.glob("*") if img.is_file()]
		imgs.sort(key=lambda img: img.name.lower())
		random.shuffle(imgs)

		n = len(imgs)
		n_val = max(1, int(round(n * VAL_RATIO)))

		val_imgs = set(imgs[:n_val])
		train_imgs = imgs[n_val:]

		(DEST_ROOT / "train" / class_dir.name).mkdir(parents=True, exist_ok=True)
		(DEST_ROOT / "val" / class_dir.name).mkdir(parents=True, exist_ok=True)


		sym_link_file(train_imgs, "train", class_dir)
		sym_link_file(val_imgs, "val", class_dir)

if __name__ == "__main__":
	make_split()