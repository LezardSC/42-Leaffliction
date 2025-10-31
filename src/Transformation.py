from utils.utils import iter_images
from classes.Transformation import Transformation, Options
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import numpy as np
import os
import sys


def show_array(arr: np.ndarray) -> None:
	if not isinstance(arr, np.ndarray):
		raise TypeError(f"show_array attend un np.ndarray, pas {type(arr)}")
	plt.imshow(arr)
	plt.axis("off")
	plt.show()

def save_result(res: dict, outdir: str, basename: str, save_artifacts: bool=False) -> None:
	os.makedirs(outdir, exist_ok=True)
	# image principale
	out_main = os.path.join(outdir, f"{basename}__{res['tag']}.png")
	pcv.print_image(img=res["display"], filename=out_main)
	print(f"saved: {out_main}")
	# artefacts optionnels
	if save_artifacts and res.get("artifacts"):
		for k, arr in res["artifacts"].items():
			out_k = os.path.join(outdir, f"{basename}__{res['tag']}--{k}.png")
			pcv.print_image(img=arr, filename=out_k)
			print(f"saved: {out_k}")

def transform_single_file(path: str, options: Options, transform: str):
	t = Transformation()
	
	fn = getattr(t, transform)
	if fn is None:
		print(f"Error: The following transformation doesn't exist: {transform}")
		return
	res = fn(path, options)

	if res is None:
		print("Nothing to do"); return

	show_array(res["display"])


def transform_for_directory(dirpath: str, options: Options, transform: str):
	t = Transformation()
	os.makedirs(options.outdir, exist_ok=True)

	if transform == "all":
		print("Error. You can't apply all transformations for an entire directory.")
		return
	
	fn = getattr(t, transform)
	if fn is None:
		print(f"Error: The following transformation isn't valid: {transform}")
		return
	for img_path in iter_images(dirpath, True):
		basename = Path(img_path).stem
		try:
			res = fn(img_path, options)
			p = Path(img_path).parts
			save_result(res, outdir=options.outdir + p[1], basename=basename)
		except Exception as e:
			print(f"Error during the image transformation: {e} for '{img_path}'")
			return 
	
	print(f"All images in {dirpath} have been transformed. You can find them in the folder: {options.outdir}")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"-d", "--dst",
		type=str,
		default="./transformations/",
		help="The folder where the transformations will be stored."
	)
	parser.add_argument(
		"-s", "--src",
		type=str,
		default="./Apple",
		help="The image or folder that you want to transform."
	)
	parser.add_argument(
		"-t", "--transformation",
		type=str,
		default="mask",
		help="The transformation that you want"
	)

	return parser


def main():
	args = build_parser().parse_args()
	src = args.src

	if not os.path.exists(src):
		print(f"Error: The following path doesn't exist: {src}")
		sys.exit(1)

	is_dir = os.path.isdir(src)

	options = Options(
		outdir=args.dst,
		debug=None,
		writeimg=is_dir,
		result=None,
	)

	if is_dir:
		transform_for_directory(src, options, transform=args.transformation)
	elif os.path.isfile(src):
		transform_single_file(src, options, transform=args.transformation)


if __name__ == "__main__":
	main()