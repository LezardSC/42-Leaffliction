import argparse
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs


class Options:
	def __init__(
			self,
			outdir: str,
			debug: str="plot",
			writeimg: bool=True,
			result: str | None=None,
		):
		self.outdir = outdir
		self.debug = debug
		self.writeimg = writeimg
		self.result = result


class Transformation:
	def __init__(self):
		pass

	def __call__(self):
		pass

	def __repr__(self):
		pass

	def all_transformations(self, img_path: str, options: Options):
		return None

	def roi(self, img_path: str, options: Options):
		return None

	def histogram(self, img_path: str, options: Options):
		return None

	def landmarks(self, img_path: str, options: Options):
		return None

	def mask(self, img_path: str, options: Options) -> dict:
		pcv.params.debug = options.debug

		img, _, _ = pcv.readimage(img_path)
		gray = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

		bin_mask = pcv.threshold.binary(gray_img=gray, threshold=36, object_type='light')
		bin_mask = pcv.fill_holes(bin_img=bin_mask)
		bin_mask = pcv.erode(gray_img=bin_mask, ksize=3, i=1)
		bin_mask = pcv.dilate(gray_img=bin_mask, ksize=3, i=2)

		h, w = img.shape[:2]
		roi = pcv.roi.rectangle(img=img, x=0, y=0, h=h, w=w)
		labeled_mask = pcv.roi.filter(mask=bin_mask, roi=roi, roi_type="largest")

		bin_for_vis = (labeled_mask > 0).astype(np.uint8) * 255
		masked_img = pcv.apply_mask(img=img, mask=bin_for_vis, mask_color='white')

		return {
			"tag": "mask",
			"display": masked_img,  # ce qu’on montre/sauve par défaut
			"artifacts": {
				"mask_bin": bin_mask,
				"mask_labeled": (labeled_mask > 0).astype(np.uint8) * 255
			}
		}

	def gaussianblur(self, img_path: str, options: Options):
		return None

	def analyze(self, img_path: str, options: Options):
		return None


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

	transformation = Transformation()
	p = Path(path)

	if transform == "all":
		res = transformation.all_transformations(path)
	elif transform == "roi":
		res = transformation.roi(path)
	elif transform == "mask":
		res = transformation.mask(path, options)
	elif transform == "histogram":
		res = transformation.histogram(path)
	elif transform == "gaussianblur":
		res = transformation.gaussianblur(path)
	elif transform == "analyze":
		res = transformation.analyze(path)
	elif transform == "landmarks":
		res = transformation.landmarks(path)
	else:
		print(f"Error: The following transformation doesn't exist: {transform}")
		sys.exit(1)

	if res is None:
		print("Nothing to do"); return

	show_array(res["display"])
	if options.writeimg:
		save_result(res, outdir=options.outdir, basename=p.stem, save_artifacts=False)


def transform_for_directory(path, transform):
	transformation = Transformation()
	# for loop
	# save


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser()

	parser.add_argument(
		"-d", "--dst",
		type=str,
		default="../transformations/",
		help="The folder where the transformations will be stored."
	)
	parser.add_argument(
		"-s", "--src",
		type=str,
		default="../Apple/Apple_healthy/image (5).JPG",
		help="The image or folder that you want to transform."
	)
	parser.add_argument(
		"-t", "--transformation",
		type=str,
		default="all",
		help="The transformation that you want (default=all)"
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
		print("that's a directory")
		transform_for_directory(src, transform=args.transformation)
	elif os.path.isfile(src):
		print("that's a file")
		transform_single_file(src, options, transform=args.transformation)


if __name__ == "__main__":
	main()

"""
	1.	recognize if image path or directory --src
	2.	if image: show 6 transformations
	3.	if directory: save in directory
		--directory to save in a particular directory
	4. --transformation to choose a single transformation else all
"""