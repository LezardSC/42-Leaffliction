import argparse
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs

from utils import iter_images


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
		# From mask with highest x and highest y points ?
		return None

	def histogram(self, img_path: str, options: Options):
		pcv.params.debug = options.debug
		img, _, _ = pcv.readimage(img_path)

		mask = self.mask(img_path, options)["artifacts"]["mask_bin"]
		hist_figure, hist_data = pcv.visualize.histogram(img=img, mask=mask, hist_data=True)

		return {
			"tag": "landmarks",
			"display": hist_figure
		}

	def landmarks(self, img_path: str, options: Options):
		pcv.params.debug = options.debug
		img, _, _ = pcv.readimage(img_path)

		mask = self.mask(img_path, options)["artifacts"]["mask_bin"]
		homolog_pts, start_pts, stop_pts, ptvals, chain, max_dist = pcv.homology.acute(img=img, mask=mask, win=25, threshold=90)
		
		
		return {
			"tag": "landmarks",
			"display": None # find a way to pass the image
		}

	def mask(self, img_path: str, options: Options,
			saturation: int = 1,      # in %, the lower it is, the more pixels we keep.
			green_h_low: int = 35, green_h_high: int = 95,  # min and max hue. Varies between 0 and 179 on OpenCV. Green is around 60
			delta_a: float = 8,        # margin LAB.a, oppose the green of the leaf with the background. Weak value is green, strong is magenta
			delta_g: int = 4,            # margin RGB where  G > R and G > B
			expand_px: int = 2           # expand the mask a little
			) -> dict:

		pcv.params.debug = options.debug
		img, _, _ = pcv.readimage(img_path)

		blur = pcv.gaussian_blur(img=img, ksize=(5, 5)) # reduces local noise

		H = pcv.rgb2gray_hsv(rgb_img=blur, channel='h')   # Hue
		S = pcv.rgb2gray_hsv(rgb_img=blur, channel='s')   # Saturation (grey <-> bright)
		V = pcv.rgb2gray_hsv(rgb_img=blur, channel='v')   # Brightness (black <-> white)

		A = pcv.rgb2gray_lab(rgb_img=blur, channel='a')   # LAB.a (green <-> magenta)

		R = blur[:, :, 0].astype(np.int16) # int16 to avoid overflow
		G = blur[:, :, 1].astype(np.int16)
		B = blur[:, :, 2].astype(np.int16)

		# Mask by saturation
		tS = float(np.percentile(S, saturation))
		mS = pcv.threshold.binary(gray_img=S, threshold=tS, object_type='light')

		# Estimation of the background
		bg_mask = S <= np.percentile(S, 10)      # low saturation pixels is probably background
		A_bg = np.median(A[bg_mask]) if np.any(bg_mask) else float(np.median(A))

		# Testing pixels
		green_by_a = (A.astype(np.float32) <= A_bg - delta_a)            # greener than background
		green_by_h = (H >= green_h_low) & (H <= green_h_high)            # hue in the specter
		green_by_g = (G >= R + delta_g) & (G >= B + delta_g)             # G > R,B with margin
		mGreen = (green_by_a | green_by_h | green_by_g).astype(np.uint8) * 255

		# Combine saturation mask and green mask
		m = pcv.logical_and(mS, mGreen)

		# Cleaning
		m = pcv.dilate(gray_img=m, ksize=5, i=1)
		m = pcv.erode(gray_img=m, ksize=5, i=1)
		m = pcv.fill_holes(bin_img=m)
		h, w = img.shape[:2]
		roi = pcv.roi.rectangle(img=img, x=0, y=0, w=w, h=h)
		labeled = pcv.roi.filter(mask=m, roi=roi, roi_type="largest")
		bin_mask = (labeled > 0).astype(np.uint8) * 255

		# Slightly expand the mask
		if expand_px > 0:
			iters = int(np.ceil(expand_px / 2))
			bin_mask = pcv.dilate(gray_img=bin_mask, ksize=3, i=iters)

		masked_img = pcv.apply_mask(img=img, mask=bin_mask, mask_color='white')

		return {
			"tag": "mask",
			"display": masked_img,
			"artifacts": {"mask_bin": bin_mask}
		}


	def gaussianblur(self, img_path: str, options: Options):
		pcv.params.debug = options.debug
		img, _, _ = pcv.readimage(img_path)
		mask = self.mask(img_path, options)["artifacts"]["mask_bin"]

		B = pcv.rgb2gray_hsv(rgb_img=img, channel='v')   # Brightness
		B_reduced = pcv.logical_and(B, mask)
		thresholded_img = pcv.threshold.otsu(gray_img=B_reduced, object_type='dark')
		blur = pcv.gaussian_blur(img=thresholded_img, ksize=(5, 5))
		return {
			"tag": "gaussianblur",
			"display": blur,
		}

	def analyze(self, img_path: str, options: Options):
		pcv.params.debug = options.debug
		img, _, _ = pcv.readimage(img_path)

		mask = self.mask(img_path, options)["artifacts"]["mask_bin"]

		# Optionally, set a sample label name
		pcv.params.sample_label = "plant"

		# Characterize object shapes
		shape_image = pcv.analyze.size(img=img, labeled_mask=mask, n_labels=1)

		# Save returned images with more specific naming
		pcv.print_image(shape_image, '/home/malia/setaria_shape_img.png')
		return {
			"tag": "mask",
			"display": shape_image,
		}


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
	p = Path(path)

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
	for img_path in iter_images(dirpath):
		basename = Path(img_path).stem
		try:
			res = fn(img_path, options)
			save_result(res, outdir=options.outdir, basename=basename)
		except Exception as e:
			print(f"Error during the image transformation: {e} for '{img_path}'")
			return 
	
	print(f"All images in {dirpath} have been transformed. You can find them in the folder: {options.outdir}")


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
		transform_for_directory(src, options, transform=args.transformation)
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