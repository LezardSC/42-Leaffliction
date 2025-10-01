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
		return None

	def histogram(self, img_path: str, options: Options):
		return None

	def landmarks(self, img_path: str, options: Options):
		return None

	def _clean(self, mask):
		mask = pcv.fill_holes(bin_img=mask)
		mask = pcv.erode(gray_img=mask, ksize=3, i=1)
		mask = pcv.dilate(gray_img=mask, ksize=3, i=2)
		return mask

	def _largest(self, mask, img):
		h, w = img.shape[:2]
		roi = pcv.roi.rectangle(img=img, x=0, y=0, w=w, h=h)
		return pcv.roi.filter(mask=mask, roi=roi, roi_type="largest")

	def _coverage(self, mask_uint8):
		return float(np.count_nonzero(mask_uint8)) / mask_uint8.size

	def mask(self, img_path: str, options: Options, fixed_thresh: int = 36) -> dict:
		pcv.params.debug = options.debug   # garde None en run normal
		img, _, _ = pcv.readimage(img_path)

		# Prétraitement
		blur = pcv.gaussian_blur(img=img, ksize=(5, 5))
		s = pcv.rgb2gray_hsv(rgb_img=blur, channel='s')   # saturation
		v = pcv.rgb2gray_hsv(rgb_img=blur, channel='v')   # luminosité
		a = pcv.rgb2gray_lab(rgb_img=blur, channel='a')
		b = pcv.rgb2gray_lab(rgb_img=blur, channel='b')

		# 1) Construire des candidats
		candidates = []
		# a) Seuil fixe (ton ancien comportement)
		m = pcv.threshold.binary(gray_img=s, threshold=fixed_thresh, object_type='light')
		candidates.append(("s_fixed", m))
		# b) Percentile (adapté par image, mais simple)
		t = float(np.percentile(s, 60))
		m = pcv.threshold.binary(gray_img=s, threshold=t, object_type='light')
		candidates.append(("s_p60", m))
		# c) Otsu sur V (feuille souvent plus sombre => 'dark')
		m = pcv.threshold.otsu(gray_img=v, object_type='dark')
		candidates.append(("v_otsu_dark", m))
		# d) Conjonction couleur: a=dark (vert), b=light (jaune)
		ma = pcv.threshold.otsu(gray_img=a, object_type='dark')
		mb = pcv.threshold.otsu(gray_img=b, object_type='light')
		m = pcv.logical_and(ma, mb)
		candidates.append(("a_dark_and_b_light", m))

		# 2) Nettoyer + garder le plus grand objet + scorer
		scored = []
		for tag, m in candidates:
			m = self._clean(m)
			labeled = self._largest(m, img)
			m_bin = (labeled > 0).astype(np.uint8) * 255
			cov = self._coverage(m_bin)
			# Cible simple : couverture entre 2% et 70%, et on préfère ~20–35%
			plausible = 0.02 <= cov <= 0.70
			score = (1.0 if plausible else 0.0) - abs(cov - 0.28)  # 0.28 = cible douce
			scored.append((score, tag, m_bin))

		# 3) Choisir le meilleur
		scored.sort(reverse=True, key=lambda x: x[0])
		best_score, best_tag, best_mask = scored[0]

		masked_img = pcv.apply_mask(img=img, mask=best_mask, mask_color='white')

		return {
			"tag": f"mask_{best_tag}",
			"display": masked_img,                # ce que tu affiches/sauves
			"artifacts": {
				"mask_bin": best_mask             # ton masque final (0/255)
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
			result = fn(img_path, options)
			save_result(result, outdir=options.outdir, basename=basename)
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