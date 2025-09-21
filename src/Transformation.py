import argparse
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


from plantcv import plantcv as pcv
from plantcv.parallel import WorkflowInputs


class Options:
	def __init__(
			self,
			image: str,
			outdir: str,
			debug: str="plot",
			result: str="results.json",
			writeimg: bool=True,
		):
		self.image = image
		self.outdir = outdir
		self.debug = debug
		self.result = result
		self.writeimg = writeimg


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
		pcv.params.debug = options.debug

		img, filename, path = pcv.readimage(filename=options.image1)
		return None

	def histogram(self, img_path: str, options: Options):
		return None

	def landmarks(self, img_path: str, options: Options):
		return None

	def mask(self, img_path: str, options: Options):
		pcv.params.debug = options.debug

		img, _, _ = pcv.readimage(img_path)
		gray = pcv.rgb2gray_hsv(img=img, channel='s')
		bin_mask = pcv.threshold.binary(gray_img=gray, threshold=36, object_type='light')

		bin_mask = pcv.fill_holes(bin_img=bin_mask)
		bin_mask = pcv.erode(gray_img=bin_mask, ksize=3, i=1)
		bin_mask = pcv.dilate(gray_img=bin_mask, ksize=3, i=2)
		bin_mask = pcv.keep_largest(mask=bin_mask)

		masked_img = pcv.apply_mask(img=img, mask=bin_mask, mask_color='white')
		id_objects, obj_hierarchy = pcv.find_objects(img=img, mask=bin_mask)

		return {
        "img": img,
        "mask": bin_mask,
        "masked_img": masked_img,
        "id_objects": id_objects,
        "obj_hierarchy": obj_hierarchy,
    }

	def gaussianblur(self, img_path: str, options: Options):
		return None

	def analyze(self, img_path: str, options: Options):
		return None
	


def _show_img(img):
	imgnp = np.asarray(Image.open(img))
	imgplot = plt.imshow(imgnp)
	plt.axis("off")
	plt.show()


def transform_single_file(path, transform):

	transformation = Transformation()

	if transform == "all":
		img = transformation.all_transformations(path)
	elif transform == "roi":
		img = transformation.roi(path)
	elif transform == "mask":
		img = transformation.mask(path)
	elif transform == "histogram":
		img = transformation.histogram(path)
	elif transform == "gaussianblur":
		img = transformation.gaussianblur(path)
	elif transform == "analyze":
		img = transformation.analyze(path)
	elif transform == "landmarks":
		img = transformation.landmarks(path)
	else:
		print(f"Error: The following transformation doesn't exist: {transform}")
		sys.exit(1)

	_show_img(img)


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

	path = args.src
	if not os.path.exists(path):
		print(f"Error: The following path doesn't exist: {path}")
		sys.exit(1)
	elif os.path.isdir(path):
		print("that's a directory")
		transform_for_directory(path, transform=args.transformation)
	elif os.path.isfile(path):
		print("that's a file")
		transform_single_file(path, transform=args.transformation)


if __name__ == "__main__":
	main()

"""
	1.	recognize if image path or directory --src
	2.	if image: show 6 transformations
	3.	if directory: save in directory
		--directory to save in a particular directory
	4. --transformation to choose a single transformation else all
"""