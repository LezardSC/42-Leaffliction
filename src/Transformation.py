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
			writeimg: bool=False,
		):
		self.image = image
		self.outdir = outdir
		self.debug = debug
		self.result = result
		self.writeimg = writeimg



# # Set debug parameter
# pcv.params.debug = args.debug
# # Set (optional) parameters 
# pcv.params.sample_label = args.sample_label
# pcv.params.line_thickness = 15


# img, filename, path = pcv.readimage(filename=args.image1)

# circle_roi = pcv.roi.circle(img=img, x=280, y=160, r=70)



class Transformation:
	def __init__(self):
		pass

	def __call__(self):
		pass

	def __repr__(self):
		pass

	def all_transformations(self, base_img):
		return base_img

	def roi(self, base_img):
		return base_img

	def histogram(self, base_img):
		return base_img

	def landmarks(self, base_img):
		return base_img

	def mask(self, base_img):
		return base_img

	def gaussianblur(self, base_img):
		return base_img

	def analyze(self, base_img):
		return base_img


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