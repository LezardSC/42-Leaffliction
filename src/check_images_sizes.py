from PIL import Image
import os
import sys
from collections import Counter

sizes = Counter()

def check_images_sizes(root):
	if not root:
		print("Error: you have to give the path to a directory as an argument")
	if os.path.exists(root):
		for dirpath, _, filenames in os.walk(root):
			for filename in filenames:
				if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".tiff")):
					path = os.path.join(dirpath, filename)
					try:
						with Image.open(path) as img:
							sizes[img.size] += 1 #(width, height)
					except Exception as e:
						print(f"Error with {path}: {e}")
		
		print("Images size in folder:\n")
		for (w, h), count in sorted(sizes.items(), key=lambda x: (-x[1], x[0])):
			print(f"{w}x{h} : {count} image(s)")
		print(f"Total: {sum(sizes.values())} images")
	else:
		print(f"Error: the following path doesn't exist: {root}")

if __name__ == "__main__":
	if len(sys.argv) > 1:
		check_images_sizes(sys.argv[1])
	else:
		print("Error: you have to give the path to a directory as an argument")