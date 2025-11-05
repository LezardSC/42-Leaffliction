from utils.utils import iter_images
from classes.Transformation import Transformation, Options
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv
import os
import sys
from PIL import Image
import numpy as np
import tempfile


def _histogram_chart_to_ndarray(chart) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        pcv.print_image(img=chart, filename=tmp.name)
        return np.array(Image.open(tmp.name).convert("RGB"))


def _show_transforms(transformations: dict):
    nrows = 2
    ncols = 3
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    for i in range(nrows * ncols):
        axs[i // ncols][i % ncols].set_axis_off()

    i = 0
    for t, img in transformations.items():
        array = Image.fromarray(img)
        axs[i // ncols][i % ncols].imshow(array)
        axs[i // ncols][i % ncols].set_title(t)
        i += 1

    plt.show()


def save_result(res: dict,
                outdir: str,
                basename: str,
                save_artifacts: bool = False) -> None:
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


def transform_single_file(path: str, options: Options):
    # Need to apply and display all transforms
    transformations = Transformation()
    attributes = dir(Transformation)
    res = {}

    hist_img = None

    for at in attributes:
        if not at.startswith('_') and at != 'histogram':
            fn = getattr(transformations, at)
            res[at] = fn(path, options)["display"]

        elif at == 'histogram':
            try:
                out = getattr(transformations, at)(path, options)
                hist_img = _histogram_chart_to_ndarray(out["display"])
            except Exception as e:
                print(f"[warn] histogram failed: {e}")

    _show_transforms(res)

    if hist_img is not None:
        plt.figure()
        plt.imshow(hist_img)
        plt.axis('off')
        plt.title('histogram')
        plt.show()


def transform_for_directory(dirpath: str, options: Options, transform: str):
    t = Transformation()
    os.makedirs(options.outdir, exist_ok=True)

    if transform == "all":
        print("Error. Can't apply all transformations on an entire directory.")
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
            print("Error during the image transformation:")
            print(f"{e} for '{img_path}'")
            return

    print(f"All images in {dirpath} have been transformed.")
    print(f"You can find them in the folder: {options.outdir}")


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
        transform_single_file(src, options)


if __name__ == "__main__":
    main()
