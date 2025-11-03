import numpy as np
from plantcv import plantcv as pcv


class Options:
    def __init__(
            self,
            outdir: str,
            debug: str = "plot",
            writeimg: bool = True,
            result: str | None = None):
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

    def _draw_point(self, img, x: int, y: int, size: int):
        for y_img in range(len(img)):
            for x_img in range(len(img[y_img])):
                if ((x - x_img) * (x - x_img) +
                        (y - y_img) * (y - y_img) <= size * size):
                    img[y_img][x_img] = [255, 0, 0]

    def _draw_rectangle(self, img, x: int, y: int, length: int, height: int):
        for y_img in range(len(img)):
            for x_img in range(len(img[y_img])):
                if ((y_img == y or y_img == y + height) and
                        x_img > x and x_img < x + length):
                    img[y_img][x_img] = [0, 0, 255]
                if ((x_img == x or x_img == x + length) and
                        y_img > y and y_img < y + height):
                    img[y_img][x_img] = [0, 0, 255]

    def _find_roi(self, mask):
        maskindex = np.where(mask == 255)
        y = min(maskindex[0])
        height = max(maskindex[0]) - y
        x = min(maskindex[1])
        length = max(maskindex[1]) - x

        return x, y, length, height

    def roi(self, img_path: str, options: Options):
        pcv.params.debug = options.debug
        img, _, _ = pcv.readimage(img_path)

        mask = self.mask(img_path, options)["artifacts"]["mask_bin"]
        x, y, length, height = self._find_roi(mask)
        self._draw_rectangle(img, x, y, length, height)

        return {
            "tag": "roi",
            "display": img,
        }

    def histogram(self, img_path: str, options: Options):
        pcv.params.debug = options.debug
        img, _, _ = pcv.readimage(img_path)

        mask = self.mask(img_path, options)["artifacts"]["mask_bin"]
        hist_figure, hist_data = pcv.visualize.histogram(img=img,
                                                         mask=mask,
                                                         hist_data=True)

        return {
            "tag": "histogram",
            "display": hist_figure
        }

    # to display pseudolandmarks on image, need to have options='plot'
    def landmarks(self, img_path: str, options: Options):
        pcv.params.debug = options.debug
        img, _, _ = pcv.readimage(img_path)

        mask = self.mask(img_path, options)["artifacts"]["mask_bin"]

        (homolog_pts,
         start_pts,
         stop_pts,
         ptvals,
         chain,
         max_dist) = pcv.homology.acute(img=img,
                                        mask=mask,
                                        win=3,
                                        threshold=110)

        for pt in homolog_pts:
            self._draw_point(img, pt[0][0], pt[0][1], 5)

        return {
            "tag": "landmarks",
            "display": img
        }

    def mask(self, img_path: str, options: Options,
             saturation: int = 1,
             green_h_low: int = 35,
             green_h_high: int = 95,
             delta_a: float = 8,
             delta_g: int = 4,
             expand_px: int = 2) -> dict:

        pcv.params.debug = options.debug
        img, _, _ = pcv.readimage(img_path)

        blur = pcv.gaussian_blur(img=img, ksize=(5, 5))   # reduces local noise

        H = pcv.rgb2gray_hsv(rgb_img=blur, channel='h')   # Hue
        S = pcv.rgb2gray_hsv(rgb_img=blur, channel='s')   # Saturation
        A = pcv.rgb2gray_lab(rgb_img=blur, channel='a')   # LAB.a

        R = blur[:, :, 0].astype(np.int16)  # int16 to avoid overflow
        G = blur[:, :, 1].astype(np.int16)
        B = blur[:, :, 2].astype(np.int16)

        # Mask by saturation
        tS = float(np.percentile(S, saturation))
        mS = pcv.threshold.binary(gray_img=S,
                                  threshold=tS,
                                  object_type='light')

        # Estimation of the background
        b_mask = S <= np.percentile(S, 10)
        A_bg = np.median(A[b_mask]) if np.any(b_mask) else float(np.median(A))

        # Testing pixels
        green_by_a = (A.astype(np.float32) <= A_bg - delta_a)
        green_by_h = (H >= green_h_low) & (H <= green_h_high)
        green_by_g = (G >= R + delta_g) & (G >= B + delta_g)
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
        thresholded_img = pcv.threshold.otsu(gray_img=B_reduced,
                                             object_type='dark')
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
            "tag": "analyze",
            "display": shape_image,
        }

    def original(self, img_path: str, options: Options):
        img, _, _ = pcv.readimage(img_path)
        return {
            "tag": "original",
            "display": img,
        }
