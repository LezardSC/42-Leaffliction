# Leaffliction

Leaf disease classification from images, built as a 42 school project.
Given a photo of an apple or grape leaf, the program tells whether it is
healthy or which disease it has.

![Leaffliction demo](demo/leaffliction.gif)

The full pipeline is covered: dataset analysis, data augmentation, image
transformation with [PlantCV](https://plantcv.org/), and classification
with a convolutional neural network written in [PyTorch](https://pytorch.org/).

## Features

- **Distribution**: pie and bar charts of the number of images per class.
- **Augmentation**: 6 augmentations (flip, rotate, illumination, blur, crop,
  distortion). On a folder, generates new images until every class has as
  many images as the largest one, to balance the dataset.
- **Transformation**: PlantCV pipeline that isolates the leaf from its
  background (mask), finds its bounding box (ROI), places pseudo-landmarks
  along its contour, analyzes its shape and plots its color histogram.
- **Training**: a small CNN trained per fruit, with input normalization
  computed on the dataset. The best model is kept according to accuracy
  on a held-out test set.
- **Prediction**: on a single image, shows the original and transformed
  leaf, the predicted class and the confidence for each class. On a folder,
  computes the accuracy and plots a confusion matrix.

## Dataset

8 classes of 256×256 leaf photos:

| Apple | Grape |
|---|---|
| Black rot, Healthy, Rust, Scab | Black rot, Esca, Healthy, Spot |

Scripts expect one folder per fruit, with one subfolder per class:

```
Apple/
├── Apple_Black_rot/
├── Apple_healthy/
├── Apple_rust/
└── Apple_scab/
```

## Installation

Requires **Python 3.12** (PlantCV and its dependencies do not support
3.13+ yet).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run every command from the root of the repository.

```bash
# Dataset distribution
python src/Distribution.py -d Apple/

# Augment one image (saves the 6 augmented images next to it)
python src/Augmentation.py -s "Apple/Apple_rust/image (1).JPG"
# Balance a whole dataset (writes into the class folders)
python src/Augmentation.py -s Apple/

# Show every transformation of one image
python src/Transformation.py -s "Apple/Apple_rust/image (1).JPG"
# Apply one transformation to a whole folder
python src/Transformation.py -s Apple/ -t mask -d transformations/

# Train (20 epochs, saves to models/)
python src/train.py -s Apple/ -f Apple

# Predict one image, or a whole folder
python src/predict.py -s "Apple/Apple_rust/image (1).JPG" -m models/ -f Apple
python src/predict.py -s Apple/ -m models/ -f Apple
```

`predict.py` accepts `-o result.png` to save the figure instead of
displaying it.

Training saves `bestModel<Fruit>.pth`, `lastModel<Fruit>.pth` and the
normalization values `mean_std<Fruit>.csv` in `models/`, then packs the
dataset and models into a zip archive.

## Model

The network is intentionally small: two 5×5 convolutions (10 and 15
filters) with GroupNorm and ReLU, followed by a single fully connected
layer. It is trained with AdamW and a learning rate scheduler that halves
the learning rate when the test accuracy stops improving.

After balancing the dataset with augmentation, it reaches around **96%
accuracy on apple leaves and 93% on grape leaves** on the test set.

## Project structure

```
src/
├── Distribution.py         dataset charts
├── Augmentation.py         data augmentation
├── Transformation.py       PlantCV transformations
├── train.py                training
├── predict.py              prediction
├── classes/
│   ├── Network.py          CNN definition
│   └── leaf_transformations.py   PlantCV pipeline
└── utils/                  helpers (image sizes, dataset split,
                            removal of augmented images, ...)
```

## Authors

- [LezardSC](https://github.com/LezardSC)
- [Hgeffroy](https://github.com/Hgeffroy)
