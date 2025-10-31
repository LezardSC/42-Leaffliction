import torch
import torchvision
from torchvision.transforms import transforms
from PIL import Image
import matplotlib.pyplot as plt

from Network import Network
from compute_mean_std import read_mean_std



classes = ('Black Rot', 'Healthy', 'Rust', 'Scab')

def _display_result(img, classification: str):
    fig = plt.figure(figsize=(5, 7))
    fig.set_facecolor((0.1, 0.1, 0.1))

    # Images
    ax_img = fig.add_axes((0.05, 0.4, 0.9, 0.55))
    ax_img.set_axis_off()
    ax_img.imshow(img)

    # Text DL Classification
    ax_txt_classif = fig.add_axes((0.05, 0.2, 0.9, 0.1), facecolor=(0.1, 0.1, 0.1))
    ax_txt_classif.set_axis_off()
    ax_txt_classif.text(0.5, 0.02,  # Position (x, y) - centered at bottom
                '===    DL Classification    ===',
                backgroundcolor=(0.1, 0.1, 0.1),
                fontsize=15,
                color=(1, 1, 1),
                ha='center',  # Horizontal alignment
                zorder=2)
    
    # Text Result
    ax_txt_result = fig.add_axes((0.05, 0.1, 0.9, 0.1), facecolor=(0.1, 0.1, 0.1))
    ax_txt_result.set_axis_off()
    ax_txt_result.text(0.5, 0.02,  # Position (x, y) - centered at bottom
                'Class predicted : ' + classification,
                backgroundcolor=(0.1, 0.1, 0.1),
                fontsize=12,
                color=(1, 1, 1),
                ha='center',  # Horizontal alignment
                zorder=2)

    plt.show()

def predict(img_path: str):
    # Load model
    model = Network(1)
    model.load_state_dict(torch.load('models/myFirstModel.pth'))

    # Define tranformations
    mean, std = read_mean_std('models/mean_std.csv')
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Compute image
    img = Image.open(img_path)
    input = transformations(img)
    input = torch.unsqueeze(input, 0)
    model.eval()
    output = model(input)
    _, predicted = torch.max(output.data, 1)
    
    _display_result(img, classes[predicted])

    

if __name__ == "__main__":
    predict('./Apple/Apple_rust/image (1).JPG')