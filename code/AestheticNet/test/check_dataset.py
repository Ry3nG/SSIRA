import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from data.dataset import AVADataset
from utils.constants import *

# Function to denormalize images
def denormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    return image * std + mean

def save_images(images, labels, num_images=4):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
    for i, ax in enumerate(axes):
        img = denormalize(images[i])  # Denormalize the image
        ax.imshow(img.permute(1, 2, 0).clamp(0, 1))  # Clamp values to be in [0, 1]
        ax.axis('off')
        ax.set_title(f"Score: {labels[i]:.2f}")

    plt.savefig("images.png")

def main():
    # Assuming you have a labeled AVA dataset
    dataset = AVADataset(txt_file=PATH_AVA_TXT, root_dir=PATH_AVA_IMAGE, custom_transform_options=list(range(25)), default_transform=True)

    images, labels = [], []
    for _ in range(4):
        img, label = dataset[_]  # Get the first 4 images
        images.append(img)
        labels.append(label)

    save_images(images, labels)

if __name__ == "__main__":
    main()
