from collections import OrderedDict
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from data.dataset import TAD66KDataset
from models.aestheticNet import AestheticNet
from utils.constants import *
import os

def denormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(image.device)  # Move to the same device as the image
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(image.device)   # Move to the same device as the image
    return image * std + mean


def visualize_reconstruction(original_images, reconstructed_images, num_images=4):
    fig, axes = plt.subplots(2, num_images, figsize=(15, 6))
    for i in range(num_images):
        # Original Image
        img = denormalize(original_images[i]).clamp(0, 1)
        axes[0, i].imshow(img.cpu().permute(1, 2, 0))
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')

        # Reconstructed Image
        img = denormalize(reconstructed_images[i]).clamp(0, 1)
        axes[1, i].imshow(img.cpu().permute(1, 2, 0))
        axes[1, i].set_title("Reconstructed")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_PLOTS, "reconstruction_comparison.png"))

def load_checkpoint(checkpoint_path, model):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['model_state_dict']

    # Adjust for DataParallel training
    if list(state_dict.keys())[0].startswith("module."):
        # Create new state_dict without 'module.' prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # Remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict

    # Load state dict
    model.load_state_dict(state_dict)
    return model

def test_pretext_phase(checkpoint_path):
    # Initialize model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AestheticNet().to(device)

    # Load checkpoint
    model = load_checkpoint(checkpoint_path, model)

    # Load dataset
    dataset = TAD66KDataset(csv_file=PATH_LABEL_MERGE_TAD66K_TEST, root_dir=PATH_DATASET_TAD66K, custom_transform_options=list(range(24)), default_transform=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    # Get a batch of images
    original_images = next(iter(loader))
    if isinstance(original_images, (list, tuple)):
        original_images = original_images[0]  # Assuming the first item is the image

    original_images = original_images.to(device)

    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconstructed_images = model(original_images, phase='pretext')

    # Visualize
    visualize_reconstruction(original_images, reconstructed_images)

if __name__ == "__main__":
    test_pretext_phase(checkpoint_path='/home/zerui/SSIRA/code/AestheticNet/results/Models/aestheticNet-checkpoint-pretext_epoch_49_2023-11-30_10-55-32.pth')
