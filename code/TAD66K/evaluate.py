import torch
import torchvision.transforms as transforms
import torchvision.utils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models.autoencoder import ConvAutoencoder 
from utils.dataset import TAD66KDataset 
import numpy as np
import os

# constants
PATH_DATASET_TAD66K = '/home/zerui/SSIRA/dataset/TAD66K/'
PATH_LABEL_MERGE_TAD66K_TEST = '/home/zerui/SSIRA/dataset/TAD66K/labels/merge/test.csv'
PATH_LABEL_MERGE_TAD66K_TRAIN = '/home/zerui/SSIRA/dataset/TAD66K/labels/merge/train.csv'


model_path = '/home/zerui/SSIRA/code/TAD66K/models/model1120.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ConvAutoencoder().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.ToTensor(),  # Convert image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

degradation_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Match the original transform size
    transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),  # Random color jitter,
    transforms.GaussianBlur(kernel_size=(9, 9), sigma=(0.1, 5)),
    transforms.Lambda(lambda x: x + 0.1*torch.randn_like(x))  # Adding random noise
])

# load the dataset
dataset = TAD66KDataset(
    csv_file = PATH_LABEL_MERGE_TAD66K_TEST,
    root_dir = PATH_DATASET_TAD66K,
    transform = transform,
    degradation_transform=degradation_transform
)

data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

def save_image(img, filename):
    # Converts a Tensor into an image grid using make_grid and saves it to a file
    img = torchvision.utils.make_grid(img)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.figure(figsize=(20, 10))  # Adjust the size as needed
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')  # Turn off axis
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()


# Parameters
num_images_to_display = 20  # How many sets of images you want to display
batch_size = 4  # You can set this to 1 if you want to use one image at a time

# Update DataLoader to have the correct batch size
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Accumulators for images
original_images_grid = []
degraded_images_grid = []
reconstructed_images_grid = []

# Process images
model.eval()
with torch.no_grad():
    for _ in range(num_images_to_display // batch_size):
        try:
            degraded_images, original_images = next(iter(data_loader))
        except StopIteration:
            break

        degraded_images, original_images = degraded_images.to(device), original_images.to(device)
        reconstructed_images = model(degraded_images).cpu()

        original_images_grid.append(original_images.cpu())
        degraded_images_grid.append(degraded_images.cpu())
        reconstructed_images_grid.append(reconstructed_images)

# Concatenate all the batches into a single grid
original_images_grid = torch.cat(original_images_grid, dim=0)
degraded_images_grid = torch.cat(degraded_images_grid, dim=0)
reconstructed_images_grid = torch.cat(reconstructed_images_grid, dim=0)

# Save the grid of images
save_image(original_images_grid, 'original_images_grid.png')
save_image(degraded_images_grid, 'degraded_images_grid.png')
save_image(reconstructed_images_grid, 'reconstructed_images_grid.png')
