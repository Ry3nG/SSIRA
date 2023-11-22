import torch
import torch.optim as optim
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

import pandas as pd

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from utils.dataset import TAD66KDataset
from models.autoencoder import ConvAutoencoder

import logging

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# constants
PATH_DATASET_TAD66K = '/home/zerui/SSIRA/dataset/TAD66K/'
PATH_LABEL_MERGE_TAD66K_TEST = '/home/zerui/SSIRA/dataset/TAD66K/labels/merge/test.csv'
PATH_LABEL_MERGE_TAD66K_TRAIN = '/home/zerui/SSIRA/dataset/TAD66K/labels/merge/train.csv'

# load the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Using device: {device}')

# define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to 256x256
    transforms.RandomHorizontalFlip(),  # Randomly flip the images horizontally
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),  # Random color jitter
    transforms.ToTensor(),  # Convert image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# load the dataset
train_dataset = TAD66KDataset(
    csv_file = PATH_LABEL_MERGE_TAD66K_TRAIN,
    root_dir = PATH_DATASET_TAD66K,
    transform = transform
)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

logging.info(f'Train dataset size: {len(train_dataset)}')
logging.info(f'Validation dataset size: {len(val_dataset)}')

# initialize the model
model = ConvAutoencoder()
model.to(device)

logging.info(f'Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')


# define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the model
early_stopping_patience = 3  # Number of epochs to wait for improvement before stopping
best_val_loss = float('inf')  # Initialize best validation loss as infinity
epochs_no_improve = 0  # Counter for epochs with no improvement

num_epochs = 20
best_val_loss = float('inf')  # Initialize best validation loss as infinity
early_stopping_patience = 3  # Number of epochs to wait for improvement before stopping
epochs_no_improve = 0  # Counter for epochs with no improvement

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for degraded_images, original_images in train_loader:
        degraded_images, original_images = degraded_images.to(device), original_images.to(device)

        optimizer.zero_grad()
        reconstructed_images = model(degraded_images)
        loss = criterion(reconstructed_images, original_images)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    with torch.no_grad():
        for degraded_images, original_images in val_loader:
            degraded_images, original_images = degraded_images.to(device), original_images.to(device)
            reconstructed_images = model(degraded_images)
            loss = criterion(reconstructed_images, original_images)
            val_loss += loss.item()

            # Convert images to numpy for PSNR and SSIM calculation
            for i in range(len(original_images)):
                orig = original_images[i].cpu().numpy().transpose(1, 2, 0)
                reconstructed = reconstructed_images[i].cpu().numpy().transpose(1, 2, 0)

                # Normalize the images between 0 and 1 for PSNR and SSIM calculation
                orig = (orig - orig.min()) / (orig.max() - orig.min())
                reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min())

                total_psnr += compare_psnr(orig, reconstructed)
                total_ssim += compare_ssim(orig, reconstructed, multichannel=True)

    avg_val_loss = val_loss / len(val_loader)
    avg_psnr = total_psnr / len(val_loader.dataset)
    avg_ssim = total_ssim / len(val_loader.dataset)

    # Early stopping check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve == early_stopping_patience:
        logging.info(f'Early stopping triggered after {epoch+1} epochs!')
        break

    logging.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {avg_val_loss}, Avg PSNR: {avg_psnr}, Avg SSIM: {avg_ssim}')

# Save the trained model
torch.save(model.state_dict(), 'conv_autoencoder.pth')
