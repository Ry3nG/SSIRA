from torchvision import transforms
import torchvision.transforms.functional as TF
import random

# Standard transforms
def get_standard_transforms():
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),  # Resize to 256x256
            transforms.ToTensor(),  # Convert image to PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
        ]
    )


# Degradation transformations
def get_degradation_transforms():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Flip the images randomly
        transforms.GaussianBlur(kernel_size=5),  # Blur the image
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Change brightness, contrast, saturation and hue randomly
        transforms.RandomRotation(15),  # Random rotation of 30 degrees
    ])
