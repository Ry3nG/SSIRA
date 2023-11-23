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
        transforms.RandomHorizontalFlip(),  # flipping images randomly
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Slightly blur the image
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Slightly alter brightness, contrast, and saturation
    ])

