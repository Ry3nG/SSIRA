from torchvision import transforms
import torchvision.transforms.functional as TF
import random
import torch

# Standard transforms
def get_standard_transforms():
    return transforms.Compose(
        [
            transforms.Resize((256, 256),antialias= True),  # Resize to 256x256
            transforms.ToTensor(),  # Convert image to PyTorch Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
        ]
    )


# Degradation transformations
"""
def get_degradation_transforms():
    return transforms.Compose([
        #transforms.RandomHorizontalFlip(),  # flipping images randomly
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Slightly blur the image
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Slightly alter brightness, contrast, and saturation
    ])
"""

# Custom transform for Gaussian noise
class GaussianNoiseTransform:
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

# Custom degradation transform
class CustomDegradationTransform:
    def __init__(self):
        pass

    def __call__(self, image):
        image_tensor = transforms.ToTensor()(image)
        # Randomly choose a degradation
        degradation_choice = random.choice(['gaussian_noise', 'blur', 'color_jitter', 'perspective','none'])

        if degradation_choice == 'gaussian_noise':
            image_tensor = GaussianNoiseTransform(mean=0., std=random.uniform(0.1, 0.5))(image_tensor)
            type_label = 0
            level_label = random.uniform(0.1, 0.5)

        elif degradation_choice == 'blur':
            image_tensor = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5.0))(image_tensor)
            type_label = 1
            level_label = random.uniform(0.1, 5.0)

        elif degradation_choice == 'color_jitter':
            image_tensor = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)(image_tensor)
            type_label = 2
            level_label = random.uniform(0.1, 0.5)

        elif degradation_choice == 'perspective':
            image_tensor = transforms.RandomPerspective(distortion_scale=0.05, p=1.0)(image_tensor)
            type_label = 3
            level_label = 0.05
        
        elif degradation_choice == 'none':
            type_label = 4
            level_label = 0.0
        
        
        
        # Resize the transformed image to ensure consistent size
        resize_transform = transforms.Resize((256, 256),antialias= True)
        image_tensor = resize_transform(image_tensor)

        # normalize the image_tensor
        image_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225])(image_tensor)

        return image_tensor, type_label, level_label

# Function to get degradation transforms
def get_degradation_transforms():
    return CustomDegradationTransform()