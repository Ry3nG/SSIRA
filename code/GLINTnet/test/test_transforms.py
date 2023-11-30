import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.transforms import CustomTransform, image_manipulation

def test_image_manipulation():
    img_path = '/home/zerui/SSIRA/dataset/TAD66K/65827290@N0250069447888.jpg'  # Path to test image
    image_cv = cv.imread(img_path)
    image_cv = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)

    manipulation_options = list(range(24))  # All available manipulation options
    fig, axes = plt.subplots(4, 6, figsize=(15, 10))  # Adjust subplot grid as needed

    for idx, opt in enumerate(manipulation_options):
        manipulated_img = image_manipulation(image_cv.copy(), opt)
        ax = axes[idx // 6, idx % 6]  # Determine the subplot position
        ax.imshow(cv.cvtColor(manipulated_img, cv.COLOR_BGR2RGB))
        ax.set_title(f"Option {opt}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("manipulation_examples.png")
    plt.show()

def test_custom_transform():
    img_path = '/home/zerui/SSIRA/dataset/TAD66K/65827290@N0250069447888.jpg'  # Path to test image
    image_cv = cv.imread(img_path)
    image_cv = cv.cvtColor(image_cv, cv.COLOR_BGR2RGB)

    transform = CustomTransform(manipulation_options=[0, 1, 3, 9, 12, 21])
    transformed_tensor = transform(image_cv)

    if isinstance(transformed_tensor, torch.Tensor):
        print("Custom transform applied successfully and returned a tensor.")
    else:
        print("Custom transform did not return a tensor.")

if __name__ == "__main__":
    test_image_manipulation()
    test_custom_transform()
