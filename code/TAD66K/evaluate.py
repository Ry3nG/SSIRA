import datetime
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from data.dataset import TAD66KDataset
from models.degradationCNN import DegradationCNN
from utils.transforms import get_standard_transforms, get_degradation_transforms
from utils.utils import save_model, load_model, validate_model
import utils.constants as constants

import logging

# index_to_class mapping
index_to_class = {0: 'noise', 1: 'blur', 2: 'color', 3: 'perspective', 4: 'none'}

# Function to display images and predictions
def visualize_comparisons(original_images, degraded_images, actuals, preds, actual_levels, pred_levels):
    plt.figure(figsize=(10, 5))
    for i in range(len(degraded_images)):
        # Show original image
        ax = plt.subplot(2, len(degraded_images), i + 1)
        plt.imshow(original_images[i].cpu().numpy().transpose(1, 2, 0))
        ax.title.set_text('Original')
        plt.axis('off')

        # Show degraded image with prediction and level
        ax = plt.subplot(2, len(degraded_images), i + 1 + len(degraded_images))
        plt.imshow(degraded_images[i].cpu().numpy().transpose(1, 2, 0))
        ax.title.set_text(f'Predicted: {index_to_class[preds[i]]}\nLevel: {pred_levels[i]:.2f}\nActual: {index_to_class[actuals[i]]}\nLevel: {actual_levels[i]:.2f}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('comparison_predictions_levels.png')
    plt.show()

# load the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model = DegradationCNN(num_degradation_types=5)
model_name = "model_epoch_90_2023-11-25 11:29:02.937164.pth"
model_path = constants.PATH_MODELS + model_name
model = load_model(model, model_path)
model = model.to(device)  # Move the model to the GPU
model.eval()

# Setup logging
tic = datetime.datetime.now()
log_file = os.path.join(constants.PATH_LOGS, f"evaluate_{model_name}_{tic}.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
)
logging.info("Imported packages.")


# load the dataset
test_dataset = TAD66KDataset(
    csv_file=constants.PATH_LABEL_MERGE_TAD66K_TEST,
    root_dir=constants.PATH_DATASET_TAD66K,
    transform=get_standard_transforms(),
    degradation_transform=get_degradation_transforms(),
)

# Create the dataloader
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=constants.NUM_WORKERS,
)


all_labels = []
all_preds = []

# Evaluate the model
with torch.no_grad():
    for i,(original_images, degraded_images, labels_type, labels_level) in enumerate(test_loader):
        degraded_images = degraded_images.to(device)
        labels_type = labels_type.numpy()
        outputs, level_outputs = model(degraded_images)
        _, preds = torch.max(outputs, 1)
        pred_levels = level_outputs.view(-1).cpu().numpy()
        all_labels.extend(labels_type)
        all_preds.extend(preds.cpu().numpy())

        # Visualize the first batch of images
        if i == 0:
            visualize_comparisons(
                original_images[:5], 
                degraded_images[:5], 
                labels_type[:5], 
                preds[:5].cpu().numpy(),
                labels_level[:5], 
                pred_levels[:5]
            )

# Metrics calculation
print(classification_report(all_labels, all_preds))

# Confusion matrix
plt.figure(figsize=(10, 10))
conf_mat = confusion_matrix(all_labels, all_preds)
sns.heatmap(conf_mat, annot=True, fmt='d')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
# Save results and plots
plt.savefig('confusion_matrix.png')
plt.show()


