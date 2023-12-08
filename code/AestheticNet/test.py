import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from models.aestheticNet import AestheticNet
from data.dataset import AVADataset
from data.dataset_split import AVADataset_Split
import matplotlib.pyplot as plt

from utils.constants import *
from scipy.stats import spearmanr, pearsonr


model_root_path = "/home/zerui/SSIRA/code/AestheticNet/results/Models/"
model_name = "aestheticNet-checkpoint-aesthetic_epoch_69_2023-12-05_14-01-47.pth"
model_path = model_root_path + model_name


def calculate_srcc(preds, true_scores):
    return spearmanr(preds, true_scores)[0]


def calculate_plcc(preds, true_scores):
    return pearsonr(preds, true_scores)[0]


def calculate_binary_accuracy(preds, true_scores, threshold=5.0):
    preds_binary = [1 if p >= threshold else 0 for p in preds]
    true_scores_binary = [1 if t >= threshold else 0 for t in true_scores]
    accurate_count = sum(p == t for p, t in zip(preds_binary, true_scores_binary))
    return accurate_count / len(preds)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Adjust keys if model was trained using DataParallel
    if list(state_dict.keys())[0].startswith("module."):
        # Model saved with DataParallel, but current setup does not use it
        new_state_dict = {
            k[7:]: v for k, v in state_dict.items()
        }  # Remove 'module.' prefix
    else:
        # Model saved without DataParallel, but current setup uses it
        new_state_dict = {"module." + k: v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    return model


def denormalize(image):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1).to(image.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1).to(image.device)
    return image * std + mean


def plot_and_save_images(
    images,
    predicted_scores,
    ground_truth_scores,
    filename="composite_image_" + model_name + ".png",
):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 5, 10))

    for i in range(num_images):
        ax = axes[i]
        img = (
            denormalize(images[i]).permute(1, 2, 0).numpy()
        )  # Denormalize and convert from CxHxW to HxWxC
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(
            f"Pred: {predicted_scores[i]:.2f}\nTrue: {ground_truth_scores[i]:.2f}"
        )

    plt.subplots_adjust()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# Collect a small number of sample images for display
sample_images = []
sample_pred_scores = []
sample_true_scores = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate and load the model
model = AestheticNet()
model = load_checkpoint(model, model_path, device)
model = model.to(device)
model.eval()


# Read AVA test IDs
def read_ava_ids(file_path):
    with open(file_path, "r") as file:
        test_ids = [line.strip() for line in file]
    return test_ids


ava_test_ids = read_ava_ids(PATH_AVA_TEST_IDS)

# Create an instance of AVADataset for testing
test_dataset = AVADataset(
    txt_file=PATH_AVA_TXT,
    root_dir=PATH_AVA_IMAGE,
    custom_transform_options=[23],
    default_transform=True,
    include_ids=True,
    ids=ava_test_ids,
)
test_dataset = AVADataset_Split(
    csv_files=["/home/zerui/SSIRA/dataset/AVA_Split/test_hlagcn.csv"],
    root_dir=PATH_AVA_IMAGE,
    custom_transform_options=[23],
    split="hlagcn",
)

# Create DataLoader for the test dataset
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=NUM_WORKERS
)

# Predict and evaluate
predicted_scores = []
ground_truth_scores = []

with torch.no_grad():
    for data in test_loader:
        images = data[0]
        scores = data[1]
        images = images.to(device)
        predicted_scores_batch = model(images, phase="aesthetic").squeeze().cpu()
        predicted_scores.extend(predicted_scores_batch.tolist())
        ground_truth_scores.extend(scores.tolist())

        if len(sample_images) < 10:  # Corrected to stop collecting after enough samples
            remaining_samples = 10 - len(sample_images)
            sample_images.extend(images.cpu()[:remaining_samples])
            sample_pred_scores.extend(
                predicted_scores_batch[:remaining_samples].tolist()
            )
            sample_true_scores.extend(scores[:remaining_samples].tolist())

# Calculate the metrics after the loop
srcc = calculate_srcc(predicted_scores, ground_truth_scores)
plcc = calculate_plcc(predicted_scores, ground_truth_scores)
acc = calculate_binary_accuracy(predicted_scores, ground_truth_scores)

# Calculate Mean Squared Error (MSE)
mse = torch.mean(
    (torch.tensor(predicted_scores) - torch.tensor(ground_truth_scores)) ** 2
).item()

# Output results
print(f"SRCC: {srcc:.4f}, PLCC: {plcc:.4f}, Accuracy: {acc:.4f}, MSE: {mse:.4f}")

# Plot and save sample images with scores
plot_and_save_images(sample_images, sample_pred_scores, sample_true_scores)
