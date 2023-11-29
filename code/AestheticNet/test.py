import torch
from torch.utils.data import DataLoader
from models.aestheticNet import AestheticNet
from data.dataset import TAD66kDataset_Labeled
from torch.nn import DataParallel
from utils.constants import *

def load_model(model_path):
    model = AestheticNet()
    model = DataParallel(model)
    
    # Load the state dict with DataParallel structure
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
    
    model.eval()  # Set the model to evaluation mode
    return model

def load_test_data(test_dataset_path, root_dir, batch_size=32, num_workers=4):
    test_dataset = TAD66kDataset_Labeled(csv_file=test_dataset_path, root_dir=root_dir, custom_transform_options=[...])  # Ensure to pass all necessary arguments
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return test_loader

def test_model(model, test_loader, device, threshold=0.5):
    total_loss = 0
    total_correct = 0
    total_samples = 0

    criterion = torch.nn.L1Loss()

    with torch.no_grad():
        for images, scores in test_loader:
            images, scores = images.to(device), scores.to(device)
            aesthetic_scores = model(images, phase='aesthetic')
            aesthetic_scores = aesthetic_scores.view(-1)

            loss = criterion(aesthetic_scores, scores)
            total_loss += loss.item()

            # Compute accuracy
            correct_predictions = torch.abs(aesthetic_scores - scores) <= threshold
            total_correct += correct_predictions.sum().item()
            total_samples += scores.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples
    return avg_loss, accuracy





def main():
    print("Starting test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = '/home/zerui/SSIRA/code/AestheticNet/results/Models/aestheticNet_shadowtrain.pth'  # Replace with your model's path
    test_dataset_path = PATH_LABEL_MERGE_TAD66K_TEST  # Replace with your test dataset's path
    root_dir = PATH_DATASET_TAD66K 

    print(f"Using device: {device}")
    model = load_model(model_path)
    model.to(device)  # Move model to the appropriate device

    print("Model loaded successfully.")

    test_loader = load_test_data(test_dataset_path, root_dir)
    print("Test data loaded successfully.")

    avg_loss,accuracy = test_model(model, test_loader, device)

    print(f"Average Test Loss: {avg_loss}")
    print(f"Accuracy: {accuracy}")

    print("Done.")

if __name__ == "__main__":
    main()
