import torch
from torch.nn.functional import cosine_similarity



def save_model(model, optimizer, scheduler, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, path)


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model

def validate_model(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    total_cosine_sim = 0.0
    with torch.no_grad():
        for degraded_images, original_images in val_loader:
            degraded_images, original_images = degraded_images.to(device), original_images.to(device)
            output1, output2 = model((degraded_images, original_images)) # tuple!
            loss = criterion(output1, output2)
            val_loss += loss.item()

            # Inside the validate_model function
            cosine_sim = cosine_similarity(output1, output2).mean()
            total_cosine_sim += cosine_sim.item()


    avg_val_loss = val_loss / len(val_loader)
    avg_distance = total_cosine_sim / len(val_loader)
    return avg_val_loss, avg_distance

