import torch
from torch.nn.functional import cosine_similarity



def save_model(model, optimizer, scheduler, epoch, path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        #'scheduler_state_dict': scheduler.state_dict()
    }
    torch.save(checkpoint, path)


def load_model(model, path):
    checkpoint = torch.load(path)
    state_dict = checkpoint['model_state_dict']
    
    # Adjust for DataParallel training
    if next(iter(state_dict)).startswith('module.'):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    return model



def validate_model(model, val_loader, criterion_type, criterion_level, device):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0

    with torch.no_grad():
        for original_images, degraded_images, labels_type, labels_level in val_loader:
            degraded_images = degraded_images.to(device)
            labels_type = labels_type.to(device, dtype=torch.long)
            labels_level = labels_level.to(device, dtype=torch.float)

            type_output, level_output = model(degraded_images)
            loss_type = criterion_type(type_output, labels_type)
            loss_level = criterion_level(level_output.view(-1), labels_level)
            loss = loss_type + loss_level

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


