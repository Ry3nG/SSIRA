import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from data.dataset import TAD66KDataset, AVADataset
from models.aestheticNet import AestheticNet
from utils.losses import ReconstructionLoss, AestheticScoreLoss
from utils.constants import *
from utils.transforms import CustomTransform

import logging
import datetime
import os



def train(model, dataloader, criterion,optimizer, device, phase):
    model.train() # set model to training mode
    total_loss = 0.0

    for batch in dataloader:
        if phase == 'pretext':
            inputs = batch.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, phase = phase)
            loss = criterion(outputs, inputs)
        elif phase == 'aesthetic':
            images, scores = batch
            images,scores = images.to(device), scores.to(device)
            optimizer.zero_grad()
            outputs = model(images, phase = phase)
            loss = criterion(outputs, scores)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, phase):
    model.eval() # set model to evaluation mode
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if phase == 'pretext':
                inputs = batch.to(device)
                outputs = model(inputs, phase = phase)
                loss = criterion(outputs, inputs)
            elif phase == 'aesthetic':
                images, scores = batch
                images,scores = images.to(device), scores.to(device)
                outputs = model(images, phase = phase)
                loss = criterion(outputs, scores)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def save_model(model, epoch, save_dir, filename):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"{filename}_epoch_{epoch}.pth")
    torch.save(model.state_dict(), save_path)

def setup_logging(current_time):
    if not os.path.exists(PATH_LOGS):
        os.makedirs(PATH_LOGS)
    logging.basicConfig(filename=os.path.join(PATH_LOGS, f'training {current_time}.log'),
                        level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        filemode='w')

def main():
    # save training start time
    tic = datetime.datetime.now()

    # setup logging
    setup_logging(tic)

    # training start message
    logging.info(f'Training started at {tic}')


    # Logging the hyperparameters
    logging.info(f'Batch Size: {BATCH_SIZE}')
    logging.info(f'Number of Epochs: {NUM_EPOCHS}')
    logging.info(f'Learning Rate: {LEARNING_RATE}')


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f'Using device: {device}')
    logging.info(torch.cuda.get_device_name(device))

    # define transforms
    custom_transform_options = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    custom_transform = CustomTransform(custom_transform_options)


    # initialize and split the datasets for training and validation
    full_train_dataset_pretext = TAD66KDataset(csv_file=PATH_LABEL_MERGE_TAD66K_TRAIN, 
                                           root_dir=PATH_DATASET_TAD66K, 
                                           custom_transform_options=custom_transform_options)

    train_size_pretext = int(TRAIN_VAL_SPLIT_RATIO * len(full_train_dataset_pretext))
    val_size_pretext = len(full_train_dataset_pretext) - train_size_pretext
    train_dataset_pretext, val_dataset_pretext = random_split(full_train_dataset_pretext, [train_size_pretext, val_size_pretext])

    full_train_dataset_aesthetic = AVADataset(txt_file=PATH_AVA_TXT, 
                                          root_dir=PATH_AVA_IMAGE, 
                                          custom_transform_options=custom_transform_options)
    train_size_aesthetic = int(TRAIN_VAL_SPLIT_RATIO * len(full_train_dataset_aesthetic))
    val_size_aesthetic = len(full_train_dataset_aesthetic) - train_size_aesthetic
    train_dataset_aesthetic, val_dataset_aesthetic = random_split(full_train_dataset_aesthetic, [train_size_aesthetic, val_size_aesthetic])

    # create dataloaders
    train_loader_pretext = DataLoader(train_dataset_pretext, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader_pretext = DataLoader(val_dataset_pretext, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    train_loader_aesthetic = DataLoader(train_dataset_aesthetic, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader_aesthetic = DataLoader(val_dataset_aesthetic, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # initialize the model and loss function
    model = AestheticNet().to(device)
    criterion_pretext = ReconstructionLoss().to(device)
    criterion_aesthetic = AestheticScoreLoss().to(device)

    # initialize the optimizer
    optimizer_pretext = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer_aesthetic = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training and validation loop 
    for epoch in range(NUM_EPOCHS):
        # Train in pretext phase
        train_loss_pretext = train(model, train_loader_pretext, criterion_pretext, optimizer_pretext, device, 'pretext')
        val_loss_pretext = validate(model, val_loader_pretext, criterion_pretext, device, 'pretext')
        logging.info(f'Epoch {epoch+1}/{NUM_EPOCHS}, Pretext Phase, Train Loss: {train_loss_pretext:.4f}, Val Loss: {val_loss_pretext:.4f}')

        

    for epoch in range(NUM_EPOCHS):
        # Train in aesthetic phase
        train_loss_aesthetic = train(model, train_loader_aesthetic, criterion_aesthetic, optimizer_aesthetic, device, 'aesthetic')
        val_loss_aesthetic = validate(model, val_loader_aesthetic, criterion_aesthetic, device, 'aesthetic')
        logging.info(f'Epoch {epoch+1}/{NUM_EPOCHS}, Aesthetic Phase, Train Loss: {train_loss_aesthetic:.4f}, Val Loss: {val_loss_aesthetic:.4f}')
    
    # Save model
    save_model(model, epoch, PATH_MODEL_RESULTS, 'aestheticNet')


if __name__ == "__main__":
    main()
    