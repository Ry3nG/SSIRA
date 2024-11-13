# SSIRA (Self-Supervised Image Reconstruction and Aesthetics Assessment)
> My URECA project on Self-Supervised-Image-Reconstruction-and-Aesthetics-Assessment

A deep learning project that combines self-supervised learning with aesthetic assessment of images using a two-phase training approach.

## Project Overview

SSIRA uses a novel architecture (AestheticNet) that learns image features through self-supervised reconstruction before fine-tuning for aesthetic score prediction. The model consists of:
- An encoder based on ResNet50
- A decoder for image reconstruction
- A regression head for aesthetic score prediction

## Architecture

The training process occurs in two phases:
1. **Pretext Phase**: The model learns to reconstruct images using an encoder-decoder architecture
2. **Aesthetic Phase**: The learned features are used to predict aesthetic scores through a regression head

### Key Components:
- **Encoder**: Modified ResNet50 for feature extraction
- **Decoder**: Transposed convolution layers for image reconstruction
- **Regression Head**: Multi-layer perceptron for aesthetic score prediction

## Datasets

The model trains on two datasets:
- **TAD66K**: Used for self-supervised pretext training
- **AVA Dataset**: Used for aesthetic score prediction training

## Requirements

- PyTorch
- torchvision
- PIL
- pandas
- matplotlib
- tensorboard

## Training

To train the model:

```bash
python train.py [arguments]
```

### Important Arguments:
```
--batch_size            Batch size for training (default: 32)
--pretext_num_epochs    Number of epochs for pretext training
--aes_num_epochs       Number of epochs for aesthetic training
--learning_rate_pretext Learning rate for pretext phase
--learning_rate_aesthetic Learning rate for aesthetic phase
--checkpoint_path      Path to load a saved checkpoint
```

## Model Features

- Mixed precision training for efficiency
- Learning rate scheduling
- Checkpoint saving and loading
- TensorBoard integration for training visualization
- Multi-GPU support through DataParallel

## Training Visualization

The training process generates:
- Loss plots for both phases
- TensorBoard logs
- Detailed training logs
- CSV files with training metrics

## Project Structure

```
├── data/
│   ├── dataset.py          # Dataset classes
│   └── dataset_split.py    # Dataset splitting utilities
├── models/
│   ├── aestheticNet.py     # Main model architecture
│   ├── encoder.py          # ResNet50-based encoder
│   └── decoder.py          # Decoder architecture
├── utils/
│   ├── losses.py           # Loss functions
│   ├── transforms.py       # Custom transformations
│   └── constants.py        # Project constants
└── train.py                # Training script
```

## Future Work

- Implementation of global and local feature extraction
- Integration of attention mechanisms
- Adaptive feature integration
- Enhanced self-supervised learning strategies
