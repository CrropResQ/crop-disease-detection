# ResNet50 Experiment Results

This folder contains the ResNet50 implementation and experiment results for the Crop Disease Detection project.

## Folder Structure

### Best
Contains the best-performing model obtained during hyperparameter tuning.

- Validation Accuracy: **92.92%**
- Learning Rate: **1e-5**
- Batch Size: **32**
- Optimizer: **Adam**

### Final
Contains the final retrained model using the selected hyperparameters for further experiments and ensemble learning.

- Validation Accuracy: **90.17%**
- Learning Rate: **0.0005**
- Batch Size: **32**
- Optimizer: **Adam**

## Current Implementation

The `resnet50.py` file in the root of this folder is the latest implementation and should be used for future training and ensemble experiments.