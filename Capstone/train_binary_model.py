import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models
import numpy as np

def train(x_train_arr, y_train_arr, x_val_arr, y_val_arr):
    batch = 32
    epochs = 5
    epochs_fine = 10
    lr = 1e-3
    lr_fine = 1e-5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_train = torch.from_numpy(y_train_arr).float().view(-1, 1)
    y_val = torch.from_numpy(y_val_arr).float().view(-1, 1)

    x_train = torch.from_numpy(x_train_arr).float()
    x_val = torch.from_numpy(x_val_arr).float()

    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)


    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  #Binary output
    model.to(device)

    loss_fn = nn.BCEWithLogitsLoss()  #binary cross entropy with logits
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()


    for param in model.parameters():
        param.requires_grad = True

    optimizer_fine = optim.Adam(model.parameters(), lr=lr_fine)

    for epoch in range(epochs_fine):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_fine.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer_fine.step()

        model.eval()
        val_corrects, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = (outputs > 0).float()
                val_corrects += (preds == labels).sum()
                total += labels.size(0)

        val_acc = val_corrects.double() / total
        print(f"Epoch {epoch+1}/{epochs_fine}, Validation Accuracy: {val_acc:.4f}")

    return model
