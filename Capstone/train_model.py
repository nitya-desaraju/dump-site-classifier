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
    lr_fine =  1e-5
    num_classes = y_train_arr.shape[1]

    y_train_idx = np.argmax(y_train_arr, axis = 1)
    y_val_idx = np.argmax(y_val_arr, axis = 1)

    device = torch.device("cpu")
   
    x_train = torch.from_numpy(x_train_arr).float()
    y_train = torch.from_numpy(y_train_idx).long()
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    x_val = torch.from_numpy(x_val_arr).float()
    y_val = torch.from_numpy(y_val_idx).long()
    val_dataset = TensorDataset(x_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False)

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    for param in model.parameters():
        param.requires_grad = False
    
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
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
        for inputs,labels in train_loader:
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
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
        val_acc = val_corrects.double() / total

    return model

