import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from typing import Tuple
import csv
import pandas as pd

class CNN(nn.Module):
    def __init__(self, model_type, use_extra_params):
        super(CNN, self).__init__()
        self.model_type = model_type
        self.use_extra_params = use_extra_params

        if model_type == 'resnet18':
            self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.feature_extractor.fc = nn.Identity()
            feature_dim = 512
        elif model_type == 'efficientnet_b0':
            self.feature_extractor = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
            self.feature_extractor.classifier = nn.Identity()
            feature_dim = 1280
        elif model_type == 'simple_cnn':
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
            feature_dim = 64
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.reduce = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU()
        )

        if self.use_extra_params:
            self.param_fc_2 = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Linear(16, 64),
                nn.ReLU()
            )
            self.param_fc_3 = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 64),
                nn.ReLU()
            )
            self.fc1 = nn.Sequential(
                nn.Linear(128 + 64, 64),
                nn.ReLU(),
            )
        else:
            self.fc1 = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(),
            )

        self.fc2 = nn.Linear(64, 1)

    def forward(self, x, extra_params = None):
        x = self.feature_extractor(x)
        x = self.reduce(x)

        if self.use_extra_params:
            if extra_params is None:
                raise ValueError("extra_params is required but not provided.")
            if extra_params.shape[1] == 2:
                p = self.param_fc_2(extra_params)
            elif extra_params.shape[1] == 3:
                p = self.param_fc_3(extra_params)
            else:
                raise ValueError('error - extra_params shape')
            x = torch.cat([x, p], dim=1)

        out = self.fc1(x)
        out = self.fc2(out)
        return out.squeeze(1)

def train(model: CNN, train_loader: DataLoader, criterion, optimizer, device)->float:
    model.train()
    total_loss = 0.0
    total = 0

    for inputs, labels, extra_params in tqdm(train_loader, desc = "Training", leave = False):
        inputs, labels, extra_params = inputs.to(device), labels.to(device), extra_params.to(device)
        
        outputs = model(inputs, extra_params)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        total += inputs.size(0)
    avg_loss = total_loss / total
    return avg_loss


def validate(model: CNN, val_loader: DataLoader, criterion, device)->Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    all_labels = []
    all_outputs = []

    with torch.no_grad():
        for inputs, labels, extra_params in tqdm(val_loader, desc = "Validating", leave = False):
            inputs, labels, extra_params = inputs.to(device), labels.to(device), extra_params.to(device)

            outputs = model(inputs, extra_params)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())
    avg_loss = total_loss / total
    mae = mean_absolute_error(all_labels, all_outputs)
    return avg_loss, mae

def test(model: CNN, test_loader: DataLoader, criterion, device, target):
    model.eval()
    predictions = []
    y_true = []
    y_pred = []
    
    df = pd.read_csv('Earthquake.csv')
    label_mag = dict(zip(df['Image'], df['Mag']))
    label_lon = dict(zip(df['Image'], df['Lon']))
    label_lat = dict(zip(df['Image'], df['Lat']))
    label_dep = dict(zip(df['Image'], df['Depth']))

    with torch.no_grad():
        for images, image_names, extra_params in test_loader:
            images, extra_params = images.to(device), extra_params.to(device)
            outputs = model(images, extra_params)

            predicted = outputs

            for name, pred in zip(image_names, predicted.cpu()):
                if target == 'mag':
                    predictions.append({'id': name, 'Mag': label_mag[name], 'prediction': round(pred.item(), 1)})
                    y_true.append(label_mag[name]),
                    y_pred.append(round(pred.item(), 1))
                elif target == 'dep':
                    predictions.append({'id': name, 'Dep': label_dep[name], 'prediction': round(pred.item(), 1)})
                    y_true.append(label_dep[name]),
                    y_pred.append(round(pred.item(), 1))
                elif target == 'lon':
                    predictions.append({'id': name, 'Lon': label_lon[name], 'prediction': round(pred.item(), 3)})
                    y_true.append(label_lon[name]),
                    y_pred.append(round(pred.item(), 2))
                else:
                    predictions.append({'id': name, 'Lat': label_lat[name], 'prediction': round(pred.item(), 3)})
                    y_true.append(label_lat[name]),
                    y_pred.append(round(pred.item(), 2))
    MAE = round(mean_absolute_error(y_true, y_pred), 3)
    with open(f'Prediction({target}) - MAE = {MAE}.csv', mode = 'w', newline = '') as csvfile:
        if target == 'mag':
            fieldnames = ['id', 'Mag', 'prediction']
        elif target == 'dep':
            fieldnames = ['id', 'Dep', 'prediction']
        elif target == 'lon':
            fieldnames = ['id', 'Lon', 'prediction']
        else:
            fieldnames = ['id', 'Lat', 'prediction']
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)
    print(f"Predictions saved, with MAE = {MAE}'")
    return MAE