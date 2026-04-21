import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import transforms 
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings
from dataset import HAM10000Dataset
from transformers import ViTForImageClassification
import numpy as np

warnings.filterwarnings("ignore")

IMG_DIR = "/path/to/HAM10000"  # 替换为你的路径
CSV_FILE = "/path/to//HAM10000_metadata.csv"# 替换为你的路径
BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

full_dataset = HAM10000Dataset(CSV_FILE, IMG_DIR, transform=train_transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = val_transform  

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

num_classes = full_dataset.num_classes

model = ViTForImageClassification.from_pretrained(
    "path/to/vit-base-patch16-224",# 替换为你的路径
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)
model.to(DEVICE)

labels = [full_dataset.meta.iloc[i]['label'] for i in range(len(full_dataset))]
class_weights = compute_class_weight(
    'balanced',
    classes=np.arange(num_classes), 
    y=labels
)
class_weights = torch.FloatTensor(class_weights).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights) #加权交叉熵
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(pixel_values=images)  
        loss = criterion(outputs.logits, labels) 
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(pixel_values=images)
            preds = outputs.logits.argmax(dim=1)  
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f}")


print("\nFinal Classification Report:")
target_names = list(full_dataset.label_map.keys())
print(classification_report(all_labels, all_preds, target_names=target_names))