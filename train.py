import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm
import os
import json
from data_loader import get_dataloaders

def train_model(model_name, model, train_loader, val_loader, device, epochs=30, lr=1e-4):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    best_acc = 0.0
    best_model_path = f'models/{model_name}_best.pth'
    history = {'train_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'{model_name} Epoch {epoch+1}'):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = correct / total
        scheduler.step(val_acc)
        history['train_loss'].append(running_loss / len(train_loader))
        history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    return best_acc, history

def build_model(model_name, num_classes):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'mobilenet_v3':
        model = models.mobilenet_v3_large(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == 'vit_b16':
        from transformers import ViTForImageClassification
        vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_classes)
        class ViTWrapper(nn.Module):
            def __init__(self, vit_model):
                super().__init__()
                self.vit_model = vit_model
            def forward(self, x):
                return self.vit_model(pixel_values=x).logits
        model = ViTWrapper(vit)
    else:
        raise ValueError('Unknown model')
    return model

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data/PlantVillage'
    batch_size = 32
    train_loader, val_loader, _, classes = get_dataloaders(data_dir, batch_size)
    num_classes = len(classes)

    models_to_train = ['resnet50', 'densenet121', 'efficientnet_b0', 'mobilenet_v3', 'vit_b16']
    results = {}
    for name in models_to_train:
        print(f'Training {name}...')
        model = build_model(name, num_classes).to(device)
        best_acc, history = train_model(name, model, train_loader, val_loader, device, epochs=10)  
        results[name] = {'best_acc': best_acc, 'history': history}
        print(f'{name} best validation accuracy: {best_acc:.4f}')

    os.makedirs('runs', exist_ok=True)
    with open('runs/training_results.json', 'w') as f:
        json.dump(results, f, indent=4)