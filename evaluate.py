import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
import time
from train import build_model
from data_loader import get_dataloaders

def evaluate_model(model, test_loader, device):
    """Вычисляет метрики качества."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'confusion_matrix': cm.tolist(),
        'predictions': all_preds.tolist(),
        'labels': all_labels.tolist()
    }

def measure_inference_time(model, device, input_size=(1, 3, 224, 224), num_iter=100):
    """Измеряет среднее время инференса на CPU (в миллисекундах)."""
    model.eval()
    device_cpu = torch.device('cpu')
    model_cpu = model.to(device_cpu)
    dummy_input = torch.randn(input_size).to(device_cpu)
    for _ in range(10):
        _ = model_cpu(dummy_input)
    start = time.time()
    for _ in range(num_iter):
        _ = model_cpu(dummy_input)
    elapsed = (time.time() - start) / num_iter
    return elapsed * 1000  

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = 'data/PlantVillage'   
    batch_size = 32
    _, _, test_loader, classes = get_dataloaders(data_dir, batch_size)
    num_classes = len(classes)

    model_names = ['resnet50', 'densenet121', 'efficientnet_b0', 'mobilenet_v3', 'vit_b16']
    results = {}

    os.makedirs('runs', exist_ok=True)

    for name in model_names:
        print(f'\n=== Оценка модели: {name} ===')
        model_path = f'models/{name}_best.pth'
        if not os.path.exists(model_path):
            print(f'Модель {name} не найдена, пропускаем.')
            continue

        model = build_model(name, num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        metrics = evaluate_model(model, test_loader, device)

        inference_ms = measure_inference_time(model, device)
        metrics['inference_time_ms'] = inference_ms

        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        metrics['model_size_mb'] = size_mb

        results[name] = metrics

        cm = np.array(metrics['confusion_matrix'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix - {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(f'runs/cm_{name}.png')
        plt.close()

        print(f'Accuracy: {metrics["accuracy"]:.4f}')
        print(f'Precision (macro): {metrics["precision"]:.4f}')
        print(f'Recall (macro): {metrics["recall"]:.4f}')
        print(f'F1-score (macro): {metrics["f1"]:.4f}')
        print(f'Inference time (CPU): {inference_ms:.2f} ms')
        print(f'Model size: {size_mb:.2f} MB')

    with open('runs/evaluation_metrics.json', 'w') as f:
        json.dump(results, f, indent=4)

    print('\n Оценка завершена. Результаты сохранены в runs/evaluation_metrics.json')