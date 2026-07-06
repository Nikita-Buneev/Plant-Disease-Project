import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import build_model

def load_best_model(model_name='efficientnet_b0', num_classes=15, device='cpu', model_dir='models'):
    model = build_model(model_name, num_classes).to(device)
    model_path = os.path.join(model_dir, f'{model_name}_best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict_image(model, image_path, class_names, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        top_probs, top_idxs = torch.topk(probs, 3, dim=1)
    top_probs = top_probs.cpu().numpy().flatten()
    top_idxs = top_idxs.cpu().numpy().flatten()
    results = [(class_names[idx], float(prob)) for idx, prob in zip(top_idxs, top_probs)]
    return results