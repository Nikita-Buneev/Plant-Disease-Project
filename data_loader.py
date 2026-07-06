import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG'}

class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_path = os.path.join(self.root_dir, cls)
            for fname in os.listdir(cls_path):
                ext = os.path.splitext(fname)[1]
                if ext in ALLOWED_EXTENSIONS:
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls]))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            raise RuntimeError(f"Не удалось открыть {img_path}")
        image = np.array(image)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image, label

def get_dataloaders(data_dir, batch_size=32):
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.RandomRotate90(),
        A.HorizontalFlip(),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15),
        A.ColorJitter(brightness=0.1, contrast=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    dataset_train = PlantDiseaseDataset(data_dir, 'train', transform=train_transform)
    dataset_val = PlantDiseaseDataset(data_dir, 'val', transform=val_transform)
    dataset_test = PlantDiseaseDataset(data_dir, 'test', transform=val_transform)

    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, dataset_train.classes