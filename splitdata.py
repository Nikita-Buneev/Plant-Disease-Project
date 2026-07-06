import os
import shutil
from sklearn.model_selection import train_test_split

src = 'data/PlantVillage'
dst = 'data/PlantVillage'

for split in ['train', 'val', 'test']:
    split_path = os.path.join(dst, split)
    if os.path.exists(split_path):
        shutil.rmtree(split_path)
        print(f"Удалена {split_path}")

os.makedirs(os.path.join(dst, 'train'), exist_ok=True)
os.makedirs(os.path.join(dst, 'val'), exist_ok=True)
os.makedirs(os.path.join(dst, 'test'), exist_ok=True)

if not os.path.exists(src):
    raise FileNotFoundError(f"Папка {src} не найдена.")

exclude = {'train', 'val', 'test'}
classes = [d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)) and d not in exclude]
if not classes:
    raise ValueError(f"В {src} нет папок классов.")

print(f"Найдено классов: {len(classes)}")

for cls in classes:
    cls_path = os.path.join(src, cls)
    files = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
    if not files:
        print(f"Класс {cls} пуст, пропускаем")
        continue
    print(f"Класс {cls}: {len(files)} файлов")

    train_files, test_files = train_test_split(files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42)

    for split, file_list in [('train', train_files), ('val', val_files), ('test', test_files)]:
        split_dir = os.path.join(dst, split, cls)
        os.makedirs(split_dir, exist_ok=True)
        for fname in file_list:
            src_path = os.path.join(cls_path, fname)
            dst_path = os.path.join(split_dir, fname)
            try:
                shutil.copy(src_path, dst_path)
            except Exception as e:
                print(f"Ошибка: {e}")

    print(f"Класс {cls}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

print("Разбиение завершено.")