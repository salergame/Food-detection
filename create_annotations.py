import cv2
import os
from pathlib import Path

def create_annotations():
    images_dir = Path('data/images/train')
    labels_dir = Path('data/labels/train')
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in images_dir.glob('*.jpg'):
        # Здесь нужно реализовать логику создания аннотаций
        # Формат YOLO: <class> <x_center> <y_center> <width> <height>
        # Значения нормализованы от 0 до 1
        label_path = labels_dir / f"{img_path.stem}.txt"
        
        # Пример записи аннотации:
        # with open(label_path, 'w') as f:
        #     f.write(f"0 0.5 0.5 0.8 0.8\n") 