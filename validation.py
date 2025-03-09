import os
import yaml
from pathlib import Path
import cv2
import numpy as np

class DatasetValidator:
    def __init__(self, config_path='dataset.yaml'):
        self.config = self.load_config(config_path)
        
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def validate_structure(self):
        """Проверка структуры папок и файлов"""
        required_dirs = [
            Path(self.config['path']) / self.config['train'],
            Path(self.config['path']) / self.config['val'],
            Path(self.config['path']) / self.config['test']
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(f"Директория {dir_path} не найдена")
                
    def validate_annotations(self):
        """Проверка корректности аннотаций"""
        for split in ['train', 'val', 'test']:
            img_dir = Path(self.config['path']) / self.config[split]
            label_dir = img_dir.parent.parent / 'labels' / split
            
            for img_path in img_dir.glob('*.jpg'):
                label_path = label_dir / f"{img_path.stem}.txt"
                
                if not label_path.exists():
                    print(f"Предупреждение: отсутствует аннотация для {img_path}")
                    continue
                
                # Проверка формата аннотаций
                with open(label_path, 'r') as f:
                    for line in f:
                        try:
                            class_id, x, y, w, h = map(float, line.strip().split())
                            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                print(f"Ошибка: некорректные координаты в {label_path}")
                        except:
                            print(f"Ошибка: некорректный формат в {label_path}") 