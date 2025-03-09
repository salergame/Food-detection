import torch
from ultralytics import YOLO
import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from tqdm import tqdm
import shutil
import random
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def check_dataset():
    """
    Проверяет корректность датасета
    """
    splits = ['train', 'val', 'test']
    for split in splits:
        img_dir = f'data/images/{split}'
        label_dir = f'data/labels/{split}'
        
        images = set(os.path.splitext(f)[0] for f in os.listdir(img_dir))
        labels = set(os.path.splitext(f)[0] for f in os.listdir(label_dir))
        
        print(f"\nПроверка {split} сплита:")
        print(f"Изображений: {len(images)}")
        print(f"Аннотаций: {len(labels)}")
        print(f"Изображения без аннотаций: {images - labels}")
        print(f"Аннотации без изображений: {labels - images}")

def split_dataset(source_images, source_labels, output_dir, split=(0.7, 0.15, 0.15)):
    """
    Разделяет датасет на train/val/test
    """
    # Создаем директории
    for split_name in ['train', 'val', 'test']:
        os.makedirs(f"{output_dir}/images/{split_name}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{split_name}", exist_ok=True)
    
    # Получаем список файлов
    image_files = [f for f in os.listdir(source_images) if f.endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(image_files)
    
    # Вычисляем размеры сплитов
    train_size = int(len(image_files) * split[0])
    val_size = int(len(image_files) * split[1])
    
    # Распределяем файлы
    splits = {
        'train': image_files[:train_size],
        'val': image_files[train_size:train_size + val_size],
        'test': image_files[train_size + val_size:]
    }
    
    # Копируем файлы
    for split_name, files in splits.items():
        for filename in files:
            # Копируем изображение
            shutil.copy2(
                os.path.join(source_images, filename),
                os.path.join(output_dir, 'images', split_name, filename)
            )
            
            # Копируем аннотацию
            label_file = os.path.splitext(filename)[0] + '.txt'
            if os.path.exists(os.path.join(source_labels, label_file)):
                shutil.copy2(
                    os.path.join(source_labels, label_file),
                    os.path.join(output_dir, 'labels', split_name, label_file)
                )

def prepare_dataset():
    """
    Подготовка и аугментация датасета
    """
    # Настройка аугментации
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
        A.GaussNoise(p=0.3),
        A.Rotate(limit=30, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Resize(640, 640)
    ])

    # Пути к папкам
    input_dir = "data/images/train"
    output_dir = "data/images/train_augmented"
    os.makedirs(output_dir, exist_ok=True)

    # Аугментация изображений
    for img_name in tqdm(os.listdir(input_dir)):
        if img_name.endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_dir, img_name)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Применяем аугментацию
            for i in range(3):  # создаем 3 аугментированные версии каждого изображения
                augmented = transform(image=image)
                aug_image = augmented['image']
                
                # Сохраняем аугментированное изображение
                aug_name = f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
                aug_path = os.path.join(output_dir, aug_name)
                Image.fromarray(aug_image).save(aug_path)

def train_model():
    """
    Обучение модели YOLO
    """
    # Инициализация модели
    model = YOLO('yolov8n.yaml')  # создаем новую модель

    # Настройка параметров обучения
    results = model.train(
        data='dataset.yaml',
        epochs=100,  # количество эпох
        imgsz=640,  # размер изображения
        batch=16,  # размер батча
        patience=20,  # ранняя остановка
        save=True,  # сохранение лучших весов
        device='0' if torch.cuda.is_available() else 'cpu',  # использование GPU если доступно
        workers=8,  # количество рабочих процессов
        pretrained=True,  # использование предварительно обученных весов
        optimizer='Adam',  # оптимизатор
        lr0=0.001,  # начальная скорость обучения
        weight_decay=0.0005,  # L2 регуляризация
        warmup_epochs=3,  # эпохи разогрева
        cos_lr=True,  # косинусное затухание скорости обучения
        name='food_detection_model'  # название эксперимента
    )

    return model

def validate_paths():
    required_paths = [
        'data/images/train',
        'data/images/val',
        'data/images/test',
        'data/labels/train',
        'data/labels/val',
        'data/labels/test'
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Путь {path} не существует")

if __name__ == "__main__":
    # Разделение датасета на train/val/test
    # Укажите пути к вашим папкам с изображениями и аннотациями
    split_dataset(
        source_images='data\images',  # замените на путь к вашим изображениям
        source_labels='data\labels',  # замените на путь к вашим аннотациям
        output_dir='data'
    )
    
    # Проверка корректности датасета
    check_dataset()
    
    # Подготовка датасета (аугментация)
    prepare_dataset()
    
    # Обучение модели
    model = train_model()