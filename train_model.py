import os
from pathlib import Path
from ultralytics import YOLO
import shutil

def prepare_directories():
    """Создание необходимых директорий"""
    # Сначала создаем временную папку для хранения оригинальных данных
    Path('data/original/images').mkdir(parents=True, exist_ok=True)
    Path('data/original/labels').mkdir(parents=True, exist_ok=True)
    
    # Затем создаем папки для разделенного датасета
    for split in ['train', 'val', 'test']:
        Path(f'data/images/{split}').mkdir(parents=True, exist_ok=True)
        Path(f'data/labels/{split}').mkdir(parents=True, exist_ok=True)

def move_original_data():
    """Перемещаем оригинальные данные во временную папку"""
    # Перемещаем изображения
    source_images = Path('data/images/train')
    if source_images.exists():
        for img in source_images.glob('*.*'):
            shutil.move(str(img), f'data/original/images/{img.name}')
            
    # Перемещаем аннотации
    source_labels = Path('data/labels/train')
    if source_labels.exists():
        for label in source_labels.glob('*.txt'):
            shutil.move(str(label), f'data/original/labels/{label.name}')

def split_dataset(split=(0.7, 0.15, 0.15)):
    """Разделяет датасет на train/val/test"""
    import random
    
    # Получаем список файлов из временной папки
    image_files = list(Path('data/original/images').glob('*.*'))
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
        for img_path in files:
            # Копируем изображение
            shutil.copy2(
                str(img_path),
                f'data/images/{split_name}/{img_path.name}'
            )
            
            # Копируем аннотацию если она существует
            label_path = Path('data/original/labels') / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(
                    str(label_path),
                    f'data/labels/{split_name}/{label_path.name}'
                )

def train_model():
    """Обучение модели"""
    # Получаем абсолютный путь к текущей директории
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    model = YOLO('yolov8n.yaml')
    results = model.train(
        data=os.path.join(current_dir, 'dataset.yaml'),  # используем полный путь
        epochs=100,
        imgsz=640,
        batch=16,
        name='food_detection_model',
        device='cpu'  # явно указываем использование CPU
    )
    return model

def print_dataset_info():
    """Выводит информацию о датасете"""
    print("\nПроверка структуры датасета:")
    base_dir = Path('data')
    
    for split in ['train', 'val', 'test']:
        img_dir = base_dir / 'images' / split
        label_dir = base_dir / 'labels' / split
        
        if img_dir.exists() and label_dir.exists():
            img_count = len(list(img_dir.glob('*.*')))
            label_count = len(list(label_dir.glob('*.txt')))
            print(f"\n{split}:")
            print(f"  Изображений: {img_count}")
            print(f"  Аннотаций: {label_count}")
        else:
            print(f"\n{split} - директории не существуют!")

def verify_and_fix_data():
    """Проверяет и исправляет структуру данных"""
    orig_img_dir = Path('data/original/images')
    orig_label_dir = Path('data/original/labels')
    
    if not orig_img_dir.exists() or not orig_label_dir.exists():
        print("Ошибка: отсутствуют оригинальные данные!")
        return False
    
    # Очищаем существующие split-директории
    for split in ['train', 'val', 'test']:
        for dtype in ['images', 'labels']:
            split_dir = Path(f'data/{dtype}/{split}')
            if split_dir.exists():
                for file in split_dir.glob('*.*'):
                    file.unlink()
    
    # Заново разделяем датасет
    split_dataset()
    
    # Проверяем результат
    print_dataset_info()
    return True

if __name__ == "__main__":
    # 1. Создаем директории
    prepare_directories()
    
    # 2. Перемещаем оригинальные данные (только если они еще не перемещены)
    if not Path('data/original/images').exists() or len(list(Path('data/original/images').glob('*.*'))) == 0:
        move_original_data()
    
    # 3. Проверяем и исправляем структуру данных
    if verify_and_fix_data():
        print("\nНачинаем обучение...")
        # 4. Обучаем модель
        model = train_model()
    else:
        print("\nОшибка в структуре данных. Обучение невозможно.") 