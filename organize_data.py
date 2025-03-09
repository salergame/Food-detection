import os
import shutil
from pathlib import Path

def organize_dataset():
    # Пути
    source_dir = Path('OIDv4_ToolKit/OID/Dataset/train')  # откуда берем изображения
    target_images_dir = Path('data/images/train')  # куда складываем изображения
    
    # Создаем целевую директорию если её нет
    target_images_dir.mkdir(parents=True, exist_ok=True)
    
    # Проходим по всем классам (Apple, Banana и т.д.)
    for class_dir in source_dir.iterdir():
        if class_dir.is_dir():
            # Проходим по всем изображениям в папке класса
            for img_file in class_dir.glob('*.jpg'):
                # Копируем изображение в целевую папку
                shutil.copy2(
                    img_file,
                    target_images_dir / img_file.name
                )
                print(f"Скопирован файл: {img_file.name}")

if __name__ == "__main__":
    organize_dataset() 