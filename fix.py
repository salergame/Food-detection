import os
from pathlib import Path

# Создаем структуру папок
for split in ['train', 'val', 'test']:
    Path(f'data/images/{split}').mkdir(parents=True, exist_ok=True)
    Path(f'data/labels/{split}').mkdir(parents=True, exist_ok=True)

# Теперь выполняем разделение датасета
split_dataset(
    source_images='data/images/train',
    source_labels='data/labels/train',
    output_dir='data',
    split=(0.7, 0.15, 0.15)
)
