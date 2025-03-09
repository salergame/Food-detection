import cv2
import os
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from train import split_dataset, train_model

class LabelingTool:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Инструмент разметки")
        
        # Список классов продуктов
        self.classes = {
            0: "apple",
            1: "banana",
            2: "orange",
            3: "bread",
            4: "cheese"
            # Добавьте свои классы
        }
        
        self.current_class = tk.StringVar(value="apple")
        self.current_image_path = None
        self.image_paths = []
        self.current_image_index = 0
        
        # Добавляем счетчик
        self.counter_var = tk.StringVar(value="0/0")
        self.labeled_count = 0
        
        self.last_labeled_index = 0
        
        self.boxes = []  # Список для хранения всех боксов
        
        self.setup_ui()
        
    def setup_ui(self):
        # Верхняя панель с кнопками
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill='x', padx=5, pady=5)
        
        # Добавляем счетчик в интерфейс
        self.counter_label = ttk.Label(
            control_frame,
            textvariable=self.counter_var,
            font=('Arial', 12, 'bold')
        )
        self.counter_label.pack(side='right', padx=10)
        
        # Кнопка загрузки папки
        ttk.Button(
            control_frame, 
            text="Открыть папку", 
            command=self.load_image_folder
        ).pack(side='left', padx=5)
        
        # Выпадающий список классов
        ttk.Combobox(
            control_frame,
            textvariable=self.current_class,
            values=list(self.classes.values())
        ).pack(side='left', padx=5)
        
        # Кнопки навигации
        ttk.Button(
            control_frame,
            text="Предыдущее",
            command=self.prev_image
        ).pack(side='left', padx=5)
        
        ttk.Button(
            control_frame,
            text="Следующее",
            command=self.next_image
        ).pack(side='left', padx=5)
        
        # Новая кнопка для перехода к последней размеченной
        ttk.Button(
            control_frame,
            text="К последней размеченной",
            command=self.goto_last_labeled
        ).pack(side='left', padx=5)
        
        # Добавляем кнопку очистки разметки
        ttk.Button(
            control_frame,
            text="Очистить разметку",
            command=self.clear_current_labels
        ).pack(side='left', padx=5)
        
        # Холст для изображения
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack(expand=True, fill='both')
        
        # Привязка событий мыши
        self.canvas.bind("<Button-1>", self.start_bbox)
        self.canvas.bind("<B1-Motion>", self.draw_bbox)
        self.canvas.bind("<ButtonRelease-1>", self.end_bbox)
        
        # Добавляем привязку правой кнопки мыши
        self.canvas.bind("<Button-3>", self.remove_bbox)
        
        self.start_x = None
        self.start_y = None
        
    def find_last_labeled_index(self):
        """Находит индекс последнего размеченного изображения"""
        label_dir = Path("data/labels/train")
        for i in range(len(self.image_paths) - 1, -1, -1):
            if (label_dir / f"{self.image_paths[i].stem}.txt").exists():
                return i
        return 0
    
    def goto_last_labeled(self):
        """Переходит к последнему размеченному изображению"""
        if not self.image_paths:
            return
            
        last_index = self.find_last_labeled_index()
        # Переходим к следующему после последнего размеченного
        target_index = min(last_index + 1, len(self.image_paths) - 1)
        self.current_image_index = target_index
        self.load_current_image()
        self.update_counter()
        
        # Показываем информационное сообщение
        if target_index == last_index + 1:
            tk.messagebox.showinfo(
                "Информация",
                f"Переход к следующему неразмеченному изображению (#{target_index + 1})"
            )
    
    def load_image_folder(self):
        folder_path = filedialog.askdirectory(title="Выберите папку с изображениями")
        if folder_path:
            self.image_paths = list(Path(folder_path).glob("*.jpg"))
            self.current_image_index = 0
            
            # Подсчет уже размеченных изображений
            label_dir = Path("data/labels/train")
            self.labeled_count = sum(1 for img in self.image_paths 
                                   if (label_dir / f"{img.stem}.txt").exists())
            
            # Находим последнее размеченное изображение
            last_labeled = self.find_last_labeled_index()
            self.current_image_index = min(last_labeled + 1, len(self.image_paths) - 1)
            
            self.update_counter()
            self.load_current_image()
            
            # Показываем информацию о загруженных изображениях
            tk.messagebox.showinfo(
                "Загрузка завершена",
                f"Загружено {len(self.image_paths)} изображений\n"
                f"Размечено: {self.labeled_count}\n"
                f"Переход к изображению #{self.current_image_index + 1}"
            )
    
    def update_counter(self):
        total = len(self.image_paths)
        self.counter_var.set(f"Размечено: {self.labeled_count}/{total} "
                           f"({(self.labeled_count/total*100):.1f}%)")
    
    def load_current_image(self):
        if not self.image_paths:
            return
            
        self.current_image_path = self.image_paths[self.current_image_index]
        image = Image.open(self.current_image_path)
        
        # Масштабирование изображения под размер холста
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        image = image.resize((canvas_width, canvas_height), Image.LANCZOS)
        
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor='nw', image=self.photo)
        
        # Загрузка существующих аннотаций
        self.load_annotations()
    
    def load_annotations(self):
        if not self.current_image_path:
            return
            
        self.boxes = []  # Очищаем список боксов
        label_path = Path(f"data/labels/train/{self.current_image_path.stem}.txt")
        if label_path.exists():
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    
                    # Преобразование координат из формата YOLO в пиксели
                    canvas_width = self.canvas.winfo_width()
                    canvas_height = self.canvas.winfo_height()
                    
                    x1 = int((x_center - width/2) * canvas_width)
                    y1 = int((y_center - height/2) * canvas_height)
                    x2 = int((x_center + width/2) * canvas_width)
                    y2 = int((y_center + height/2) * canvas_height)
                    
                    # Сохраняем информацию о боксе
                    box = {
                        'coords': (x1, y1, x2, y2),
                        'class_id': int(class_id),
                        'yolo_coords': (x_center, y_center, width, height)
                    }
                    self.boxes.append(box)
                    
                    # Отрисовка существующих боксов
                    self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", tags="box")
    
    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.load_current_image()
            self.update_counter()
    
    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_current_image()
            self.update_counter()
    
    def start_bbox(self, event):
        self.start_x = event.x
        self.start_y = event.y
        
    def draw_bbox(self, event):
        if self.start_x and self.start_y:
            self.canvas.delete("temp_box")
            self.canvas.create_rectangle(
                self.start_x, self.start_y,
                event.x, event.y,
                outline="red",
                tags="temp_box"
            )
    
    def end_bbox(self, event):
        if not self.current_image_path:
            return
            
        # Получаем размеры холста
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Преобразование координат в формат YOLO
        x_center = (self.start_x + event.x) / (2 * canvas_width)
        y_center = (self.start_y + event.y) / (2 * canvas_height)
        width = abs(event.x - self.start_x) / canvas_width
        height = abs(event.y - self.start_y) / canvas_height
        
        # Обновляем счетчик, если это первая разметка для изображения
        label_path = Path("data/labels/train") / f"{self.current_image_path.stem}.txt"
        if not label_path.exists():
            self.labeled_count += 1
            self.update_counter()
        
        # Добавляем новый бокс в список
        box = {
            'coords': (self.start_x, self.start_y, event.x, event.y),
            'class_id': list(self.classes.keys())[
                list(self.classes.values()).index(self.current_class.get())
            ],
            'yolo_coords': (x_center, y_center, width, height)
        }
        self.boxes.append(box)
        
        # Записываем аннотацию
        label_dir = Path("data/labels/train")
        label_dir.mkdir(parents=True, exist_ok=True)
        
        with open(label_path, "a") as f:
            f.write(f"{box['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Отрисовываем постоянный бокс
        self.canvas.create_rectangle(
            self.start_x, self.start_y,
            event.x, event.y,
            outline="green",
            tags="box"
        )
    
    def remove_bbox(self, event):
        """Удаление бокса по клику правой кнопкой мыши"""
        x, y = event.x, event.y
        
        # Проверяем каждый бокс
        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = box['coords']
            # Если клик внутри бокса
            if x1 <= x <= x2 and y1 <= y <= y2:
                # Удаляем бокс из списка
                self.boxes.pop(i)
                # Перерисовываем все боксы
                self.redraw_boxes()
                # Сохраняем обновленную разметку
                self.save_annotations()
                break
    
    def clear_current_labels(self):
        """Очистка всей разметки текущего изображения"""
        if not self.current_image_path:
            return
            
        if tk.messagebox.askyesno("Подтверждение", 
                                 "Вы уверены, что хотите удалить всю разметку для этого изображения?"):
            self.boxes = []
            self.canvas.delete("box")
            
            # Удаляем файл разметки
            label_path = Path(f"data/labels/train/{self.current_image_path.stem}.txt")
            if label_path.exists():
                label_path.unlink()
                self.labeled_count -= 1
                self.update_counter()
    
    def redraw_boxes(self):
        """Перерисовка всех боксов"""
        self.canvas.delete("box")
        for box in self.boxes:
            x1, y1, x2, y2 = box['coords']
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="green", tags="box")
    
    def save_annotations(self):
        """Сохранение обновленной разметки в файл"""
        if not self.current_image_path:
            return
            
        label_path = Path(f"data/labels/train/{self.current_image_path.stem}.txt")
        
        if self.boxes:
            with open(label_path, 'w') as f:
                for box in self.boxes:
                    x_center, y_center, width, height = box['yolo_coords']
                    class_id = box['class_id']
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        else:
            # Если боксов нет, удаляем файл разметки
            if label_path.exists():
                label_path.unlink()
                self.labeled_count -= 1
                self.update_counter()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    tool = LabelingTool()
    tool.run()

# Разделение датасета
split_dataset(
    source_images='data/images/train',  # папка с размеченными изображениями
    source_labels='data/labels/train',  # папка с аннотациями
    output_dir='data',
    split=(0.7, 0.15, 0.15)  # соотношение train/val/test
)

# Обучение модели
model = train_model() 