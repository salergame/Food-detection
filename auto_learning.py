from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import shutil
import time
import json
from datetime import datetime
import threading

class AutoLearningSystem:
    def __init__(self):
        self.model = YOLO('C:/Users/Asus/runs/detect/food_detection_model4/weights/best.pt')
        self.frame_buffer = []  # Буфер для хранения кадров
        self.buffer_size = 50   # Максимальный размер буфера
        self.confidence_threshold = 0.7  # Порог уверенности для автоматической разметки
        self.auto_learning_enabled = False
        self.setup_directories()
        
    def setup_directories(self):
        """Создание необходимых директорий"""
        self.new_data_dir = Path('new_data')
        self.new_images_dir = self.new_data_dir / 'images'
        self.new_labels_dir = self.new_data_dir / 'labels'
        
        self.new_images_dir.mkdir(parents=True, exist_ok=True)
        self.new_labels_dir.mkdir(parents=True, exist_ok=True)
    
    def capture_and_learn(self):
        """Захват видео и автоматическое обучение"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Ошибка: Не удалось открыть камеру")
            return
        
        window_name = 'Auto-Learning System'
        cv2.namedWindow(window_name)
        
        # Добавляем трекбар для порога уверенности
        cv2.createTrackbar('Confidence', window_name, 
                          int(self.confidence_threshold * 100), 100, 
                          lambda x: setattr(self, 'confidence_threshold', x/100))
        
        frame_count = 0
        last_save_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Получаем предсказания
            results = self.model(frame)
            
            # Отрисовываем результаты
            annotated_frame = frame.copy()
            high_confidence_detections = False
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf > self.confidence_threshold:
                        high_confidence_detections = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        class_name = self.model.names[cls]
                        
                        # Рисуем бокс и подпись
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{class_name}: {conf:.2f}"
                        cv2.putText(annotated_frame, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Показываем статус автообучения
            status = "Auto-Learning: ON" if self.auto_learning_enabled else "Auto-Learning: OFF"
            cv2.putText(annotated_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Сохраняем кадр, если есть уверенные определения
            current_time = time.time()
            if (self.auto_learning_enabled and 
                high_confidence_detections and 
                current_time - last_save_time > 2):  # Сохраняем каждые 2 секунды
                
                self.save_frame_with_annotations(frame, results)
                last_save_time = current_time
            
            cv2.imshow(window_name, annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                # Включение/выключение автообучения
                self.auto_learning_enabled = not self.auto_learning_enabled
            elif key == ord('r'):
                # Запуск переобучения модели
                threading.Thread(target=self.retrain_model).start()
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_frame_with_annotations(self, frame, results):
        """Сохранение кадра и его аннотаций"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = self.new_images_dir / f"frame_{timestamp}.jpg"
        label_path = self.new_labels_dir / f"frame_{timestamp}.txt"
        
        # Сохраняем изображение
        cv2.imwrite(str(image_path), frame)
        
        # Сохраняем аннотации в формате YOLO
        with open(label_path, 'w') as f:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if float(box.conf[0]) > self.confidence_threshold:
                        # Конвертируем координаты в формат YOLO
                        x1, y1, x2, y2 = box.xyxy[0]
                        width = frame.shape[1]
                        height = frame.shape[0]
                        
                        # Нормализуем координаты
                        x_center = ((x1 + x2) / 2) / width
                        y_center = ((y1 + y2) / 2) / height
                        w = (x2 - x1) / width
                        h = (y2 - y1) / height
                        
                        cls = int(box.cls[0])
                        f.write(f"{cls} {x_center} {y_center} {w} {h}\n")
    
    def retrain_model(self):
        """Переобучение модели на новых данных"""
        try:
            # Создаем временную директорию для обучения
            temp_train_dir = Path('temp_train')
            temp_train_dir.mkdir(exist_ok=True)
            
            # Копируем новые данные
            shutil.copytree(self.new_images_dir, temp_train_dir / 'images', dirs_exist_ok=True)
            shutil.copytree(self.new_labels_dir, temp_train_dir / 'labels', dirs_exist_ok=True)
            
            # Создаем конфигурацию для дообучения
            data_yaml = {
                'path': str(temp_train_dir),
                'train': 'images',
                'val': 'images',  # Используем те же данные для валидации
                'names': self.model.names
            }
            
            yaml_path = temp_train_dir / 'data.yaml'
            with open(yaml_path, 'w') as f:
                json.dump(data_yaml, f)
            
            # Дообучаем модель
            self.model.train(
                data=str(yaml_path),
                epochs=10,  # Меньше эпох для быстрого дообучения
                imgsz=640,
                batch=16,
                name='auto_learning'
            )
            
            # Обновляем модель
            self.model = YOLO('runs/detect/auto_learning/weights/best.pt')
            
            print("Модель успешно переобучена!")
            
        except Exception as e:
            print(f"Ошибка при переобучении: {e}")
        finally:
            # Очищаем временные файлы
            if temp_train_dir.exists():
                shutil.rmtree(temp_train_dir)

class AutoLearningGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Auto-Learning Food Detection")
        self.setup_gui()
        
    def setup_gui(self):
        # Создаем и настраиваем элементы интерфейса
        ttk.Button(
            self.root,
            text="Запустить систему",
            command=self.start_system
        ).pack(pady=5, padx=10, fill=tk.X)
        
        ttk.Button(
            self.root,
            text="Выход",
            command=self.root.quit
        ).pack(pady=5, padx=10, fill=tk.X)
    
    def start_system(self):
        try:
            system = AutoLearningSystem()
            system.capture_and_learn()
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    gui = AutoLearningGUI()
    gui.run() 