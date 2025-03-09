from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path
import numpy as np
import edge_tts
import asyncio
import pygame
import tempfile
import os
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont

class FoodDetector:
    def __init__(self):
        model_path = Path('C:/Users/Asus/runs/detect/food_detection_model4/weights/best.pt')
        if not model_path.exists():
            model_path = Path('best.pt')
            if not model_path.exists():
                raise FileNotFoundError("Файл модели не найден!")
        
        print(f"Загружаем модель из: {model_path}")
        self.model = YOLO(str(model_path))
        
        pygame.mixer.init()
        
        self.voice = "ru-RU-DmitryNeural" 
        # self.voice = "ru-RU-SvetlanaNeural" 
        
        self.temp_dir = Path(tempfile.gettempdir()) / "food_detection_audio"
        self.temp_dir.mkdir(exist_ok=True)
        
        for file in self.temp_dir.glob("*.mp3"):
            try:
                file.unlink()
                print(f"Удален старый аудиофайл: {file}")
            except Exception as e:
                print(f"Не удалось удалить файл {file}: {e}")
        
        self.calories = {
            'apple': {'calories': 52, 'name_ru': 'яблоко'},
            'banana': {'calories': 89, 'name_ru': 'банан'},
            'orange': {'calories': 47, 'name_ru': 'апельсин'},
            'bread': {'calories': 265, 'name_ru': 'хлеб'},
            'cheese': {'calories': 402, 'name_ru': 'сыр'}
        }
        
        # Словарь для отслеживания последнего оповещения
        self.last_announcement = {}
        self.announcement_cooldown = 3
        
        # Предварительная генерация аудио для всех продуктов
        asyncio.run(self.prepare_audio())
    
    async def prepare_audio(self):
        """Предварительная генерация аудио для всех продуктов"""
        try:
            # Проверяем доступные голоса
            voices = await edge_tts.list_voices()
            russian_voices = [v for v in voices if "RU" in v["ShortName"]]
            
            if not russian_voices:
                print("Не найдены русские голоса!")
                return
            
            print("Доступные русские голоса:")
            for v in russian_voices:
                print(f"- {v['ShortName']} ({v['Gender']})")
            
            # Важно! Проверяем, какой голос установлен ПЕРЕД проверкой его наличия
            print(f"Текущий выбранный голос: {self.voice}")
            
            # Проверяем, существует ли выбранный голос
            if not any(v["ShortName"] == self.voice for v in russian_voices):
                print(f"Голос {self.voice} не найден, используем первый доступный")
                self.voice = russian_voices[0]["ShortName"]
            
            print(f"Используем голос: {self.voice}")
            
            for food, info in self.calories.items():
                base_text = f"Обнаружен {info['name_ru']}, примерно {int(info['calories'])} калорий"
                
                file_path = self.temp_dir / f"{food}.mp3"
                try:
                    communicate = edge_tts.Communicate(base_text, self.voice)
                    await communicate.save(str(file_path))
                    print(f"Создан аудиофайл для {info['name_ru']}")
                except Exception as e:
                    print(f"Ошибка при создании аудио для {info['name_ru']}: {e}")
                
        except Exception as e:
            print(f"Ошибка при подготовке аудио: {e}")
    
    def announce_food(self, food_name, calories):
        """Озвучивание обнаруженного продукта"""
        current_time = datetime.now()
        
        # Проверяем время последнего оповещения
        if (food_name not in self.last_announcement or 
            current_time - self.last_announcement[food_name] > timedelta(seconds=self.announcement_cooldown)):
            
            # Воспроизводим предварительно сгенерированный звук
            audio_file = self.temp_dir / f"{food_name}.mp3"
            
            if audio_file.exists():
                pygame.mixer.music.load(str(audio_file))
                pygame.mixer.music.play()
                
                # Обновляем время последнего оповещения
                self.last_announcement[food_name] = current_time
    
    def process_frame(self, frame):
        """Обработка одного кадра"""
        # Получаем предсказания
        results = self.model(frame)
        
        total_calories = 0
        
        # Отрисовываем результаты
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Получаем координаты бокса
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Получаем класс и уверенность
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = self.model.names[cls]
                
                # Рисуем бокс
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Добавляем подпись с названием класса и уверенностью
                if class_name in self.calories:
                    label = f"{self.calories[class_name]['name_ru']}: {conf:.2f}"
                    # Используем шрифт, поддерживающий кириллицу
                    frame = put_russian_text(frame, label, (x1, y1-10))
                
                # Оцениваем калории
                if class_name in self.calories:
                    box_area = (x2 - x1) * (y2 - y1)
                    portion = (box_area / (frame.shape[0] * frame.shape[1])) * 200
                    cal = int((self.calories[class_name]['calories'] * portion) / 100)  # Округляем калории
                    total_calories += cal
                    cal_label = f"{cal} ккал"
                    frame = put_russian_text(frame, cal_label, (x1, y1-30))
                    
                    # Озвучиваем продукт, если уверенность высокая
                    if conf > 0.6:
                        self.announce_food(class_name, cal)
        
        # Добавляем общую калорийность
        total_label = f"Всего: {int(total_calories)} ккал"
        frame = put_russian_text(frame, total_label, (10, 30))
        
        return frame

    def webcam_detection(self):
        """Запуск определения в реальном времени через веб-камеру"""
        cap = cv2.VideoCapture(0)  # 0 - встроенная камера, 1 - внешняя
        
        # Проверяем, открылась ли камера
        if not cap.isOpened():
            print("Ошибка: Не удалось открыть камеру")
            return
        
        print("Нажмите 'q' для выхода, 's' для сохранения кадра")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка: Не удалось получить кадр")
                break
            
            # Обрабатываем кадр
            processed_frame = self.process_frame(frame)
            
            # Показываем FPS
            frame_count += 1
            cv2.putText(processed_frame, f"Frame: {frame_count}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # Показываем результат
            cv2.imshow('Food Detection (Press q to quit, s to save)', processed_frame)
            
            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Сохраняем кадр
                output_path = Path('results') / f"webcam_frame_{frame_count}.jpg"
                output_path.parent.mkdir(exist_ok=True)
                cv2.imwrite(str(output_path), processed_frame)
                print(f"Кадр сохранен в: {output_path}")
        
        # Освобождаем ресурсы
        cap.release()
        cv2.destroyAllWindows()

    def detect_image(self, image_path):
        """Определение на одном изображении"""
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Ошибка: Не удалось загрузить изображение {image_path}")
            return None
        
        processed_img = self.process_frame(img.copy())
        
        # Сохраняем результат
        output_path = Path('results') / f"detected_{Path(image_path).name}"
        output_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(output_path), processed_img)
        print(f"Результат сохранен в: {output_path}")
        
        return processed_img

class DetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Food Detection")
        self.setup_gui()
        
    def setup_gui(self):
        # Создаем кнопки
        ttk.Button(
            self.root,
            text="Открыть изображение",
            command=self.detect_image
        ).pack(pady=5, padx=10, fill=tk.X)
        
        ttk.Button(
            self.root,
            text="Запустить веб-камеру",
            command=self.start_webcam
        ).pack(pady=5, padx=10, fill=tk.X)
        
        ttk.Button(
            self.root,
            text="Выход",
            command=self.root.quit
        ).pack(pady=5, padx=10, fill=tk.X)
    
    def detect_image(self):
        file_path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                detector = FoodDetector()
                result_img = detector.detect_image(file_path)
                
                if result_img is not None:
                    cv2.imshow('Результат', result_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
            except Exception as e:
                print(f"Произошла ошибка: {e}")
    
    def start_webcam(self):
        try:
            detector = FoodDetector()
            detector.webcam_detection()
        except Exception as e:
            print(f"Произошла ошибка: {e}")
    
    def run(self):
        self.root.mainloop()

def put_russian_text(img, text, position, font_size=32, color=(0, 255, 0)):
    """Добавление русского текста на изображение"""
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    
    # Используем системный шрифт, поддерживающий кириллицу
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
        
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

if __name__ == "__main__":
    gui = DetectionGUI()
    gui.run()
