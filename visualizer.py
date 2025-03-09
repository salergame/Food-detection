import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

class ResultVisualizer:
    def __init__(self):
        self.colors = np.random.randint(0, 255, size=(80, 3))
    
    def draw_detections(self, image, detections, calorie_info):
        """Отрисовка обнаруженных объектов и информации о калориях"""
        img = image.copy()
        
        for det, info in zip(detections, calorie_info):
            bbox = det[:4].astype(int)
            class_id = int(det[5])
            conf = det[4]
            
            color = self.colors[class_id]
            
            # Рисуем bbox
            cv2.rectangle(img, bbox[:2], bbox[2:], color, 2)
            
            # Добавляем текст
            label = f"{info['product']}: {info['calories']:.1f} kcal"
            cv2.putText(img, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Добавляем общую информацию
        total_calories = sum(info['calories'] for info in calorie_info)
        cv2.putText(img, f"Total: {total_calories:.1f} kcal",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img
    
    def plot_statistics(self, history):
        """Построение графиков обучения"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='train')
        plt.plot(history['val_loss'], label='val')
        plt.title('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['map'], label='mAP')
        plt.title('mAP')
        plt.legend()
        
        plt.tight_layout()
        plt.show() 