import yaml
from pathlib import Path
import cv2
import numpy as np

class CalorieCalculator:
    def __init__(self, config_path='dataset.yaml'):
        self.config = self.load_config(config_path)
        self.model = None
        
    def load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def estimate_portion_size(self, bbox_area, reference_size=100):
        """Оценка размера порции на основе площади bbox"""
        # Это упрощенная формула, можно улучшить
        return (bbox_area * reference_size) / (640 * 640)
    
    def calculate_calories(self, detections, image_size):
        total_calories = 0
        results = []
        
        for det in detections:
            class_id = int(det[5])
            confidence = det[4]
            bbox = det[:4]  # x1, y1, x2, y2
            
            # Получаем название продукта
            product_name = self.config['names'][class_id]
            
            # Вычисляем площадь bbox
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            
            # Оцениваем размер порции
            portion = self.estimate_portion_size(bbox_area)
            
            # Вычисляем калории
            calories = (self.config['calories'][product_name] * portion) / 100
            
            results.append({
                'product': product_name,
                'calories': calories,
                'portion': portion,
                'confidence': confidence
            })
            
            total_calories += calories
            
        return total_calories, results 