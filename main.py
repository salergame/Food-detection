from pathlib import Path
import cv2
import torch
from validation import DatasetValidator
from calorie_calculator import CalorieCalculator
from visualizer import ResultVisualizer

def main():
    # Инициализация компонентов
    validator = DatasetValidator()
    calculator = CalorieCalculator()
    visualizer = ResultVisualizer()
    
    # Проверка датасета
    validator.validate_structure()
    validator.validate_annotations()
    
    # Загрузка обученной модели
    model = torch.hub.load('ultralytics/yolov5', 'custom', 
                          path='runs/train/exp/weights/best.pt')
    
    # Обработка изображения
    image_path = "test_image.jpg"
    img = cv2.imread(image_path)
    
    # Получение предсказаний
    results = model(img)
    detections = results.pred[0]
    
    # Расчет калорий
    total_calories, calorie_info = calculator.calculate_calories(
        detections, img.shape[:2])
    
    # Визуализация результатов
    output_img = visualizer.draw_detections(img, detections, calorie_info)
    
    # Сохранение результата
    cv2.imwrite("result.jpg", output_img)
    
    print(f"Общая калорийность: {total_calories:.1f} ккал")

if __name__ == "__main__":
    main() 