from ultralytics import YOLO
import cv2

class CalorieDetector:
    def __init__(self):
        self.model = YOLO('runs/detect/food_detection_model4/weights/best.pt')
        self.calorie_dict = {
            'apple': 52,     # ккал на 100г
            'banana': 89,
            'orange': 47,
            'bread': 265,
            'cheese': 402
        }
    
    def estimate_portion(self, box_area, food_type):
        """Приблизительная оценка порции на основе площади бокса"""
        # Это упрощенная формула, можно улучшить
        normalized_area = box_area / (640 * 640)  # нормализуем по размеру изображения
        return normalized_area * 200  # предполагаем, что максимальная порция 200г
    
    def detect_and_calculate(self, image_path):
        # Загружаем изображение
        img = cv2.imread(image_path)
        height, width = img.shape[:2]
        
        # Получаем предсказания
        results = self.model(img)
        
        total_calories = 0
        detected_foods = []
        
        # Обрабатываем результаты
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Получаем данные бокса
                x1, y1, x2, y2 = box.xyxy[0]
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Получаем тип продукта
                food_type = self.model.names[cls]
                
                # Вычисляем площадь бокса
                box_area = (x2 - x1) * (y2 - y1)
                
                # Оцениваем порцию
                portion = self.estimate_portion(box_area, food_type)
                
                # Вычисляем калории
                calories = (self.calorie_dict[food_type] * portion) / 100
                
                detected_foods.append({
                    'food': food_type,
                    'portion': portion,
                    'calories': calories,
                    'confidence': conf
                })
                
                total_calories += calories
                
                # Отрисовка на изображении
                cv2.rectangle(img, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (0, 255, 0), 2)
                
                label = f"{food_type}: {calories:.0f} kcal"
                cv2.putText(img, label, 
                           (int(x1), int(y1) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)
        
        # Добавляем общую калорийность
        cv2.putText(img, 
                   f"Total: {total_calories:.0f} kcal",
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   1, (0, 255, 0), 2)
        
        # Сохраняем результат
        output_path = 'calories_' + image_path.split('/')[-1]
        cv2.imwrite(output_path, img)
        
        return detected_foods, total_calories

# Пример использования
if __name__ == "__main__":
    detector = CalorieDetector()
    foods, total = detector.detect_and_calculate("path/to/test/image.jpg")
    print(f"\nОбнаруженные продукты:")
    for food in foods:
        print(f"{food['food']}: {food['portion']:.0f}г - {food['calories']:.0f} ккал")
    print(f"\nОбщая калорийность: {total:.0f} ккал") 