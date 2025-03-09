from ultralytics import YOLO
import cv2
import numpy as np

def detect_food(image_path):
    # Загрузка обученной модели
    model = YOLO('runs/train/exp/weights/best.pt')
    
    # Загрузка изображения
    img = cv2.imread(image_path)
    
    # Получение предсказаний
    results = model(img)
    
    # Визуализация результатов
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Получение координат
            x1, y1, x2, y2 = box.xyxy[0]
            # Получение класса и уверенности
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Отрисовка бокса
            cv2.rectangle(img, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            
            # Добавление подписи
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.putText(img, label, 
                       (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2)
    
    # Сохранение результата
    cv2.imwrite('result.jpg', img)
    return results

# Пример использования
if __name__ == "__main__":
    results = detect_food('test_image.jpg')
    print("Обнаружены продукты:", 
          [model.names[int(box.cls[0])] for result in results for box in result.boxes]) 