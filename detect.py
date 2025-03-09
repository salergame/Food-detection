from ultralytics import YOLO
import cv2

def detect_food(image_path):
    # Загружаем обученную модель
    model = YOLO('runs/detect/food_detection_model4/weights/best.pt')
    
    # Загружаем изображение
    img = cv2.imread(image_path)
    
    # Делаем предсказание
    results = model(img)
    
    # Визуализируем результаты
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Получаем координаты
            x1, y1, x2, y2 = box.xyxy[0]
            # Получаем класс и уверенность
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Рисуем бокс
            cv2.rectangle(img, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            
            # Добавляем подпись
            label = f"{model.names[cls]}: {conf:.2f}"
            cv2.putText(img, label, 
                       (int(x1), int(y1) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2)
    
    # Сохраняем результат
    output_path = 'detected_' + image_path.split('/')[-1]
    cv2.imwrite(output_path, img)
    return results

# Пример использования
if __name__ == "__main__":
    # Путь к тестовому изображению
    test_image = "path/to/test/image.jpg"
    results = detect_food(test_image) 