import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("models/best-calculo-peso.pt")


def inference(image_bytes: bytes) -> list:
    """
    Recibe la imagen en bytes, realiza la inferencia y devuelve una lista de detecciones.

    Cada detección es un dict con:
      - label: nombre de la clase detectada
      - confidence: nivel de confianza (float)
      - bbox: [x1, y1, x2, y2]
    """
    # Convertir bytes a imagen OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Ejecutar inferencia
    results = model(img)

    # Parsear resultados a JSON-friendly
    detections = []
    for result in results:
        # Cada result.boxes contiene los bounding boxes
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model.names.get(cls_id, str(cls_id))
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

    return detections
