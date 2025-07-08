from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn

from app.model import inference

app = FastAPI(
    title="YOLO Object Detection API",
    description="Microservicio de detección de objetos usando YOLO y FastAPI",
    version="1.0.0"
)

# Configuración de CORS
origins = [
    "*"  # Permite todas las URLs; restringe aquí si tienes dominios específicos
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos de Pydantic para la respuesta


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]


class DetectionResponse(BaseModel):
    filename: str
    detections: List[Detection]


@app.post("/detect/", response_model=DetectionResponse)
async def detect_objects(file: UploadFile = File(...)):
    """
    Endpoint para subir una imagen y obtener las detecciones.
    Recibe una imagen como multipart/form-data y devuelve un JSON con las detecciones.
    """
    # Leer contenido de la imagen
    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error al leer el archivo: {e}")

    # Ejecutar inferencia
    try:
        detections = inference(image_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error durante la inferencia: {e}")

    return DetectionResponse(filename=file.filename, detections=detections)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", host="0.0.0.0", port=8000, reload=True
    )
