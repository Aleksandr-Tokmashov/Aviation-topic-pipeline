# app/main.py
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from typing import List
from datetime import datetime

from app.config import settings
from app.schemas import TextRequest, PredictionResponse, BatchRequest
from app.services import ModelService, get_model_service

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Создание приложения
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="API для тематического моделирования авиационных текстов",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальный сервис моделей
model_service = get_model_service()

# Загрузка моделей при старте
@app.on_event("startup")
async def startup_event():
    """Загрузка моделей при старте приложения"""
    logger.info("Загрузка моделей...")
    success = model_service.load_models()
    if success:
        logger.info("Модели успешно загружены")
    else:
        logger.warning("Некоторые модели не загружены")

# Эндпоинты
@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health():
    """Проверка состояния сервиса"""
    models_status = model_service.get_all_models_info()
    
    # Проверяем, что хотя бы одна модель загружена
    models_loaded = any(info["loaded"] for info in models_status.values())
    
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models": models_status,
        "environment": settings.ENVIRONMENT,
    }

@app.get("/models")
async def get_models():
    """Получение информации о моделях"""
    return model_service.get_all_models_info()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    """
    Предсказание темы для одного текста
    
    - text: текст для анализа
    - model_type: тип модели (lda, nmf, bertopic)
    """
    try:
        result = model_service.predict(
            text=request.text,
            model_type=request.model_type,
            return_probabilities=request.return_probabilities,
            return_top_words=request.return_top_words
        )
        return PredictionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка сервера")

@app.post("/predict/file")
async def predict_file(
    file: UploadFile = File(...),
    model_type: str = "lda"
):
    """
    Предсказание тем из файла
    
    Поддерживаемые форматы: CSV, TXT
    """
    if not file.filename.endswith(('.csv', '.txt')):
        raise HTTPException(status_code=400, detail="Поддерживаются только CSV и TXT файлы")
    
    try:
        content = await file.read()
        
        if file.filename.endswith('.csv'):
            import pandas as pd
            import io
            df = pd.read_csv(io.BytesIO(content))
            
            # Ищем столбец с текстом
            text_columns = ['text', 'message', 'content', 'post']
            text_column = None
            for col in df.columns:
                if col.lower() in text_columns:
                    text_column = col
                    break
            
            if not text_column:
                raise HTTPException(status_code=400, detail="CSV должен содержать столбец 'text' или 'message'")
            
            texts = df[text_column].astype(str).tolist()
        else:
            # TXT файл
            texts = content.decode('utf-8').splitlines()
        
        # Ограничиваем количество
        texts = texts[:settings.MAX_BATCH_SIZE]
        
        # Предсказание
        from app.schemas import ModelType
        result = model_service.predict_batch(
            texts=texts,
            model_type=ModelType(model_type),
        )
        
        return {
            "filename": file.filename,
            "model_type": model_type,
            **result
        }
        
    except Exception as e:
        logger.error(f"Ошибка обработки файла: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {str(e)}")

# Для запуска напрямую
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD
    )