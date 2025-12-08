# app/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ModelType(str, Enum):
    """Типы моделей"""
    LDA = "lda"
    NMF = "nmf"
    BERTOPIC = "bertopic"

class TextRequest(BaseModel):
    """Запрос для предсказания"""
    text: str = Field(..., min_length=10, max_length=10000, description="Текст для анализа")
    model_type: ModelType = Field(ModelType.LDA, description="Тип модели")
    return_probabilities: bool = Field(False, description="Возвращать вероятности")
    return_top_words: bool = Field(True, description="Возвращать топ-слова")

class BatchRequest(BaseModel):
    """Пакетный запрос"""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="Список текстов")
    model_type: ModelType = Field(ModelType.LDA, description="Тип модели")
    parallel_processing: bool = Field(True, description="Параллельная обработка")

class TopicInfo(BaseModel):
    """Информация о теме"""
    topic_id: int
    topic_name: str
    probability: Optional[float] = None
    top_words: Optional[List[str]] = None

class PredictionResponse(BaseModel):
    """Ответ с предсказанием"""
    request_id: str
    text: str
    model_type: ModelType
    timestamp: datetime
    processing_time_ms: float
    main_topic: TopicInfo
    alternative_topics: Optional[List[TopicInfo]] = None

class ModelInfo(BaseModel):
    """Информация о модели"""
    type: str
    loaded: bool
    topics_count: Optional[int] = None
    topics: Optional[List[str]] = None
    description: Optional[str] = None