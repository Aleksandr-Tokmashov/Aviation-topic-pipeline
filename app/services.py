# app/services.py
import pickle
import numpy as np
from typing import Dict, List, Optional, Any
import os
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import pymorphy3
import nltk
from nltk.corpus import stopwords
from datetime import datetime

from app.config import settings
from app.schemas import ModelType

# Загрузка стоп-слов (делаем один раз)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    """Предобработка текста"""
    
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        self.stopwords = set(stopwords.words('russian'))
        self.extra_stopwords = {'наш', 'свой', 'который', 'это', 'весь', 'сам'}
    
    def preprocess(self, text: str) -> str:
        """Основная функция предобработки"""
        if not isinstance(text, str):
            return ""
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление спецсимволов
        text = re.sub(r'[^а-яё\s]', ' ', text)
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Лемматизация
        words = text.split()
        lemmas = []
        
        for word in words:
            if len(word) > 2 and word not in self.stopwords:
                if not word.isdigit():
                    parsed = self.morph.parse(word)[0]
                    lemma = parsed.normal_form
                    if lemma not in self.extra_stopwords:
                        lemmas.append(lemma)
        
        return " ".join(lemmas)

class ModelService:
    """Сервис для работы с моделями"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.preprocessor = TextPreprocessor()
        self.cache: Dict[str, Any] = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def load_models(self) -> bool:
        """Загрузка моделей из файлов"""
        model_files = {
            "lda": os.path.join(settings.MODEL_DIR, "lda_model.pkl"),
            "nmf": os.path.join(settings.MODEL_DIR, "nmf_model.pkl"),
            "bertopic": os.path.join(settings.MODEL_DIR, "bertopic_model.pkl")
        }
        
        success = True
        for name, path in model_files.items():
            try:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.models[name] = pickle.load(f)
                    print(f"✅ Модель {name} загружена")
                else:
                    print(f"⚠️ Файл модели {name} не найден: {path}")
                    success = False
            except Exception as e:
                print(f"❌ Ошибка загрузки модели {name}: {e}")
                success = False
        
        return success
    
    def get_model(self, model_type: ModelType) -> Any:
        """Получение модели по типу"""
        model = self.models.get(model_type.value)
        if not model:
            raise ValueError(f"Модель {model_type.value} не загружена")
        return model
    
    def predict(
        self,
        text: str,
        model_type: ModelType,
        return_probabilities: bool = False,
        return_top_words: bool = True
    ) -> Dict[str, Any]:
        """Предсказание темы для текста"""
        start_time = time.time()
        
        # Проверка кэша
        cache_key = f"{model_type.value}_{hashlib.md5(text.encode()).hexdigest()}"
        if settings.CACHE_ENABLED and cache_key in self.cache:
            result = self.cache[cache_key]
            result["cached"] = True
            return result
        
        # Получение модели
        model_data = self.get_model(model_type)
        
        if model_type in [ModelType.LDA, ModelType.NMF]:
            # Предобработка
            processed_text = self.preprocessor.preprocess(text)
            
            # Векторизация
            X = model_data['vectorizer'].transform([processed_text])
            
            # Предсказание
            if model_type == ModelType.LDA:
                topic_dist = model_data['model'].transform(X)[0]
            else:  # NMF
                topic_dist = model_data['model'].transform(X)[0]
            
            topic_id = int(np.argmax(topic_dist))
            probability = float(topic_dist[topic_id])
            
            # Название темы
            topics = model_data.get('topics', [])
            topic_name = topics[topic_id] if topic_id < len(topics) else f"Тема {topic_id+1}"
            
            # Топ-слова
            top_words = []
            if return_top_words:
                if model_type == ModelType.LDA:
                    top_words = self._get_lda_top_words(model_data, topic_id)
                else:
                    top_words = self._get_nmf_top_words(model_data, topic_id)
        
        else:  # BERTopic
            topic_model = model_data['model']
            topics, probs = topic_model.transform([text])
            topic_id = int(topics[0])
            

            topic_name = model_data.get('topics', [])[topic_id+1] if topic_id < len(model_data.get('topics', [])) else f"Тема {topic_id}"
            probability = float(probs[0][topic_id]) if probs is not None else 0.0
            
            # Топ-слова
            top_words = []
            if return_top_words and topic_id != -1 and topic_id in topic_model.get_topics():
                topic_words = topic_model.get_topic(topic_id)
                top_words = [word for word, _ in topic_words[:10]]
        
        # Формирование результата
        result = {
            "request_id": hashlib.md5(f"{text}{time.time()}".encode()).hexdigest()[:12],
            "text": text[:100] + "..." if len(text) > 100 else text,
            "model_type": model_type,
            "timestamp": datetime.fromtimestamp(time.time()),
            "processing_time_ms": (time.time() - start_time) * 1000,
            "main_topic": {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "probability": probability,
                "top_words": top_words[:5] if top_words else None
            },
            "cached": False
        }
        
        # Альтернативные темы
        if return_probabilities and model_type in [ModelType.LDA, ModelType.NMF]:
            alt_topics = []
            sorted_indices = np.argsort(topic_dist)[::-1][1:4]
            for idx in sorted_indices:
                if topic_dist[idx] > 0.1:
                    alt_name = topics[idx] if idx < len(topics) else f"Тема {idx+1}"
                    alt_topics.append({
                        "topic_id": int(idx),
                        "topic_name": alt_name,
                        "probability": float(topic_dist[idx])
                    })
            if alt_topics:
                result["alternative_topics"] = alt_topics
        
        # Сохранение в кэш
        if settings.CACHE_ENABLED:
            self.cache[cache_key] = result
            if len(self.cache) > settings.CACHE_MAX_SIZE:
                self.cache.pop(next(iter(self.cache)))
        
        return result
    
    def _get_lda_top_words(self, model_data: Dict, topic_id: int, n_words: int = 10) -> List[str]:
        """Топ-слова для LDA"""
        vectorizer = model_data['vectorizer']
        model = model_data['model']
        feature_names = vectorizer.get_feature_names_out()
        topic = model.components_[topic_id]
        top_indices = topic.argsort()[:-n_words-1:-1]
        return [feature_names[i] for i in top_indices]
    
    def _get_nmf_top_words(self, model_data: Dict, topic_id: int, n_words: int = 10) -> List[str]:
        """Топ-слова для NMF"""
        vectorizer = model_data['vectorizer']
        model = model_data['model']
        feature_names = vectorizer.get_feature_names_out()
        topic = model.components_[topic_id]
        top_indices = topic.argsort()[:-n_words-1:-1]
        return [feature_names[i] for i in top_indices]
    
    def predict_batch(
        self,
        texts: List[str],
        model_type: ModelType,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """Пакетное предсказание"""
        start_time = time.time()
        
        if parallel:
            # Параллельная обработка
            futures = []
            for text in texts:
                future = self.executor.submit(
                    self.predict, text, model_type, False, True
                )
                futures.append(future)
            
            predictions = []
            for future in as_completed(futures):
                try:
                    predictions.append(future.result())
                except Exception as e:
                    predictions.append({
                        "error": str(e),
                        "main_topic": {"topic_id": -1, "topic_name": "Ошибка"}
                    })
        else:
            # Последовательная обработка
            predictions = []
            for text in texts:
                try:
                    predictions.append(self.predict(text, model_type, False, True))
                except Exception as e:
                    predictions.append({
                        "error": str(e),
                        "main_topic": {"topic_id": -1, "topic_name": "Ошибка"}
                    })
        
        total_time = (time.time() - start_time) * 1000
        
        # Статистика
        successful = sum(1 for p in predictions if p.get("main_topic", {}).get("topic_id", -1) != -1)
        
        return {
            "request_id": hashlib.md5(str(time.time()).encode()).hexdigest()[:12],
            "model_type": model_type,
            "total_texts": len(texts),
            "successful_predictions": successful,
            "failed_predictions": len(texts) - successful,
            "total_processing_time_ms": total_time,
            "average_processing_time_ms": total_time / len(texts) if texts else 0,
            "predictions": predictions
        }
    
    def get_all_models_info(self) -> Dict[str, Any]:
        """Информация о всех моделях"""
        info = {}
        
        for model_type in ModelType:
            try:
                model_data = self.get_model(model_type)
                info[model_type.value] = {
                    "type": model_type.value,
                    "loaded": True,
                    "topics_count": model_data.get('n_topics', 0),
                    "topics": model_data.get('topics', []),
                    "description": {
                        ModelType.LDA: "Latent Dirichlet Allocation",
                        ModelType.NMF: "Non-negative Matrix Factorization",
                        ModelType.BERTOPIC: "BERTopic с трансформерами"
                    }[model_type]
                }
            except:
                info[model_type.value] = {
                    "type": model_type.value,
                    "loaded": False
                }
        
        return info
    


# Singleton instance
_model_service_instance = None

def get_model_service() -> ModelService:
    """Получение экземпляра сервиса"""
    global _model_service_instance
    if _model_service_instance is None:
        _model_service_instance = ModelService()
    return _model_service_instance