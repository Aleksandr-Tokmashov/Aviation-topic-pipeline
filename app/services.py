import pickle
import numpy as np
from typing import Dict, List, Any
import os
import time
import hashlib
import re
import pymorphy3
import nltk
from nltk.corpus import stopwords

from app.config import settings
from app.schemas import ModelType

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
        
        text = text.lower()
        text = re.sub(r'[^а-яё\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
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
    
    def load_models(self):
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
                    print(f"Модель {name} загружена")
                else:
                    print(f"Файл модели {name} не найден: {path}")
                    success = False
            except Exception as e:
                print(f"Ошибка загрузки модели {name}: {e}")
                success = False
        
        return success
    
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
    
    def get_model(self, model_type: ModelType):
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
        
        model_data = self.get_model(model_type)
        
        if model_type in [ModelType.LDA, ModelType.NMF]:
            processed_text = self.preprocessor.preprocess(text)
            X = model_data['vectorizer'].transform([processed_text])
            
            # Предсказание
            if model_type == ModelType.LDA:
                topic_dist = model_data['model'].transform(X)[0]
            else:  # NMF
                topic_dist = model_data['model'].transform(X)[0]
            
            topic_id = int(np.argmax(topic_dist))
            probability = float(topic_dist[topic_id])
            
            topics = model_data.get('topics', [])
            topic_name = topics[topic_id] if topic_id < len(topics) else f"Тема {topic_id+1}"
            
            top_words = []
            if return_top_words:
                if model_type == ModelType.LDA:
                    top_words = self._get_nmf_lda_top_words(model_data, topic_id)
                else:
                    top_words = self._get_nmf_lda_top_words(model_data, topic_id)
        
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
    
        result = {
            "text": text[:100] + "..." if len(text) > 100 else text,
            "model_type": model_type,
            "main_topic": {
                "topic_id": topic_id,
                "topic_name": topic_name,
                "probability": probability,
                "top_words": top_words[:5] if top_words else None
            }
        }
        
        return result
    
    def _get_nmf_lda_top_words(self, model_data: Dict, topic_id: int, n_words: int = 10) -> List[str]:
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
    ) -> Dict[str, Any]:
        """Пакетное предсказание"""
 
        predictions = []
        for text in texts:
            try:
                predictions.append(self.predict(text, model_type, False, True))
            except Exception as e:
                predictions.append({
                    "error": str(e),
                    "main_topic": {"topic_id": -1, "topic_name": "Ошибка"}
                })
        
        return {
            "model_type": model_type,
            "total_texts": len(texts),
            "predictions": predictions
        }

_model_service_instance = None

def get_model_service() -> ModelService:
    """Получение экземпляра сервиса"""
    global _model_service_instance
    if _model_service_instance is None:
        _model_service_instance = ModelService()
    return _model_service_instance