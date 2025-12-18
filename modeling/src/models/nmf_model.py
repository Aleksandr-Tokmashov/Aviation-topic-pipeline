"""
NMF (Non-negative Matrix Factorization) модель
Простая реализация
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from typing import Tuple, List, Dict
from omegaconf import DictConfig


class NMFModel:    
    def __init__(self, config: DictConfig):
        self.config = config
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        
    def train(self, texts: List[str], stop_words: List[str] = None) -> Tuple[NMF, TfidfVectorizer]:
        print(f"Обучение NMF с {self.config.best_n_topics} темами...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            min_df=2,
            max_df=0.8,
            stop_words=stop_words,
            ngram_range=(1, 2) 
        )
        
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"  Размер матрицы: {X.shape}")
        print(f"  Размер словаря: {len(self.feature_names)}")
        
        self.model = NMF(
            n_components=self.config.best_n_topics,
            random_state=self.config.random_state,
            max_iter=200,
            init='nndsvda'
        ) 
        
        self.model.fit(X)
        
        return self.model, self.vectorizer
    
    def get_top_words(self, n_words: int = 10) -> Dict[int, List[str]]:
        if self.model is None or self.feature_names is None:
            raise ValueError("Модель еще не обучена!")
        
        topics_words = {}
        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[:-n_words - 1:-1]
            top_words = [self.feature_names[i] for i in top_indices]
            topics_words[topic_idx] = top_words
        
        return topics_words
    
    def get_topic_weights(self) -> List[float]:
        if self.model is None:
            raise ValueError("Модель еще не обучена!")
        
        # Сумма по строкам = вклад каждой темы
        topic_weights = self.model.components_.sum(axis=1)
        topic_weights_normalized = topic_weights / topic_weights.sum()
        
        return topic_weights_normalized.tolist()
    
    def predict_topics(self, texts: List[str]) -> List[int]:
        if self.model is None or self.vectorizer is None:
            raise ValueError("Модель еще не обучена!")
        
        X = self.vectorizer.transform(texts)
        topic_distributions = self.model.transform(X)
        predicted_topics = topic_distributions.argmax(axis=1)
        
        return predicted_topics.tolist()
    
    def save_model(self, filepath: str):
        import pickle
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Модель сохранена: {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str):
        import pickle
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        nmf_instance = cls(model_data['config'])
        nmf_instance.model = model_data['model']
        nmf_instance.vectorizer = model_data['vectorizer']
        nmf_instance.feature_names = model_data['feature_names']
        
        return nmf_instance


def train_nmf_simple(texts: List[str], config: DictConfig, stop_words: List[str] = None) -> Tuple[NMF, TfidfVectorizer]:
    vectorizer = TfidfVectorizer(
        max_features=config.max_features,
        min_df=2,
        max_df=0.8,
        stop_words=stop_words
    )
    
    X = vectorizer.fit_transform(texts)

    nmf = NMF(
        n_components=config.best_n_topics,
        random_state=config.random_state,
        max_iter=200,
        init='nndsvda'
    )
    
    nmf.fit(X)
    
    return nmf, vectorizer