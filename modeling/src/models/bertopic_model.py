import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from omegaconf import DictConfig
import warnings
warnings.filterwarnings('ignore')


class BERTopicModel:
    def __init__(self, config: DictConfig):
        self.config = config
        self.model = None
        self.topics = None
        self.probabilities = None
        
    def train(self, texts: List[str]) -> Any:
        print(f"Обучение BERTopic с {self.config.best_n_topics} темами...")
        print(f"  Min topic size: {self.config.best_min_topic_size}")
        print(f"  Embedding model: {self.config.embedding_model}")
        
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            from umap import UMAP
            from hdbscan import HDBSCAN
            from sklearn.feature_extraction.text import CountVectorizer
            
            try:
                from nltk.corpus import stopwords
                russian_stopwords = stopwords.words("russian")
            except:
                russian_stopwords = []
            
            embedding_model = SentenceTransformer(self.config.embedding_model)
            
            umap_model = UMAP(
                n_neighbors=10,
                n_components=3,
                min_dist=0.05,
                metric='cosine',
                random_state=self.config.random_state
            )
            
            hdbscan_model = HDBSCAN(
                min_cluster_size=self.config.best_min_topic_size,
                min_samples=5,
                cluster_selection_method='leaf',
                metric='euclidean',
                prediction_data=True
            )
            
            vectorizer_model = CountVectorizer(
                stop_words=russian_stopwords,
                ngram_range=(1, 1),
                min_df=2,
                max_df=0.8
            )
            
            self.model = BERTopic(
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                nr_topics=self.config.best_n_topics,
                calculate_probabilities=True,
                verbose=True
            )
            
            self.topics, self.probabilities = self.model.fit_transform(texts)
            
            topic_info = self.model.get_topic_info()
            n_topics_found = len(topic_info[topic_info["Topic"] != -1])
            outliers = topic_info[topic_info["Topic"] == -1]["Count"].values[0] if -1 in topic_info["Topic"].values else 0
            
            print(f"  Найдено тем: {n_topics_found}")
            print(f"  Выбросов: {outliers} ({outliers/len(texts)*100:.1f}%)")
            
            if n_topics_found > 0:
                print("\n  Топ-3 темы:")
                sorted_topics = topic_info[topic_info.Topic != -1].sort_values('Count', ascending=False)
                
                for i, (_, row) in enumerate(sorted_topics.head(3).iterrows(), 1):
                    topic_id = row['Topic']
                    count = row['Count']
                    words = self.model.get_topic(topic_id)
                    top_words = [word for word, _ in words[:3]]
                    print(f"    {i}. Тема #{topic_id} ({count} док.): {', '.join(top_words)}")
            
            return self.model
            
        except ImportError as e:
            print(f"  Ошибка: {e}")
            print("  Установите bertopic: pip install bertopic sentence-transformers umap-learn hdbscan")
            return None
    
    def get_topic_info(self) -> Optional[pd.DataFrame]:
        if self.model is None:
            return None
        
        import pandas as pd
        topic_info = self.model.get_topic_info()
        return topic_info
    
    def get_top_words(self, topic_id: int, n_words: int = 10) -> List[Tuple[str, float]]:
        if self.model is None:
            return []
        
        return self.model.get_topic(topic_id)[:n_words]
    
    def visualize_topics(self):
        if self.model is None:
            print("Модель не обучена!")
            return
        
        try:
            fig = self.model.visualize_topics()
            fig.show()
            return fig
        except Exception as e:
            print(f"Ошибка визуализации: {e}")
            return None
    
    def visualize_barchart(self, n_topics: int = 10):
        if self.model is None:
            print("Модель не обучена!")
            return
        
        try:
            topic_info = self.model.get_topic_info()
            valid_topics = len(topic_info[topic_info["Topic"] != -1])
            n_topics = min(n_topics, valid_topics)
            
            fig = self.model.visualize_barchart(top_n_topics=n_topics)
            fig.show()
            return fig
        except Exception as e:
            print(f"Ошибка визуализации: {e}")
            return None
    
    def predict_topics(self, texts: List[str]) -> List[int]:
        if self.model is None:
            raise ValueError("Модель еще не обучена!")
        
        topics, _ = self.model.transform(texts)
        return topics.tolist()
    
    def save_model(self, filepath: str):
        if self.model is None:
            print("Нет модели для сохранения!")
            return
        
        import pickle
        model_data = {
            'model': self.model,
            'topics': self.topics,
            'probabilities': self.probabilities,
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
        
        bertopic_instance = cls(model_data['config'])
        bertopic_instance.model = model_data['model']
        bertopic_instance.topics = model_data['topics']
        bertopic_instance.probabilities = model_data['probabilities']
        
        return bertopic_instance


def train_bertopic_simple(texts: List[str], config: DictConfig) -> Any:
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        
        print(f"Обучение BERTopic с {config.best_n_topics} темами...")

        embedding_model = SentenceTransformer(config.embedding_model)
        
        topic_model = BERTopic(
            embedding_model=embedding_model,
            nr_topics=config.best_n_topics,
            min_topic_size=config.best_min_topic_size,
            verbose=True
        )
        
        topics, probabilities = topic_model.fit_transform(texts)
        
        topic_info = topic_model.get_topic_info()
        n_topics_found = len(topic_info[topic_info["Topic"] != -1])
        print(f"  Найдено тем: {n_topics_found}")
        
        return topic_model, topics, probabilities
        
    except ImportError:
        print("BERTopic не установлен. Установите: pip install bertopic sentence-transformers")
        return None, None, None