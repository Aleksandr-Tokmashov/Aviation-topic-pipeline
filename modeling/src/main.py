import sys
from pathlib import Path
from typing import List
import warnings
warnings.filterwarnings('ignore')

src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

import pandas as pd
import mlflow
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from utils import load_config, filter_data, setup_directories
from data_loader import load_data
from preprocessor import get_preprocessor
from models import LDAModel, NMFModel, BERTopicModel
from visualization import create_wordcloud, visualize_topic_words


def run_lda_experiment(texts: List[str], config, stop_words, experiment_config):
    print("ЭКСПЕРИМЕНТ LDA")
    
    lda_config = load_config("modeling/config/experiment/lda.yaml")
    
    lda_model = LDAModel(lda_config)
    lda_model.train(texts, stop_words=stop_words)
    
    top_words = lda_model.get_top_words(n_words=10)
    
    print("\nТоп-слова LDA:")
    for topic_id, words in top_words.items():
        print(f"  Тема {topic_id}: {', '.join(words[:5])}...")
    
    with mlflow.start_run(run_name="lda_experiment"):
        mlflow.log_param("model_type", "LDA")
        mlflow.log_param("n_topics", lda_config.best_n_topics)
        mlflow.log_param("max_features", lda_config.max_features)
        mlflow.log_param("random_state", lda_config.random_state)
        
        model_path = Path(config.paths.models) / "lda_model.pkl"
        lda_model.save_model(str(model_path))
        
        mlflow.log_artifact(str(model_path))
        
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            for topic_id, words in top_words.items():
                f.write(f"Тема {topic_id}: {', '.join(words)}\n")
            mlflow.log_artifact(f.name, "topics")
    
    return lda_model


def run_nmf_experiment(texts: List[str], config, stop_words, experiment_config):
    print("ЭКСПЕРИМЕНТ NMF")

    nmf_config = load_config("modeling/config/experiment/nmf.yaml")
    
    nmf_model = NMFModel(nmf_config)
    nmf_model.train(texts, stop_words=stop_words)
    
    top_words = nmf_model.get_top_words(n_words=10)
    
    print("\nТоп-слова NMF:")
    for topic_id, words in top_words.items():
        print(f"  Тема {topic_id}: {', '.join(words[:5])}...")
    
    
    with mlflow.start_run(run_name="nmf_experiment"):
        mlflow.log_param("model_type", "NMF")
        mlflow.log_param("n_topics", nmf_config.best_n_topics)
        mlflow.log_param("max_features", nmf_config.max_features)
        mlflow.log_param("random_state", nmf_config.random_state)
        
        model_path = Path(config.paths.models) / "nmf_model.pkl"
        nmf_model.save_model(str(model_path))
        
        mlflow.log_artifact(str(model_path))
    
    return nmf_model


def run_bertopic_experiment(texts: List[str], config, experiment_config):
    print("ЭКСПЕРИМЕНТ BERTopic")

    bertopic_config = load_config("modeling/config/experiment/bertopic.yaml")
    
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer

    with mlflow.start_run(run_name="bertopic_experiment"):
        mlflow.log_params({
            "model_type": "BERTopic",
            "n_topics": bertopic_config.best_n_topics,
            "min_topic_size": bertopic_config.best_min_topic_size,
            "embedding_model": bertopic_config.embedding_model,
            "random_state": bertopic_config.random_state
        })
        
        print("Обучение BERTopic...")
        
        try:
            embedding_model = SentenceTransformer(bertopic_config.embedding_model)
            
            topic_model = BERTopic(
                embedding_model=embedding_model,
                min_topic_size=bertopic_config.best_min_topic_size,
                nr_topics=bertopic_config.best_n_topics,
                verbose=True
            )
            
            topics, probabilities = topic_model.fit_transform(texts)
            
            topic_info = topic_model.get_topic_info()

            n_topics_found = len(topic_info[topic_info["Topic"] != -1])
            outliers_count = topic_info[topic_info["Topic"] == -1]["Count"].values[0] if -1 in topic_info["Topic"].values else 0
            
            mlflow.log_metrics({
                "n_topics_found": n_topics_found,
                "outliers_count": outliers_count,
                "outliers_percent": outliers_count / len(texts) * 100
            })
            
            print(f"Найдено тем: {n_topics_found}")
            print(f"Выбросов: {outliers_count} ({outliers_count/len(texts)*100:.1f}%)")

            import tempfile
            import pickle
            
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
                pickle.dump(topic_model, open(tmp.name, 'wb'))
                mlflow.log_artifact(tmp.name, "bertopic_model")
        
            return topic_model
            
        except Exception as e:
            print(f"Ошибка при обучении BERTopic: {e}")
            mlflow.log_param("error", str(e))
            return None


def main():
    config = load_config()
    setup_directories(config)
    
    df = load_data(config)
    print(f"   Загружено строк: {len(df)}")

    filtered_df = filter_data(df, config)

    preprocessor = get_preprocessor(config)
    
    tqdm.pandas(desc="Обработка текстов")
    filtered_df['processed_text'] = filtered_df['message'].progress_apply(preprocessor)

    filtered_df = filtered_df[filtered_df['processed_text'].str.len() > 0]
    texts = filtered_df['processed_text'].tolist()
    print(f"   Обработано текстов: {len(texts)}")

    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    print(f"   Эксперимент: {config.mlflow.experiment_name}")

    try:
        from nltk.corpus import stopwords
        russian_stopwords = stopwords.words("russian")
    except:
        russian_stopwords = []
        print("   Русские стоп-слова не найдены")
    
    experiment_config = load_config("modeling/config/experiment/experiment.yaml") if Path("modeling/config/experiment/experiment.yaml").exists() else None
    
    lda_model = run_lda_experiment(texts, config, russian_stopwords, experiment_config)

    nmf_model = run_nmf_experiment(texts, config, russian_stopwords, experiment_config)
    
    bertopic_model = run_bertopic_experiment(texts, config, experiment_config)
    
    output_path = Path(config.paths.processed_data) / "processed_posts.csv"
    filtered_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"   Данные сохранены: {output_path}")
    
    print("ПАЙПЛАЙН УСПЕШНО ЗАВЕРШЕН!")
    print(f"Обработано сообщений: {len(filtered_df)}")
    print(f"Сохраненные модели: models/")
    print(f"MLflow эксперимент: {config.mlflow.experiment_name}")


if __name__ == "__main__":
    main()