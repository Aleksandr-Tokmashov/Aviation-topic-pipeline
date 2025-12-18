import pandas as pd
from pathlib import Path
import yaml
from omegaconf import OmegaConf
import os


def load_config(config_path: str = "modeling/config/config.yaml"):
    if not Path(config_path).exists():
        print(f"Config file not found: {config_path}")
        return OmegaConf.create({})
    
    return OmegaConf.load(config_path)


def save_config(config, config_path: str):
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, config_path)


def setup_directories(config):
    directories = [
        config.paths.raw_data,
        config.paths.processed_data,
        config.paths.models,
        "temp-data",
        "temp-charts",
        "mlruns"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Созданы директории: {len(directories)}")


def filter_data(df: pd.DataFrame, config):
    if df.empty:
        return df
    
    df = df.dropna(subset=['message']).drop_duplicates(subset=['message'])
    
    df['message_length'] = df['message'].apply(lambda x: len(str(x)))
    df['word_count'] = df['message'].str.split().str.len().fillna(0)
    
    filtered = df[
        (df['word_count'] <= config.data.filters.max_word_count) &
        (df['message_length'] >= config.data.filters.min_length) &
        (df['message_length'] <= config.data.filters.max_length)
    ].copy()
    
    print(f"Фильтрация: {len(filtered)}/{len(df)} строк осталось")
    return filtered


def log_to_mlflow(params: dict, metrics: dict = None, artifacts: list = None):
    import mlflow
    
    if params:
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    if metrics:
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
    
    if artifacts:
        for artifact_path in artifacts:
            if Path(artifact_path).exists():
                mlflow.log_artifact(artifact_path)