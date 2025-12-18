import matplotlib.pyplot as plt
from wordcloud import WordCloud
from typing import List
import seaborn as sns
import numpy as np


def create_wordcloud(texts: List[str], max_words: int = 100):
    all_text = ' '.join(texts)
    
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=max_words,
        colormap='viridis',
        collocations=False
    ).generate(all_text)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Облако слов (обработанные тексты)', fontsize=16)
    
    return wordcloud


def visualize_topic_words(model, vectorizer, n_words=10):
    feature_names = vectorizer.get_feature_names_out()
    n_topics = model.n_components
    
    n_cols = min(3, n_topics)
    n_rows = (n_topics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*4))
    
    if n_topics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for topic_idx, topic in enumerate(model.components_):
        if topic_idx >= len(axes):
            break
            
        ax = axes[topic_idx]
        top_indices = topic.argsort()[:-n_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        weights = topic[top_indices]
        
        y_pos = np.arange(len(top_words))
        ax.barh(y_pos, weights, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_words)
        ax.invert_yaxis()
        ax.set_xlabel('Weight')
        ax.set_title(f'Topic {topic_idx}')
    
    for idx in range(n_topics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    return fig