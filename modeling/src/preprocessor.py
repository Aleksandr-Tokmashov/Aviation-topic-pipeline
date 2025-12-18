import re
import pymorphy3
from nltk.corpus import stopwords
from omegaconf import DictConfig

def get_preprocessor(config: DictConfig):
    morph = pymorphy3.MorphAnalyzer()
    russian_stopwords = set(stopwords.words("russian"))
    extra_stopwords = set(config.preprocessing.extra_stopwords)
    
    def preprocess_text(text: str) -> str:
        if not isinstance(text, str):
            return ""
    
        text = text.lower()
        
    
        text = re.sub(r'[^а-яё\s]', ' ', text)
        
        words = text.split()
        
        lemmas = []
        for word in words:
            if (word not in russian_stopwords and 
                len(word) > config.preprocessing.min_word_length):
                try:
                    lemma = morph.parse(word)[0].normal_form
                    if lemma not in extra_stopwords:
                        lemmas.append(lemma)
                except:
                    continue
        
        return " ".join(lemmas)
    
    return preprocess_text