"""News classification module."""

import time
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings('ignore')


class NewsClassifier:
    """Modern news classifier with multiple methods."""
    
    def __init__(self, method: str = 'bge_large'):
        """
        Initialize classifier.
        
        Args:
            method: Classification method ('bge_large', 'e5_large', 'zero_shot', etc.)
        """
        self.method = method
        self.categories = [
            'politics', 'business', 'technology', 'sports', 
            'entertainment', 'health', 'science', 'world', 'crime'
        ]
        
        self.category_descriptions = {
            'politics': 'Government, elections, politicians, policy, congress, parliament',
            'business': 'Companies, markets, finance, stocks, trade, economy',
            'technology': 'Computers, software, AI, startups, tech companies',
            'sports': 'Games, teams, athletes, championships, football, basketball',
            'entertainment': 'Movies, music, celebrities, TV shows, films, arts',
            'health': 'Medical news, diseases, treatments, hospitals, vaccines',
            'science': 'Research, discoveries, climate, environment, space',
            'world': 'International news, countries, conflicts, global events',
            'crime': 'Police, arrests, courts, investigations, criminal activity'
        }
        
        self.models = {
            'bge_large': 'BAAI/bge-large-en-v1.5',
            'bge_base': 'BAAI/bge-base-en-v1.5', 
            'e5_large': 'intfloat/e5-large-v2',
            'e5_base': 'intfloat/e5-base-v2',
            'mpnet': 'sentence-transformers/all-mpnet-base-v2',
            'minilm': 'all-MiniLM-L6-v2'
        }
        
        self.model = None
        self.classifier = None
        
    def _load_model(self):
        """Load the specified model."""
        if self.model is not None:
            return
            
        print(f"🔄 Loading {self.method}...")
        
        if self.method == 'zero_shot':
            device = 0 if torch.cuda.is_available() else -1
            self.classifier = pipeline(
                "zero-shot-classification", 
                model="facebook/bart-large-mnli",
                device=device
            )
        elif self.method in self.models:
            model_name = self.models[self.method]
            self.model = SentenceTransformer(model_name)
        else:
            # Default fallback
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"✅ Model loaded")
    
    def classify_single(self, text: str) -> Dict[str, Any]:
        """
        Classify a single text.
        
        Args:
            text: Text to classify
            
        Returns:
            Classification result
        """
        self._load_model()
        start_time = time.time()
        
        if self.method == 'zero_shot':
            result = self.classifier(text, self.categories)
            return {
                'category': result['labels'][0],
                'confidence': result['scores'][0],
                'method': self.method,
                'time': time.time() - start_time
            }
        else:
            # Semantic similarity approach
            text_emb = self.model.encode([text])
            
            # Handle E5 models with special prefixes
            if 'e5' in self.method:
                text_emb = self.model.encode([f"query: {text}"])
                cat_embs = self.model.encode([f"passage: {desc}" for desc in self.category_descriptions.values()])
            else:
                cat_embs = self.model.encode(list(self.category_descriptions.values()))
            
            similarities = cosine_similarity(text_emb, cat_embs)[0]
            best_idx = np.argmax(similarities)
            
            return {
                'category': self.categories[best_idx],
                'confidence': float(similarities[best_idx]),
                'method': self.method,
                'time': time.time() - start_time
            }
    
    def classify_batch(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify multiple news articles.
        
        Args:
            news_list: List of news dictionaries with 'title' and 'description'
            
        Returns:
            List of news with classification results
        """
        print(f"🗂️ Classifying {len(news_list)} articles with {self.method}...")
        
        results = []
        for i, news in enumerate(news_list):
            if i % 20 == 0:
                print(f"  Processing {i}/{len(news_list)}...")
            
            text = f"{news['title']}. {news.get('description', '')}"
            classification = self.classify_single(text)
            
            result = news.copy()
            result.update(classification)
            results.append(result)
        
        print(f"✅ Classification complete")
        return results
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available classification methods."""
        return ['bge_large', 'bge_base', 'e5_large', 'e5_base', 'mpnet', 'minilm', 'zero_shot'] 