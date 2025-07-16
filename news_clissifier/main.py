#!/usr/bin/env python3
"""
News Classification Comparison: LLM vs BERT
Simple experiment to compare different approaches for news classification.
"""

import feedparser
import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import time
import os
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from openai import OpenAI
from langdetect import detect
import re
import warnings
warnings.filterwarnings('ignore')

class NewsCollector:
    """Simple news collector from RSS feeds"""
    
    def __init__(self):
        self.sources = {
            'bbc': 'http://feeds.bbci.co.uk/news/rss.xml',
            'cnn': 'http://rss.cnn.com/rss/edition.rss',
            'reuters': 'https://feeds.reuters.com/reuters/topNews',
            'techcrunch': 'https://techcrunch.com/feed/',
            'guardian': 'https://www.theguardian.com/world/rss',
            'npr': 'https://feeds.npr.org/1001/rss.xml',
        }
    
    def collect_news(self, max_total: int = 100) -> List[Dict[str, Any]]:
        """Collect news from RSS sources"""
        print("📰 Collecting news...")
        
        all_news = []
        seen_titles = set()
        
        for source, url in self.sources.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:30]:  # Max 30 per source
                    title = entry.get('title', '').strip()
                    description = entry.get('description', '').strip()
                    
                    if not title or title in seen_titles:
                        continue
                    
                    # Filter English only
                    try:
                        if detect(title) != 'en':
                            continue
                    except:
                        continue
                    
                    # Keep reasonable length
                    if len(title + description) > 800:
                        continue
                    
                    all_news.append({
                        'title': title,
                        'description': description,
                        'source': source,
                        'url': entry.get('link', '')
                    })
                    seen_titles.add(title)
                    
                    if len(all_news) >= max_total:
                        break
                
                if len(all_news) >= max_total:
                    break
                    
            except Exception as e:
                print(f"Error with {source}: {e}")
        
        print(f"✅ Collected {len(all_news)} news articles")
        return all_news[:max_total]

class FreeClassifier:
    """Free classification methods"""
    
    def __init__(self):
        self.categories = ['politics', 'business', 'technology', 'sports', 
                          'entertainment', 'health', 'science', 'world', 'crime']
        
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
        
        # Расширенные ключевые слова для каждой категории
        self.category_keywords = {
            'politics': ['government', 'election', 'president', 'congress', 'senate', 'parliament', 'minister', 'politician', 'vote', 'law', 'policy', 'democratic', 'republican', 'campaign', 'administration'],
            'business': ['company', 'business', 'market', 'stock', 'finance', 'economy', 'bank', 'trade', 'revenue', 'profit', 'investment', 'earnings', 'corporate', 'startup', 'ipo'],
            'technology': ['technology', 'tech', 'ai', 'computer', 'software', 'startup', 'apple', 'google', 'microsoft', 'innovation', 'digital', 'cybersecurity', 'blockchain', 'cryptocurrency'],
            'sports': ['sports', 'football', 'basketball', 'soccer', 'team', 'player', 'game', 'match', 'championship', 'olympics', 'baseball', 'tennis', 'golf', 'hockey', 'tournament'],
            'entertainment': ['movie', 'film', 'music', 'celebrity', 'actor', 'singer', 'show', 'tv', 'entertainment', 'arts', 'hollywood', 'concert', 'album', 'streaming', 'netflix'],
            'health': ['health', 'medical', 'doctor', 'hospital', 'treatment', 'vaccine', 'disease', 'patient', 'medicine', 'covid', 'pandemic', 'virus', 'drug', 'clinical', 'healthcare'],
            'science': ['science', 'research', 'study', 'discovery', 'scientist', 'climate', 'environment', 'space', 'nature', 'nasa', 'experiment', 'biology', 'physics', 'chemistry'],
            'world': ['international', 'global', 'country', 'nation', 'world', 'foreign', 'diplomatic', 'war', 'conflict', 'ukraine', 'china', 'russia', 'europe', 'asia', 'africa'],
            'crime': ['crime', 'police', 'arrest', 'court', 'investigation', 'murder', 'fraud', 'trial', 'criminal', 'law enforcement', 'robbery', 'theft', 'violence', 'shooting']
        }
        
        self.models = {}
        
        # Продвинутые модели для тестирования
        self.advanced_models = {
            # E5 модели (Microsoft)
            'e5_large': 'intfloat/e5-large-v2',
            'e5_base': 'intfloat/e5-base-v2',
            'e5_small': 'intfloat/e5-small-v2',
            
            # BGE модели (BAAI/Beijing Academy of AI)
            'bge_large': 'BAAI/bge-large-en-v1.5',
            'bge_base': 'BAAI/bge-base-en-v1.5',
            'bge_small': 'BAAI/bge-small-en-v1.5',
            
            # MPNet модели
            'mpnet_base': 'sentence-transformers/all-mpnet-base-v2',
            
            # Instructor модели (task-specific embeddings)
            'instructor_large': 'hkunlp/instructor-large',
            'instructor_base': 'hkunlp/instructor-base',
            
            # Multilingual модели
            'multilingual_e5': 'intfloat/multilingual-e5-large',
            'paraphrase_multilingual': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
            
            # Специализированные модели
            'sentence_t5': 'sentence-transformers/sentence-t5-base',
            'gtr_t5': 'sentence-transformers/gtr-t5-base',
            
            # Оригинальные (для сравнения)
            'minilm': 'all-MiniLM-L6-v2',
            'distilbert': 'sentence-transformers/all-distilroberta-v1',
        }
    
    def load_model(self, model_type: str):
        """Load specified model"""
        if model_type in self.models:
            return
            
        print(f"🔄 Loading {model_type}...")
        
        try:
            if model_type == 'sentence_bert':
                self.models[model_type] = SentenceTransformer('all-MiniLM-L6-v2')
            elif model_type == 'zero_shot_bart':
                device = 0 if torch.cuda.is_available() else -1
                self.models[model_type] = pipeline("zero-shot-classification", 
                                                 model="facebook/bart-large-mnli", 
                                                 device=device)
            elif model_type == 'distilbert_classifier':
                try:
                    # Try to load a pre-trained news classifier
                    self.models[model_type] = pipeline("text-classification", 
                                                     model="cardiffnlp/twitter-roberta-base-emotion",
                                                     device=0 if torch.cuda.is_available() else -1)
                except:
                    # Fallback to zero-shot
                    self.models[model_type] = self.models.get('zero_shot_bart') or pipeline("zero-shot-classification", 
                                                                                           model="facebook/bart-large-mnli")
            elif model_type in self.advanced_models:
                model_name = self.advanced_models[model_type]
                
                # Специальная обработка для Instructor моделей
                if 'instructor' in model_type:
                    from InstructorEmbedding import INSTRUCTOR
                    self.models[model_type] = INSTRUCTOR(model_name)
                else:
                    self.models[model_type] = SentenceTransformer(model_name)
            
            print(f"✅ Loaded {model_type}")
            
        except Exception as e:
            print(f"❌ Failed to load {model_type}: {e}")
            # Fallback to MiniLM
            if model_type != 'sentence_bert':
                self.models[model_type] = SentenceTransformer('all-MiniLM-L6-v2')
    
    def classify_keyword_based(self, text: str) -> Dict[str, Any]:
        """Simple keyword-based classification"""
        start_time = time.time()
        
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[category] = score
        
        best_category = max(scores, key=scores.get) if max(scores.values()) > 0 else 'world'
        confidence = scores[best_category] / len(self.category_keywords[best_category])
        
        return {
            'category': best_category,
            'confidence': min(confidence, 1.0),
            'method': 'keyword_based',
            'time': time.time() - start_time
        }
    
    def classify_semantic(self, text: str) -> Dict[str, Any]:
        """Classify using semantic similarity"""
        if 'sentence_bert' not in self.models:
            self.load_model('sentence_bert')
        
        start_time = time.time()
        model = self.models['sentence_bert']
        
        # Get embeddings
        text_emb = model.encode([text])
        cat_embs = model.encode(list(self.category_descriptions.values()))
        
        # Find best match
        similarities = cosine_similarity(text_emb, cat_embs)[0]
        best_idx = np.argmax(similarities)
        
        return {
            'category': self.categories[best_idx],
            'confidence': float(similarities[best_idx]),
            'method': 'sentence_bert',
            'time': time.time() - start_time
        }
    
    def classify_advanced_semantic(self, text: str, model_name: str) -> Dict[str, Any]:
        """Classify using advanced semantic models"""
        if model_name not in self.models:
            self.load_model(model_name)
        
        start_time = time.time()
        model = self.models[model_name]
        
        try:
            # Специальная обработка для Instructor моделей
            if 'instructor' in model_name:
                # Instructor требует инструкции
                instruction = "Represent the news article for classification: "
                text_emb = model.encode([[instruction, text]])
                cat_instructions = [[instruction, desc] for desc in self.category_descriptions.values()]
                cat_embs = model.encode(cat_instructions)
            else:
                # E5 модели требуют префикса для лучшего качества
                if 'e5' in model_name:
                    text_with_prefix = f"query: {text}"
                    cat_with_prefix = [f"passage: {desc}" for desc in self.category_descriptions.values()]
                    text_emb = model.encode([text_with_prefix])
                    cat_embs = model.encode(cat_with_prefix)
                else:
                    # Обычные модели
                    text_emb = model.encode([text])
                    cat_embs = model.encode(list(self.category_descriptions.values()))
            
            # Find best match
            similarities = cosine_similarity(text_emb, cat_embs)[0]
            best_idx = np.argmax(similarities)
            
            return {
                'category': self.categories[best_idx],
                'confidence': float(similarities[best_idx]),
                'method': f'semantic_{model_name}',
                'time': time.time() - start_time
            }
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")
            return {
                'category': 'world',
                'confidence': 0.0,
                'method': f'semantic_{model_name}_error',
                'time': time.time() - start_time
            }
    
    def classify_zero_shot(self, text: str) -> Dict[str, Any]:
        """Classify using zero-shot BART"""
        if 'zero_shot_bart' not in self.models:
            self.load_model('zero_shot_bart')
        
        start_time = time.time()
        classifier = self.models['zero_shot_bart']
        result = classifier(text, self.categories)
        
        return {
            'category': result['labels'][0],
            'confidence': result['scores'][0],
            'method': 'zero_shot_bart',
            'time': time.time() - start_time
        }
    
    def classify_zero_shot_advanced(self, text: str) -> Dict[str, Any]:
        """Classify using advanced zero-shot models"""
        start_time = time.time()
        
        try:
            # Попробуем разные zero-shot модели
            device = 0 if torch.cuda.is_available() else -1
            
            # DeBERTa-v3 (лучше чем BART для классификации)
            classifier = pipeline("zero-shot-classification", 
                                model="microsoft/deberta-v3-base", 
                                device=device)
            
            # Используем более детальные описания категорий
            candidate_labels = list(self.category_descriptions.values())
            result = classifier(text, candidate_labels)
            
            # Найдем соответствующую категорию
            best_desc = result['labels'][0]
            category = 'world'
            for cat, desc in self.category_descriptions.items():
                if desc == best_desc:
                    category = cat
                    break
            
            return {
                'category': category,
                'confidence': result['scores'][0],
                'method': 'zero_shot_deberta',
                'time': time.time() - start_time
            }
            
        except Exception as e:
            print(f"Error with advanced zero-shot: {e}")
            return self.classify_zero_shot(text)
    
    def classify_ensemble(self, text: str, models_to_use: List[str] = None) -> Dict[str, Any]:
        """Ensemble classification using multiple models"""
        start_time = time.time()
        
        if models_to_use is None:
            models_to_use = ['e5_base', 'bge_base', 'mpnet_base', 'minilm']
        
        predictions = []
        confidences = []
        
        for model_name in models_to_use:
            try:
                result = self.classify_advanced_semantic(text, model_name)
                predictions.append(result['category'])
                confidences.append(result['confidence'])
            except:
                continue
        
        if not predictions:
            return self.classify_semantic(text)
        
        # Взвешенное голосование
        category_scores = {}
        for pred, conf in zip(predictions, confidences):
            category_scores[pred] = category_scores.get(pred, 0) + conf
        
        best_category = max(category_scores, key=category_scores.get)
        avg_confidence = np.mean([conf for pred, conf in zip(predictions, confidences) if pred == best_category])
        
        return {
            'category': best_category,
            'confidence': float(avg_confidence),
            'method': f'ensemble_{len(predictions)}models',
            'time': time.time() - start_time
        }
    
    def classify_tfidf_nb(self, texts: List[str]) -> List[Dict[str, Any]]:
        """TF-IDF + Naive Bayes"""
        start_time = time.time()
        
        # Use category descriptions as training data
        train_texts = list(self.category_descriptions.values()) * 3  # Repeat for more data
        train_labels = self.categories * 3
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english')),
            ('classifier', MultinomialNB())
        ])
        
        pipeline.fit(train_texts, train_labels)
        predictions = pipeline.predict(texts)
        probabilities = pipeline.predict_proba(texts)
        
        results = []
        processing_time = (time.time() - start_time) / len(texts)
        
        for i, text in enumerate(texts):
            confidence = float(np.max(probabilities[i]))
            results.append({
                'category': predictions[i],
                'confidence': confidence,
                'method': 'tfidf_naive_bayes',
                'time': processing_time
            })
        
        return results
    
    def classify_tfidf_svm(self, texts: List[str]) -> List[Dict[str, Any]]:
        """TF-IDF + SVM"""
        start_time = time.time()
        
        train_texts = list(self.category_descriptions.values()) * 3
        train_labels = self.categories * 3
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english')),
            ('classifier', SVC(probability=True, kernel='linear'))
        ])
        
        pipeline.fit(train_texts, train_labels)
        predictions = pipeline.predict(texts)
        probabilities = pipeline.predict_proba(texts)
        
        results = []
        processing_time = (time.time() - start_time) / len(texts)
        
        for i, text in enumerate(texts):
            confidence = float(np.max(probabilities[i]))
            results.append({
                'category': predictions[i],
                'confidence': confidence,
                'method': 'tfidf_svm',
                'time': processing_time
            })
        
        return results
    
    def classify_tfidf_rf(self, texts: List[str]) -> List[Dict[str, Any]]:
        """TF-IDF + Random Forest"""
        start_time = time.time()
        
        train_texts = list(self.category_descriptions.values()) * 3
        train_labels = self.categories * 3
        
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=3000, stop_words='english')),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        
        pipeline.fit(train_texts, train_labels)
        predictions = pipeline.predict(texts)
        probabilities = pipeline.predict_proba(texts)
        
        results = []
        processing_time = (time.time() - start_time) / len(texts)
        
        for i, text in enumerate(texts):
            confidence = float(np.max(probabilities[i]))
            results.append({
                'category': predictions[i],
                'confidence': confidence,
                'method': 'tfidf_random_forest',
                'time': processing_time
            })
        
        return results
    
    def classify_tfidf_ensemble(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Ensemble of TF-IDF based classifiers"""
        start_time = time.time()
        
        train_texts = list(self.category_descriptions.values()) * 5
        train_labels = self.categories * 5
        
        # Подготовка данных
        tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
        X_train = tfidf.fit_transform(train_texts)
        X_test = tfidf.transform(texts)
        
        # Ensemble из разных классификаторов
        ensemble = VotingClassifier([
            ('nb', MultinomialNB()),
            ('svm', SVC(probability=True, kernel='linear')),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('lr', LogisticRegression(random_state=42, max_iter=1000))
        ], voting='soft')
        
        ensemble.fit(X_train, train_labels)
        predictions = ensemble.predict(X_test)
        probabilities = ensemble.predict_proba(X_test)
        
        results = []
        processing_time = (time.time() - start_time) / len(texts)
        
        for i, text in enumerate(texts):
            confidence = float(np.max(probabilities[i]))
            results.append({
                'category': predictions[i],
                'confidence': confidence,
                'method': 'tfidf_ensemble',
                'time': processing_time
            })
        
        return results
    
    def classify_batch(self, news_list: List[Dict], method: str = 'semantic') -> List[Dict]:
        """Classify multiple news items"""
        results = []
        
        # For batch methods (TF-IDF based)
        if method in ['tfidf_nb', 'tfidf_svm', 'tfidf_rf', 'tfidf_ensemble']:
            texts = [f"{news['title']}. {news.get('description', '')}" for news in news_list]
            
            if method == 'tfidf_nb':
                classifications = self.classify_tfidf_nb(texts)
            elif method == 'tfidf_svm':
                classifications = self.classify_tfidf_svm(texts)
            elif method == 'tfidf_rf':
                classifications = self.classify_tfidf_rf(texts)
            elif method == 'tfidf_ensemble':
                classifications = self.classify_tfidf_ensemble(texts)
            
            for i, news in enumerate(news_list):
                news_result = news.copy()
                news_result.update(classifications[i])
                results.append(news_result)
            
            return results
        
        # For individual methods
        for i, news in enumerate(news_list):
            if i % 20 == 0:
                print(f"  Processing {i}/{len(news_list)}...")
            
            text = f"{news['title']}. {news.get('description', '')}"
            
            if method == 'keyword':
                result = self.classify_keyword_based(text)
            elif method == 'semantic':
                result = self.classify_semantic(text)
            elif method == 'zero_shot':
                result = self.classify_zero_shot(text)
            elif method == 'zero_shot_advanced':
                result = self.classify_zero_shot_advanced(text)
            elif method == 'ensemble':
                result = self.classify_ensemble(text)
            elif method in self.advanced_models:
                result = self.classify_advanced_semantic(text, method)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            news_result = news.copy()
            news_result.update(result)
            results.append(news_result)
        
        return results

class LLMClassifier:
    """Paid LLM classifier via OpenRouter"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        self.categories = ['politics', 'business', 'technology', 'sports', 
                          'entertainment', 'health', 'science', 'world', 'crime']
        
        # Multiple models to test
        self.models = {
            'qwen_72b': "qwen/qwen-2.5-72b-instruct",  # Cheap & good
            'deepseek': "deepseek/deepseek-chat",       # Very cheap
            'claude_haiku': "anthropic/claude-3-haiku", # Fast
            'gpt4o_mini': "openai/gpt-4o-mini",         # OpenAI cheap
            'llama_70b': "meta-llama/llama-3.1-70b-instruct",  # Meta
            'mistral_large': "mistralai/mistral-large",  # Mistral
            'gemini_flash': "google/gemini-flash-1.5",   # Google
        }
    
    def classify_single(self, text: str, model: str) -> Dict[str, Any]:
        """Classify single text"""
        start_time = time.time()
        
        prompt = f"""Classify this news into one category: {self.categories}

News: {text}

Answer with only the category name:"""
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a news classifier. Answer only with the category name."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            # Find matching category
            category = 'world'  # default
            for cat in self.categories:
                if cat in result:
                    category = cat
                    break
            
            model_name = model.split("/")[-1] if "/" in model else model
            
            return {
                'category': category,
                'confidence': 0.8,  # Fixed confidence for LLM
                'method': f'llm_{model_name}',
                'time': time.time() - start_time
            }
            
        except Exception as e:
            print(f"LLM Error with {model}: {e}")
            model_name = model.split("/")[-1] if "/" in model else model
            return {
                'category': 'world',
                'confidence': 0.0,
                'method': f'llm_{model_name}_error',
                'time': time.time() - start_time
            }
    
    def classify_batch(self, news_list: List[Dict], model_key: str = 'qwen_72b') -> List[Dict]:
        """Classify multiple news items"""
        model = self.models.get(model_key, model_key)
        results = []
        
        for i, news in enumerate(news_list):
            if i % 10 == 0:
                print(f"  Processing {i}/{len(news_list)} with {model_key}...")
            
            text = f"{news['title']}. {news.get('description', '')}"
            result = self.classify_single(text, model)
            
            news_result = news.copy()
            news_result.update(result)
            results.append(news_result)
            
            # Rate limiting
            time.sleep(0.3)
        
        return results

class ExperimentRunner:
    """Main experiment runner"""
    
    def __init__(self, api_key: str = None):
        self.collector = NewsCollector()
        self.free_classifier = FreeClassifier()
        self.llm_classifier = LLMClassifier(api_key) if api_key else None
        
    def run_experiment(self, news_count: int = 50, test_all_llms: bool = False, 
                      test_mode: str = 'all', skip_slow: bool = False):
        """Run the complete experiment"""
        print("🚀 Starting COMPREHENSIVE News Classification Experiment")
        print("="*60)
        
        # Step 1: Collect news
        news_data = self.collector.collect_news(news_count)
        if not news_data:
            print("❌ No news collected!")
            return
        
        results = {}
        
        # Step 2: Test FREE methods based on mode
        print(f"\n💰 Testing FREE methods (mode: {test_mode}):")
        
        # Базовые методы
        basic_methods = [
            ('keyword', 'Keyword-based'),
            ('semantic', 'BERT Semantic (MiniLM)'),
            ('zero_shot', 'Zero-shot BART'),
            ('zero_shot_advanced', 'Zero-shot DeBERTa'),
        ]
        
        # Продвинутые эмбеддинг модели
        advanced_semantic_methods = [
            ('e5_base', 'E5-Base (Microsoft)'),
            ('bge_base', 'BGE-Base (BAAI)'),
            ('mpnet_base', 'MPNet-Base'),
            ('sentence_t5', 'Sentence-T5'),
            ('minilm', 'MiniLM (Reference)'),
        ]
        
        # Медленные продвинутые модели (пропускаем если skip_slow=True)
        if not skip_slow:
            advanced_semantic_methods.extend([
                ('e5_large', 'E5-Large (Microsoft)'),
                ('bge_large', 'BGE-Large (BAAI)'),
                ('instructor_base', 'Instructor-Base'),
                ('multilingual_e5', 'Multilingual E5'),
                ('gtr_t5', 'GTR-T5'),
            ])
        
        # TF-IDF методы
        tfidf_methods = [
            ('tfidf_nb', 'TF-IDF + Naive Bayes'),
            ('tfidf_svm', 'TF-IDF + SVM'),
            ('tfidf_rf', 'TF-IDF + Random Forest'),
            ('tfidf_ensemble', 'TF-IDF Ensemble'),
        ]
        
        # Ensemble методы
        ensemble_methods = [
            ('ensemble', 'Ensemble (4 models)'),
        ]
        
        # Выбираем методы в зависимости от режима
        if test_mode == 'basic':
            free_methods = basic_methods + tfidf_methods[:2]  # Только базовые + 2 TF-IDF
        elif test_mode == 'advanced':
            free_methods = advanced_semantic_methods
        elif test_mode == 'ensemble':
            free_methods = ensemble_methods + tfidf_methods[-1:]  # Ensemble методы
        else:  # 'all'
            free_methods = basic_methods + advanced_semantic_methods + tfidf_methods + ensemble_methods
        
        print(f"📋 Selected {len(free_methods)} methods for testing")
        
        for method_key, method_name in free_methods:
            print(f"  🔸 {method_name}...")
            try:
                method_results = self.free_classifier.classify_batch(news_data, method_key)
                results[f'free_{method_key}'] = method_results
                print(f"    ✅ Completed: {len(method_results)} classified")
            except Exception as e:
                print(f"    ❌ Error: {e}")
                results[f'free_{method_key}'] = []
        
        # Step 3: Test LLM methods (if available)
        if self.llm_classifier:
            print("\n💸 Testing PAID LLM methods:")
            
            if test_all_llms:
                # Test multiple LLMs
                llm_methods = [
                    ('qwen_72b', 'Qwen 2.5 72B'),
                    ('deepseek', 'DeepSeek Chat'),
                    ('claude_haiku', 'Claude Haiku'),
                    ('gpt4o_mini', 'GPT-4o Mini'),
                ]
            else:
                # Test just one cheap model
                llm_methods = [('qwen_72b', 'Qwen 2.5 72B')]
            
            for model_key, model_name in llm_methods:
                print(f"  🔸 {model_name}...")
                try:
                    # Test on smaller subset to save money
                    test_subset = news_data[:min(20, len(news_data))]
                    llm_results = self.llm_classifier.classify_batch(test_subset, model_key)
                    results[f'llm_{model_key}'] = llm_results
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    results[f'llm_{model_key}'] = []
        else:
            print("\n⚠️  LLM testing skipped (no API key)")
        
        # Step 4: Compare results
        print("\n📊 COMPREHENSIVE COMPARISON:")
        self.compare_methods(results)
        
        # Step 5: Save results
        self.save_results(results)
        
        return results
    
    def compare_methods(self, results: Dict[str, List[Dict]]):
        """Compare different methods"""
        print("\nDetailed Method Comparison:")
        print("-" * 80)
        print(f"{'Method':<25} {'Count':<8} {'Avg Time':<12} {'Categories':<12} {'Top Category'}")
        print("-" * 80)
        
        method_stats = []
        
        for method_name, method_results in results.items():
            if not method_results:
                print(f"{method_name:<25} {'FAILED':<8} {'-':<12} {'-':<12} {'-'}")
                continue
            
            count = len(method_results)
            avg_time = np.mean([r['time'] for r in method_results])
            categories = set(r['category'] for r in method_results)
            
            # Find most common category
            cat_counts = {}
            for r in method_results:
                cat = r['category']
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            top_category = max(cat_counts, key=cat_counts.get) if cat_counts else 'none'
            
            print(f"{method_name:<25} {count:<8} {avg_time:<12.3f} {len(categories):<12} {top_category}")
            
            method_stats.append({
                'method': method_name,
                'count': count,
                'avg_time': avg_time,
                'categories_used': len(categories),
                'category_distribution': cat_counts
            })
        
        # Show best performers
        print(f"\n🏆 SPEED RANKING (fastest first):")
        speed_ranking = sorted([s for s in method_stats if s['avg_time'] > 0], key=lambda x: x['avg_time'])
        for i, stat in enumerate(speed_ranking[:5], 1):
            print(f"  {i}. {stat['method']}: {stat['avg_time']:.3f}s")
        
        print(f"\n🎯 DIVERSITY RANKING (most categories used):")
        diversity_ranking = sorted(method_stats, key=lambda x: x['categories_used'], reverse=True)
        for i, stat in enumerate(diversity_ranking[:5], 1):
            print(f"  {i}. {stat['method']}: {stat['categories_used']} categories")
        
        # Show category preferences by method
        print(f"\n📊 Category Distribution by Method:")
        for stat in method_stats[:3]:  # Top 3 methods
            print(f"\n{stat['method']}:")
            sorted_cats = sorted(stat['category_distribution'].items(), key=lambda x: x[1], reverse=True)
            for cat, count in sorted_cats[:5]:  # Top 5 categories
                percentage = (count / stat['count']) * 100
                print(f"  {cat}: {count} ({percentage:.1f}%)")
    
    def save_results(self, results: Dict[str, List[Dict]]):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        os.makedirs('results', exist_ok=True)
        
        # Save comprehensive comparison
        filename = f"results/comprehensive_comparison_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary stats
        summary = {}
        for method_name, method_results in results.items():
            if method_results:
                summary[method_name] = {
                    'count': len(method_results),
                    'avg_time': np.mean([r['time'] for r in method_results]),
                    'categories': list(set(r['category'] for r in method_results)),
                    'category_counts': {cat: sum(1 for r in method_results if r['category'] == cat) 
                                      for cat in set(r['category'] for r in method_results)}
                }
        
        summary_filename = f"results/summary_{timestamp}.json"
        with open(summary_filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save individual CSVs
        for method_name, method_results in results.items():
            if method_results:
                df = pd.DataFrame(method_results)
                csv_filename = f"results/{method_name}_{timestamp}.csv"
                df.to_csv(csv_filename, index=False)
        
        print(f"\n💾 Results saved:")
        print(f"  📊 Main results: {filename}")
        print(f"  📈 Summary: {summary_filename}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='🚀 Comprehensive News Classification Comparison')
    parser.add_argument('--api-key', help='OpenRouter API key for LLM testing (optional)')
    parser.add_argument('--count', type=int, default=50, help='Number of news to classify (default: 50)')
    parser.add_argument('--free-only', action='store_true', help='Test only free methods')
    parser.add_argument('--all-llms', action='store_true', help='Test all LLM models (expensive!)')
    parser.add_argument('--basic-only', action='store_true', help='Test only basic methods (faster)')
    parser.add_argument('--advanced-only', action='store_true', help='Test only advanced embedding methods')
    parser.add_argument('--ensemble-only', action='store_true', help='Test only ensemble methods')
    parser.add_argument('--skip-slow', action='store_true', help='Skip slow models (large models)')
    
    args = parser.parse_args()
    
    # Get API key from env if not provided
    api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
    
    if args.free_only:
        api_key = None
        print("🆓 Running FREE-ONLY comprehensive experiment")
    elif not api_key:
        print("⚠️  No API key provided. Will test FREE methods only.")
        print("   Use --api-key or set OPENROUTER_API_KEY environment variable")
        print("   Or use --free-only flag")
    
    # Run experiment with mode selection
    runner = ExperimentRunner(api_key)
    
    # Determine test mode
    test_mode = 'all'
    if args.basic_only:
        test_mode = 'basic'
        print("🎯 Testing BASIC methods only")
    elif args.advanced_only:
        test_mode = 'advanced'
        print("🔬 Testing ADVANCED embedding methods only")
    elif args.ensemble_only:
        test_mode = 'ensemble'
        print("🤝 Testing ENSEMBLE methods only")
    
    runner.run_experiment(args.count, args.all_llms, test_mode, args.skip_slow)

if __name__ == "__main__":
    main() 