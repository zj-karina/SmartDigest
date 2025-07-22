"""Enhanced news classifier with extended categories and relevance filtering."""

import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import torch
import warnings
warnings.filterwarnings('ignore')


class EnhancedNewsClassifier:
    """Расширенный классификатор новостей с улучшенными категориями и фильтрацией релевантности."""
    
    def __init__(self, method: str = 'bge_large', relevance_threshold: float = 0.6):
        """
        Инициализация классификатора.
        
        Args:
            method: Метод классификации
            relevance_threshold: Порог релевантности (ниже = нерелевантная статья)
        """
        self.method = method
        self.relevance_threshold = relevance_threshold
        
        # Расширенные категории с более детальной классификацией
        self.categories = [
            'politics_domestic', 'politics_international', 'politics_elections',
            'business_markets', 'business_tech', 'business_economy', 
            'technology_ai', 'technology_software', 'technology_hardware',
            'sports_football', 'sports_other', 'sports_olympics',
            'entertainment_movies', 'entertainment_music', 'entertainment_tv',
            'health_medical', 'health_pandemic', 'health_mental',
            'science_research', 'science_climate', 'science_space',
            'world_conflicts', 'world_diplomacy', 'world_disasters',
            'crime_investigations', 'crime_courts', 'crime_terrorism',
            'education', 'transportation', 'environment', 'social_issues'
        ]
        
        # Описания категорий для лучшей классификации
        self.category_descriptions = {
            'politics_domestic': 'Внутренняя политика, местные политики, правительственные решения',
            'politics_international': 'Международная политика, дипломатия, отношения между странами',
            'politics_elections': 'Выборы, избирательные кампании, результаты голосования',
            
            'business_markets': 'Фондовые рынки, инвестиции, торговля, финансы',
            'business_tech': 'Технологические компании, стартапы, венчурные инвестиции',
            'business_economy': 'Экономические показатели, инфляция, безработица, ВВП',
            
            'technology_ai': 'Искусственный интеллект, машинное обучение, нейронные сети',
            'technology_software': 'Программное обеспечение, приложения, операционные системы',
            'technology_hardware': 'Компьютеры, смартфоны, процессоры, электроника',
            
            'sports_football': 'Футбол, чемпионаты, матчи, футболисты',
            'sports_other': 'Другие виды спорта, соревнования, спортсмены',
            'sports_olympics': 'Олимпийские игры, международные спортивные соревнования',
            
            'entertainment_movies': 'Фильмы, кинематограф, актеры, режиссеры',
            'entertainment_music': 'Музыка, концерты, музыканты, альбомы',
            'entertainment_tv': 'Телевидение, сериалы, шоу, телеведущие',
            
            'health_medical': 'Медицинские исследования, лечение, больницы, врачи',
            'health_pandemic': 'Пандемии, эпидемии, вирусы, вакцины',
            'health_mental': 'Психическое здоровье, стресс, депрессия',
            
            'science_research': 'Научные исследования, открытия, технологии',
            'science_climate': 'Изменение климата, экология, глобальное потепление',
            'science_space': 'Космос, астрономия, космические миссии',
            
            'world_conflicts': 'Военные конфликты, войны, терроризм',
            'world_diplomacy': 'Дипломатические отношения, переговоры, соглашения',
            'world_disasters': 'Природные катастрофы, аварии, чрезвычайные ситуации',
            
            'crime_investigations': 'Расследования преступлений, полиция, детективы',
            'crime_courts': 'Судебные процессы, приговоры, правосудие',
            'crime_terrorism': 'Терроризм, экстремизм, безопасность',
            
            'education': 'Образование, университеты, школы, студенты',
            'transportation': 'Транспорт, автомобили, авиация, железные дороги',
            'environment': 'Окружающая среда, загрязнение, природа',
            'social_issues': 'Социальные проблемы, права человека, неравенство'
        }
        
        # Ключевые слова для определения нерелевантных статей
        self.irrelevant_keywords = [
            'advertisement', 'sponsored', 'promotion', 'casino', 'gambling',
            'adult content', 'explicit', 'nsfw', 'clickbait', 'fake news',
            'horoscope', 'astrology', 'fortune telling', 'lottery',
            'weight loss miracle', 'get rich quick', 'earn money fast',
            'celebrity gossip', 'tabloid', 'scandal', 'rumor',
            'spam', 'scam', 'fraud', 'phishing', 'malware'
        ]
        
        # Ключевые слова для определения релевантных новостей
        self.relevant_keywords = [
            'breaking news', 'developing story', 'latest update', 'official statement',
            'government', 'president', 'minister', 'congress', 'parliament',
            'company', 'market', 'stock', 'economy', 'financial',
            'research', 'study', 'scientist', 'discovery', 'innovation',
            'health', 'medical', 'treatment', 'vaccine', 'disease',
            'technology', 'artificial intelligence', 'software', 'hardware',
            'sports', 'championship', 'tournament', 'athlete', 'team',
            'climate', 'environment', 'disaster', 'emergency'
        ]
        
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
        """Загружает указанную модель."""
        if self.model is not None:
            return
            
        print(f"🔄 Загрузка модели {self.method}...")
        
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
            # Fallback модель
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"✅ Модель загружена")
    
    def _check_relevance(self, text: str) -> Tuple[bool, float]:
        """
        Проверяет релевантность статьи.
        
        Args:
            text: Текст для анализа
            
        Returns:
            Tuple[is_relevant, confidence]
        """
        text_lower = text.lower()
        
        # Подсчет нерелевантных ключевых слов
        irrelevant_score = sum(1 for keyword in self.irrelevant_keywords if keyword in text_lower)
        
        # Подсчет релевантных ключевых слов
        relevant_score = sum(1 for keyword in self.relevant_keywords if keyword in text_lower)
        
        # Простая эвристика
        if irrelevant_score > 2:  # Много нерелевантных слов
            return False, 0.1
        
        if relevant_score == 0 and len(text.split()) < 10:  # Слишком короткий текст без релевантных слов
            return False, 0.3
        
        # Проверка длины и качества текста
        words = text.split()
        if len(words) < 5:  # Слишком короткий
            return False, 0.2
        
        if len(words) > 500:  # Слишком длинный (возможно spam)
            return False, 0.2
        
        # Релевантность на основе соотношения
        if relevant_score > 0:
            confidence = min(0.9, 0.5 + (relevant_score * 0.1))
            return True, confidence
        
        # Дефолтная релевантность для обычных статей
        return True, 0.7
    
    def classify_single(self, text: str) -> Dict[str, Any]:
        """
        Классифицирует одну статью.
        
        Args:
            text: Текст для классификации
            
        Returns:
            Результат классификации
        """
        self._load_model()
        start_time = time.time()
        
        # Сначала проверяем релевантность
        is_relevant, relevance_confidence = self._check_relevance(text)
        
        result = {
            'is_relevant': is_relevant,
            'relevance_confidence': relevance_confidence,
            'method': self.method,
            'time': time.time() - start_time
        }
        
        if not is_relevant:
            result.update({
                'category': 'irrelevant',
                'confidence': 0.0
            })
            return result
        
        # Классификация по категориям
        if self.method == 'zero_shot':
            classification_result = self.classifier(text, self.categories)
            result.update({
                'category': classification_result['labels'][0],
                'confidence': classification_result['scores'][0]
            })
        else:
            # Семантическое сходство
            text_emb = self.model.encode([text])
            
            # Обработка E5 моделей
            if 'e5' in self.method:
                text_emb = self.model.encode([f"query: {text}"])
                cat_embs = self.model.encode([f"passage: {desc}" for desc in self.category_descriptions.values()])
            else:
                cat_embs = self.model.encode(list(self.category_descriptions.values()))
            
            similarities = cosine_similarity(text_emb, cat_embs)[0]
            best_idx = np.argmax(similarities)
            
            result.update({
                'category': self.categories[best_idx],
                'confidence': float(similarities[best_idx])
            })
        
        result['time'] = time.time() - start_time
        return result
    
    def classify_batch(self, news_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Классифицирует множество статей.
        
        Args:
            news_list: Список новостей с 'title' и 'description'
            
        Returns:
            Список новостей с результатами классификации
        """
        print(f"🗂️ Классификация {len(news_list)} статей с {self.method}...")
        
        results = []
        relevant_count = 0
        
        for i, news in enumerate(news_list):
            if i % 20 == 0:
                print(f"  Обработка {i}/{len(news_list)}...")
            
            text = f"{news['title']}. {news.get('description', '')}"
            classification = self.classify_single(text)
            
            if classification['is_relevant']:
                relevant_count += 1
            
            result = news.copy()
            result.update(classification)
            results.append(result)
        
        print(f"✅ Классификация завершена")
        print(f"📊 Релевантных статей: {relevant_count}/{len(news_list)} ({relevant_count/len(news_list)*100:.1f}%)")
        
        return results
    
    def get_category_hierarchy(self) -> Dict[str, List[str]]:
        """
        Возвращает иерархию категорий для группировки.
        
        Returns:
            Словарь с основными категориями и их подкатегориями
        """
        return {
            'politics': ['politics_domestic', 'politics_international', 'politics_elections'],
            'business': ['business_markets', 'business_tech', 'business_economy'],
            'technology': ['technology_ai', 'technology_software', 'technology_hardware'],
            'sports': ['sports_football', 'sports_other', 'sports_olympics'],
            'entertainment': ['entertainment_movies', 'entertainment_music', 'entertainment_tv'],
            'health': ['health_medical', 'health_pandemic', 'health_mental'],
            'science': ['science_research', 'science_climate', 'science_space'],
            'world': ['world_conflicts', 'world_diplomacy', 'world_disasters'],
            'crime': ['crime_investigations', 'crime_courts', 'crime_terrorism'],
            'society': ['education', 'transportation', 'environment', 'social_issues']
        }
    
    def get_main_category(self, subcategory: str) -> str:
        """
        Получает основную категорию для подкатегории.
        
        Args:
            subcategory: Подкатегория
            
        Returns:
            Основная категория
        """
        hierarchy = self.get_category_hierarchy()
        for main_cat, sub_cats in hierarchy.items():
            if subcategory in sub_cats:
                return main_cat
        return 'other'
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Возвращает список доступных методов классификации."""
        return ['bge_large', 'bge_base', 'e5_large', 'e5_base', 'mpnet', 'minilm', 'zero_shot'] 