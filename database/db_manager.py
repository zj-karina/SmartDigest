"""MongoDB integration for SmartDigest pipeline."""

import os
import hashlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
import json
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Управляет подключением и операциями с MongoDB."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Инициализация менеджера БД.
        
        Args:
            connection_string: Строка подключения к MongoDB
        """
        self.connection_string = connection_string or os.getenv('MONGODB_CONNECTION_STRING')
        if not self.connection_string:
            raise ValueError("MongoDB connection string не найден")
        
        self.client: Optional[MongoClient] = None
        self.db: Optional[Database] = None
        self._connect()
        
    def _connect(self):
        """Устанавливает подключение к MongoDB."""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client.smartdigest
            
            # Проверяем соединение
            self.client.admin.command('ping')
            logger.info("✅ Подключение к MongoDB установлено")
            
            # Создаем индексы
            self._create_indexes()
            
        except Exception as e:
            logger.error(f"❌ Ошибка подключения к MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Создает необходимые индексы для оптимизации запросов."""
        collections = {
            'articles': [
                ('url', ASCENDING),
                ('title_hash', ASCENDING),
                ('collected_at', DESCENDING),
                ('source', ASCENDING),
                ('category', ASCENDING)
            ],
            'digests': [
                ('created_at', DESCENDING),
                ('language', ASCENDING),
                ('pipeline_version', ASCENDING)
            ],
            'processing_history': [
                ('article_id', ASCENDING),
                ('processed_at', DESCENDING),
                ('pipeline_version', ASCENDING)
            ]
        }
        
        for collection_name, indexes in collections.items():
            collection = self.db[collection_name]
            for index_fields in indexes:
                collection.create_index([index_fields])
    
    def _generate_article_hash(self, article: Dict[str, Any]) -> str:
        """Генерирует хэш для статьи на основе заголовка и контента."""
        content = f"{article.get('title', '')}{article.get('description', '')}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def save_articles(self, articles: List[Dict[str, Any]], batch_size: int = 100) -> List[str]:
        """
        Сохраняет статьи в БД с проверкой дубликатов.
        
        Args:
            articles: Список статей для сохранения
            batch_size: Размер батча для вставки
            
        Returns:
            Список ID сохраненных статей
        """
        if not articles:
            return []
        
        collection = self.db.articles
        saved_ids = []
        
        # Обрабатываем статьи батчами
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            processed_batch = []
            
            for article in batch:
                # Генерируем хэш заголовка
                title_hash = self._generate_article_hash(article)
                
                # Проверяем дубликат
                existing = collection.find_one({'title_hash': title_hash})
                if existing:
                    logger.info(f"Дубликат найден: {article.get('title', '')[:50]}...")
                    saved_ids.append(str(existing['_id']))
                    continue
                
                # Подготавливаем документ
                document = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', ''),
                    'published': article.get('published'),
                    'title_hash': title_hash,
                    'collected_at': datetime.now(timezone.utc),
                    'language': article.get('language'),
                    'category': article.get('category'),
                    'is_relevant': article.get('is_relevant', True),
                    'classification_confidence': article.get('classification_confidence'),
                    'raw_data': article  # Сохраняем оригинальные данные
                }
                
                processed_batch.append(document)
            
            # Вставляем батч
            if processed_batch:
                try:
                    result = collection.insert_many(processed_batch)
                    batch_ids = [str(id_) for id_ in result.inserted_ids]
                    saved_ids.extend(batch_ids)
                    logger.info(f"Сохранено {len(processed_batch)} статей в батче")
                except Exception as e:
                    logger.error(f"Ошибка сохранения батча: {e}")
        
        logger.info(f"Всего сохранено/найдено {len(saved_ids)} статей")
        return saved_ids
    
    def get_articles(self, 
                    limit: int = 100,
                    category: Optional[str] = None,
                    source: Optional[str] = None,
                    is_relevant: Optional[bool] = None,
                    date_from: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Получает статьи из БД с фильтрацией.
        
        Args:
            limit: Максимальное количество статей
            category: Фильтр по категории
            source: Фильтр по источнику
            is_relevant: Фильтр по релевантности
            date_from: Фильтр по дате (статьи новее указанной даты)
            
        Returns:
            Список статей
        """
        collection = self.db.articles
        
        # Строим фильтр
        filter_query = {}
        if category:
            filter_query['category'] = category
        if source:
            filter_query['source'] = source
        if is_relevant is not None:
            filter_query['is_relevant'] = is_relevant
        if date_from:
            filter_query['collected_at'] = {'$gte': date_from}
        
        # Выполняем запрос
        cursor = collection.find(filter_query).sort('collected_at', DESCENDING).limit(limit)
        articles = list(cursor)
        
        # Конвертируем ObjectId в строки
        for article in articles:
            article['_id'] = str(article['_id'])
        
        return articles
    
    def save_digest(self, digest: Dict[str, Any], pipeline_config: Dict[str, Any]) -> str:
        """
        Сохраняет готовый дайджест в БД.
        
        Args:
            digest: Данные дайджеста
            pipeline_config: Конфигурация пайплайна
            
        Returns:
            ID сохраненного дайджеста
        """
        collection = self.db.digests
        
        document = {
            'digest_data': digest,
            'pipeline_config': pipeline_config,
            'created_at': datetime.now(timezone.utc),
            'language': digest.get('language', 'unknown'),
            'events_count': len(digest.get('items', [])),
            'articles_processed': digest.get('articles_processed', 0),
            'pipeline_version': pipeline_config.get('version', '1.0')
        }
        
        result = collection.insert_one(document)
        digest_id = str(result.inserted_id)
        
        logger.info(f"Дайджест сохранен с ID: {digest_id}")
        return digest_id
    
    def get_digests(self, limit: int = 10, language: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Получает последние дайджесты.
        
        Args:
            limit: Количество дайджестов
            language: Фильтр по языку
            
        Returns:
            Список дайджестов
        """
        collection = self.db.digests
        
        filter_query = {}
        if language:
            filter_query['language'] = language
        
        cursor = collection.find(filter_query).sort('created_at', DESCENDING).limit(limit)
        digests = list(cursor)
        
        # Конвертируем ObjectId в строки
        for digest in digests:
            digest['_id'] = str(digest['_id'])
        
        return digests
    
    def mark_articles_processed(self, article_ids: List[str], pipeline_config: Dict[str, Any]):
        """
        Отмечает статьи как обработанные.
        
        Args:
            article_ids: Список ID статей
            pipeline_config: Конфигурация пайплайна
        """
        collection = self.db.processing_history
        
        documents = []
        for article_id in article_ids:
            documents.append({
                'article_id': article_id,
                'processed_at': datetime.now(timezone.utc),
                'pipeline_version': pipeline_config.get('version', '1.0'),
                'pipeline_config': pipeline_config
            })
        
        if documents:
            collection.insert_many(documents)
            logger.info(f"Отмечено как обработанные {len(documents)} статей")
    
    def update_article_relevance(self, article_id: str, is_relevant: bool, confidence: float = 0.0):
        """
        Обновляет информацию о релевантности статьи.
        
        Args:
            article_id: ID статьи
            is_relevant: Релевантность статьи
            confidence: Уверенность в классификации
        """
        collection = self.db.articles
        
        result = collection.update_one(
            {'_id': article_id},
            {
                '$set': {
                    'is_relevant': is_relevant,
                    'relevance_confidence': confidence,
                    'relevance_updated_at': datetime.now(timezone.utc)
                }
            }
        )
        
        if result.modified_count > 0:
            logger.info(f"Обновлена релевантность статьи {article_id}: {is_relevant}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Получает статистику по базе данных.
        
        Returns:
            Словарь со статистикой
        """
        stats = {}
        
        # Статистика по статьям
        articles_collection = self.db.articles
        stats['total_articles'] = articles_collection.count_documents({})
        stats['relevant_articles'] = articles_collection.count_documents({'is_relevant': True})
        stats['irrelevant_articles'] = articles_collection.count_documents({'is_relevant': False})
        
        # Статистика по категориям
        pipeline = [
            {'$group': {'_id': '$category', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]
        stats['categories'] = list(articles_collection.aggregate(pipeline))
        
        # Статистика по источникам
        pipeline = [
            {'$group': {'_id': '$source', 'count': {'$sum': 1}}},
            {'$sort': {'count': -1}}
        ]
        stats['sources'] = list(articles_collection.aggregate(pipeline))
        
        # Статистика по дайджестам
        digests_collection = self.db.digests
        stats['total_digests'] = digests_collection.count_documents({})
        
        return stats
    
    def close(self):
        """Закрывает соединение с БД."""
        if self.client:
            self.client.close()
            logger.info("Соединение с MongoDB закрыто") 