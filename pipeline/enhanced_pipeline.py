"""Enhanced news digest pipeline with MongoDB integration and improved classification."""

import os
import time
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

from collectors.rss_collector import NewsCollector
from classifiers.enhanced_classifier import EnhancedNewsClassifier
from clustering.news_clusterer import NewsClusterer
from summarization.news_summarizer import NewsSummarizer
from database.db_manager import DatabaseManager

def convert_numpy_types(obj):
    """Конвертирует numpy типы в обычные Python типы для JSON сериализации."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'isoformat'):  # datetime objects
        return obj.isoformat()
    return obj

class EnhancedDigestPipeline:
    """Улучшенный пайплайн обработки новостей с интеграцией MongoDB и расширенной классификацией."""
    
    def __init__(self, 
                 mongodb_connection: Optional[str] = None,
                 openrouter_api_key: Optional[str] = None,
                 classification_method: str = 'bge_large',
                 clustering_eps: float = 0.35,
                 summarization_model: str = 'balanced',
                 relevance_threshold: float = 0.6,
                 use_database: bool = True):
        """
        Инициализация пайплайна.
        
        Args:
            mongodb_connection: Строка подключения к MongoDB
            openrouter_api_key: API ключ для суммаризации
            classification_method: Метод классификации
            clustering_eps: Порог кластеризации
            summarization_model: Тип модели для суммаризации
            relevance_threshold: Порог релевантности
            use_database: Использовать ли базу данных
        """
        print("🚀 Инициализация Enhanced News Digest Pipeline")
        
        # Компоненты
        self.collector = NewsCollector()
        self.classifier = EnhancedNewsClassifier(
            method=classification_method,
            relevance_threshold=relevance_threshold
        )
        self.clusterer = NewsClusterer()
        self.summarizer = NewsSummarizer(openrouter_api_key)
        
        # База данных
        self.use_database = use_database
        self.db_manager = None
        if use_database:
            try:
                connection_string = (mongodb_connection or 
                                   os.getenv('MONGODB_CONNECTION_STRING'))
                self.db_manager = DatabaseManager(connection_string)
            except Exception as e:
                print(f"⚠️ Ошибка подключения к БД: {e}")
                print("📝 Продолжаем без базы данных")
                self.use_database = False
        
        # Параметры
        self.classification_method = classification_method
        self.clustering_eps = clustering_eps
        self.summarization_model = summarization_model
        self.relevance_threshold = relevance_threshold
        
        # Состояние
        self.raw_news = []
        self.classified_news = []
        self.relevant_news = []
        self.irrelevant_news = []
        self.clusters = {}
        self.digest = {}
        
        print(f"✅ Pipeline готов")
        print(f"   📊 Классификация: {classification_method}")
        print(f"   🔍 Кластеризация: eps={clustering_eps}")
        print(f"   📝 Суммаризация: {summarization_model}")
        print(f"   💾 База данных: {'✅' if self.use_database else '❌'}")
        print(f"   🎯 Порог релевантности: {relevance_threshold}")
    
    def run_complete_pipeline(self, 
                             news_count: int = 50,
                             max_events: int = 7,
                             language: str = 'english',
                             save_results: bool = True,
                             save_to_db: bool = True) -> Dict[str, Any]:
        """
        Запуск полного пайплайна.
        
        Args:
            news_count: Количество новостных статей для сбора
            max_events: Максимальное количество событий в дайджесте
            language: Язык дайджеста ('english', 'russian')
            save_results: Сохранять ли результаты в файлы
            save_to_db: Сохранять ли в базу данных
            
        Returns:
            Итоговый дайджест
        """
        pipeline_start = time.time()
        
        print("🚀 Запуск Complete Pipeline")
        print("=" * 40)
        print(f"Параметры:")
        print(f"   📰 Количество новостей: {news_count}")
        print(f"   📊 Макс. событий: {max_events}")
        print(f"   🌍 Язык: {language}")
        print(f"   💾 Сохранение в БД: {'✅' if save_to_db and self.use_database else '❌'}")
        
        try:
            # Шаг 1: Чтение новостей из MongoDB
            print(f"\n📰 ШАГ 1: Чтение новостей из MongoDB")
            print("-" * 25)
            
            start_time = time.time()
            
            if self.use_database and self.db_manager:
                # Читаем новости из MongoDB
                self.raw_news = self.db_manager.get_articles(limit=news_count)
                print(f"📊 Прочитано {len(self.raw_news)} статей из MongoDB")
                
                # Конвертируем формат MongoDB в формат для классификации
                converted_news = []
                for article in self.raw_news:
                    converted_article = {
                        'title': article.get('title', ''),
                        'description': article.get('summary', ''),  # summary вместо description
                        'url': article.get('url', ''),
                        'source': article.get('url', '').split('/')[2] if article.get('url') else 'unknown',
                        'published': article.get('publish_date'),
                        'language': 'english'  # Предполагаем английский
                    }
                    converted_news.append(converted_article)
                
                self.raw_news = converted_news
                collection_time = time.time() - start_time
                print(f"✅ Прочитано {len(self.raw_news)} статей за {collection_time:.1f}с")
            else:
                # Fallback на RSS сбор
                print("⚠️ База данных недоступна, используем RSS сбор")
                self.raw_news = self.collector.collect_news(news_count)
                collection_time = time.time() - start_time
                print(f"✅ Собрано {len(self.raw_news)} статей за {collection_time:.1f}с")
            
            if not self.raw_news:
                print("❌ Новости не найдены!")
                return {}
            
            # Сохранение сырых данных в БД (отключено для режима только чтения)
            if self.use_database and save_to_db and False:  # Отключено
                print("💾 Сохранение статей в БД...")
                saved_ids = self.db_manager.save_articles(self.raw_news)
                print(f"✅ Сохранено {len(saved_ids)} статей в БД")
            
            # Шаг 2: Классификация и фильтрация релевантности
            print(f"\n🗂️ ШАГ 2: Классификация и фильтрация")
            print("-" * 25)
            
            start_time = time.time()
            self.classified_news = self.classifier.classify_batch(self.raw_news)
            classification_time = time.time() - start_time
            
            # Разделение на релевантные и нерелевантные
            self.relevant_news = [news for news in self.classified_news if news.get('is_relevant', True)]
            self.irrelevant_news = [news for news in self.classified_news if not news.get('is_relevant', True)]
            
            print(f"✅ Классификация завершена за {classification_time:.1f}с")
            print(f"📊 Релевантных статей: {len(self.relevant_news)}")
            print(f"🗑️ Нерелевантных статей: {len(self.irrelevant_news)}")
            
            # Статистика по категориям
            from collections import Counter
            category_stats = Counter([news.get('category', 'unknown') for news in self.relevant_news])
            print(f"📈 Топ категории: {dict(list(category_stats.most_common(5)))}")
            
            if not self.relevant_news:
                print("❌ Нет релевантных новостей для обработки!")
                return {}
            
            # Шаг 3: Кластеризация
            print(f"\n🔍 ШАГ 3: Кластеризация")
            print("-" * 25)
            
            start_time = time.time()
            self.clusters = self.clusterer.cluster_by_category(
                self.relevant_news, 
                eps=self.clustering_eps, 
                min_samples=2
            )
            clustering_time = time.time() - start_time
            
            # Получение топ кластеров для суммаризации
            top_clusters = self.clusterer.get_top_clusters(
                min_size=2, 
                max_clusters=max_events
            )
            
            print(f"✅ Кластеризация завершена за {clustering_time:.1f}с")
            print(f"🎯 Найдено {len(top_clusters)} значимых событий")
            
            if not top_clusters:
                print("❌ Значимые события для дайджеста не найдены")
                return {}
            
            # Шаг 4: Суммаризация
            print(f"\n📝 ШАГ 4: Суммаризация")
            print("-" * 25)
            
            if not self.summarizer.api_key:
                print("⚠️ Нет API ключа - используются fallback резюме")
            
            start_time = time.time()
            self.digest = self.summarizer.create_daily_digest(
                top_clusters,
                max_items=max_events,
                model_type=self.summarization_model,
                language=language
            )
            summarization_time = time.time() - start_time
            
            # Добавление метаданных
            self.digest.update({
                'articles_processed': len(self.classified_news),
                'relevant_articles': len(self.relevant_news),
                'irrelevant_articles': len(self.irrelevant_news),
                'processing_time': time.time() - pipeline_start,
                'pipeline_version': '2.0_enhanced'
            })
            
            total_time = time.time() - pipeline_start
            
            # Финальная статистика
            print(f"\n🎉 ПАЙПЛАЙН ЗАВЕРШЕН")
            print("=" * 40)
            print(f"⏱️ Общее время: {total_time:.1f}с")
            print(f"📰 Обработано: {len(self.classified_news)} статей")
            print(f"✅ Релевантных: {len(self.relevant_news)} статей")
            print(f"❌ Нерелевантных: {len(self.irrelevant_news)} статей")
            print(f"📊 Событий в дайджесте: {len(self.digest.get('items', []))}")
            print(f"🎯 Дайджест: {self.digest.get('title', 'Готов')}")
            
            # Сохранение результатов
            if save_results:
                self.save_results()
            
            # Сохранение в БД (отключено для режима только чтения)
            if self.use_database and save_to_db and False:  # Отключено
                self.save_to_database()
            
            return self.digest
            
        except Exception as e:
            print(f"\n❌ Ошибка пайплайна: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def save_to_database(self):
        """Сохраняет результаты в базу данных."""
        if not self.use_database or not self.db_manager:
            return
        
        try:
            print("\n💾 Сохранение в базу данных...")
            
            # Конфигурация пайплайна
            pipeline_config = {
                'classification_method': self.classification_method,
                'clustering_eps': self.clustering_eps,
                'summarization_model': self.summarization_model,
                'relevance_threshold': self.relevance_threshold,
                'version': '2.0_enhanced'
            }
            
            # Сохранение дайджеста
            if self.digest:
                digest_id = self.db_manager.save_digest(self.digest, pipeline_config)
                print(f"✅ Дайджест сохранен в БД с ID: {digest_id}")
            
            # Обновление информации о релевантности статей
            for news in self.classified_news:
                if hasattr(news, '_id'):  # Если статья уже была сохранена
                    self.db_manager.update_article_relevance(
                        news['_id'],
                        news.get('is_relevant', True),
                        news.get('relevance_confidence', 0.0)
                    )
            
            print("✅ Результаты сохранены в БД")
            
        except Exception as e:
            print(f"❌ Ошибка сохранения в БД: {e}")
    
    def save_results(self) -> Dict[str, str]:
        """Сохраняет все результаты пайплайна в файлы."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        print(f"\n💾 Сохранение результатов...")
        
        try:
            os.makedirs('results', exist_ok=True)
            
            # Сохранение результатов классификации
            if self.classified_news:
                classification_file = f'results/classification_{timestamp}.json'
                classification_data = {
                    'metadata': {
                        'timestamp': timestamp,
                        'total_articles': len(self.classified_news),
                        'relevant_articles': len(self.relevant_news),
                        'irrelevant_articles': len(self.irrelevant_news),
                        'method': self.classification_method,
                        'relevance_threshold': self.relevance_threshold
                    },
                    'articles': convert_numpy_types(self.classified_news)
                }
                
                with open(classification_file, 'w', encoding='utf-8') as f:
                    json.dump(classification_data, f, indent=2, ensure_ascii=False)
                
                saved_files['classification'] = classification_file
                print(f"📊 Классификация: {classification_file}")
            
            # Сохранение нерелевантных статей отключено для режима только чтения
            
            # Сохранение кластеров
            if self.clusters:
                clusters_file = f'results/clusters_{timestamp}.json'
                clusters_data = convert_numpy_types(self.clusters)
                
                with open(clusters_file, 'w', encoding='utf-8') as f:
                    json.dump(clusters_data, f, indent=2, ensure_ascii=False)
                
                saved_files['clusters'] = clusters_file
                print(f"🔍 Кластеры: {clusters_file}")
            
            # Сохранение дайджеста
            if self.digest:
                digest_file = f'results/digest_{timestamp}.json'
                digest_data = convert_numpy_types(self.digest)
                
                with open(digest_file, 'w', encoding='utf-8') as f:
                    json.dump(digest_data, f, indent=2, ensure_ascii=False)
                
                saved_files['digest'] = digest_file
                print(f"📝 Дайджест: {digest_file}")
                
                # Также сохраняем в формате Telegram
                telegram_file = f'results/telegram_{timestamp}.md'
                telegram_text = self.summarizer.format_for_telegram(self.digest)
                
                with open(telegram_file, 'w', encoding='utf-8') as f:
                    f.write(telegram_text)
                
                saved_files['telegram'] = telegram_file
                print(f"📱 Telegram: {telegram_file}")
            
            print(f"✅ Все результаты сохранены")
            return saved_files
            
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return {}
    
    def show_digest_preview(self):
        """Показывает превью дайджеста."""
        if not self.digest:
            print("❌ Дайджест не создан")
            return
        
        print(f"\n📰 ПРЕВЬЮ ДАЙДЖЕСТА")
        print("=" * 50)
        print(f"🗓️ {self.digest.get('date', 'N/A')}")
        print(f"📰 {self.digest.get('title', 'Дайджест новостей')}")
        print("-" * 50)
        
        for i, item in enumerate(self.digest.get('items', []), 1):
            print(f"\n{i}. 📍 {item.get('category', 'Unknown').upper()}")
            print(f"   {item.get('summary', 'Нет резюме')}")
            print(f"   📊 Статей: {len(item.get('articles', []))}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получает статистику работы пайплайна."""
        stats = {
            'pipeline': {
                'articles_collected': len(self.raw_news),
                'articles_classified': len(self.classified_news),
                'relevant_articles': len(self.relevant_news),
                'irrelevant_articles': len(self.irrelevant_news),
                'events_created': len(self.digest.get('items', [])),
                'classification_method': self.classification_method,
                'relevance_threshold': self.relevance_threshold
            }
        }
        
        # Статистика из БД если доступна
        if self.use_database and self.db_manager:
            try:
                db_stats = self.db_manager.get_statistics()
                stats['database'] = db_stats
            except Exception as e:
                print(f"⚠️ Ошибка получения статистики БД: {e}")
        
        return stats
    
    def close(self):
        """Закрывает соединения."""
        if self.db_manager:
            self.db_manager.close() 
