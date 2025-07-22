#!/usr/bin/env python3
"""
Enhanced pipeline runner с интеграцией MongoDB и улучшенной классификацией.
Includes database integration, enhanced categories, and irrelevant article filtering.
"""

import sys
import os
import argparse

from pipeline.enhanced_pipeline import EnhancedDigestPipeline


def main():
    parser = argparse.ArgumentParser(description='SmartDigest - Enhanced Pipeline с MongoDB')
    parser.add_argument('--news-count', type=int, default=50, 
                       help='Количество новостных статей для обработки (default: 50)')
    parser.add_argument('--max-events', type=int, default=7,
                       help='Максимальное количество событий в дайджесте (default: 7)')
    parser.add_argument('--language', choices=['english', 'russian'], default='english',
                       help='Язык дайджеста (default: english)')
    parser.add_argument('--classification', default='bge_large',
                       help='Метод классификации (default: bge_large)')
    parser.add_argument('--clustering-eps', type=float, default=0.35,
                       help='Порог кластеризации (default: 0.35)')
    parser.add_argument('--model', choices=['fast', 'balanced', 'quality', 'premium'], 
                       default='balanced', help='Модель суммаризации (default: balanced)')
    parser.add_argument('--relevance-threshold', type=float, default=0.6,
                       help='Порог релевантности (default: 0.6)')
    parser.add_argument('--api-key', help='OpenRouter API key (или установите OPENROUTER_API_KEY)')
    parser.add_argument('--mongodb-connection', help='MongoDB connection string (или установите MONGODB_CONNECTION_STRING)')
    parser.add_argument('--no-save', action='store_true', help='Не сохранять результаты в файлы')
    parser.add_argument('--no-database', action='store_true', help='Отключить интеграцию с базой данных')
    parser.add_argument('--preview-only', action='store_true', help='Показать только превью')
    parser.add_argument('--stats', action='store_true', help='Показать статистику после завершения')
    
    args = parser.parse_args()
    
    print("🚀 SmartDigest - Enhanced Pipeline с MongoDB")
    print("=" * 60)
    print(f"Настройки:")
    print(f"  📰 Количество новостей: {args.news_count}")
    print(f"  📊 Макс. событий: {args.max_events}")
    print(f"  🌍 Язык: {args.language}")
    print(f"  🗂️ Классификация: {args.classification}")
    print(f"  🔍 Кластеризация eps: {args.clustering_eps}")
    print(f"  📝 Модель: {args.model}")
    print(f"  🎯 Порог релевантности: {args.relevance_threshold}")
    print(f"  💾 База данных: {'❌' if args.no_database else '✅'}")
    
    # Проверка API ключа
    api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
    if api_key:
        print(f"  🔑 API: OpenRouter (включен)")
    else:
        print(f"  ⚠️ API: Нет (fallback резюме)")
    
    # Проверка MongoDB
    mongodb_connection = args.mongodb_connection or os.getenv('MONGODB_CONNECTION_STRING')
    if not args.no_database and mongodb_connection:
        print(f"  🗄️ MongoDB: Подключение настроено")
    elif not args.no_database:
        print(f"  ⚠️ MongoDB: Будет использоваться стандартное подключение")
    else:
        print(f"  ❌ MongoDB: Отключена")
    
    print("=" * 60)
    
    try:
        # Создание улучшенного пайплайна
        pipeline = EnhancedDigestPipeline(
            mongodb_connection=mongodb_connection,
            openrouter_api_key=api_key,
            classification_method=args.classification,
            clustering_eps=args.clustering_eps,
            summarization_model=args.model,
            relevance_threshold=args.relevance_threshold,
            use_database=not args.no_database
        )
        
        # Запуск пайплайна
        digest = pipeline.run_complete_pipeline(
            news_count=args.news_count,
            max_events=args.max_events,
            language=args.language,
            save_results=not args.no_save and not args.preview_only,
            save_to_db=not args.no_database and not args.preview_only
        )
        
        if digest:
            # Показ результатов
            pipeline.show_digest_preview()
            
            if args.preview_only:
                print(f"\n📱 Telegram формат:")
                print("-" * 30)
                telegram_text = pipeline.summarizer.format_for_telegram(digest)
                print(telegram_text)
            
            # Статистика
            if args.stats:
                print(f"\n📊 СТАТИСТИКА")
                print("=" * 40)
                stats = pipeline.get_statistics()
                
                pipeline_stats = stats.get('pipeline', {})
                print(f"📰 Статей собрано: {pipeline_stats.get('articles_collected', 0)}")
                print(f"🗂️ Статей классифицировано: {pipeline_stats.get('articles_classified', 0)}")
                print(f"✅ Релевантных статей: {pipeline_stats.get('relevant_articles', 0)}")
                print(f"📊 Событий создано: {pipeline_stats.get('events_created', 0)}")
                print(f"🎯 Порог релевантности: {pipeline_stats.get('relevance_threshold', 0)}")
                
                # Статистика БД отключена для режима только чтения
            
            print(f"\n🎉 Пайплайн завершен успешно!")
            print(f"📊 Обработано {len(pipeline.raw_news)} статей")
            print(f"✅ Релевантных: {len(pipeline.relevant_news)}")
            print(f"🎯 Создан дайджест с {len(digest.get('items', []))} событиями")
            
            if not args.no_save and not args.preview_only:
                print(f"💾 Результаты сохранены в папку results/")
            
            # MongoDB сохранение отключено для режима только чтения
        else:
            print("❌ Пайплайн завершился с ошибкой")
            return 1
        
        # Закрытие соединений
        pipeline.close()
        
    except KeyboardInterrupt:
        print("\n⚠️ Пайплайн прерван пользователем")
        return 1
    except Exception as e:
        print(f"\n❌ Ошибка пайплайна: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 