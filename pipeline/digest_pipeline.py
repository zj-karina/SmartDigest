"""Main news digest pipeline."""

import os
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

from ..collectors.rss_collector import NewsCollector
from ..classifiers.news_classifier import NewsClassifier
from ..clustering.news_clusterer import NewsClusterer
from ..summarization.news_summarizer import NewsSummarizer


class DigestPipeline:
    """Complete news processing pipeline: collect → classify → cluster → summarize."""
    
    def __init__(self, 
                 openrouter_api_key: Optional[str] = None,
                 classification_method: str = 'bge_large',
                 clustering_eps: float = 0.35,
                 summarization_model: str = 'balanced'):
        """
        Initialize pipeline.
        
        Args:
            openrouter_api_key: API key for summarization
            classification_method: Classification method ('bge_large', 'e5_large', etc.)
            clustering_eps: Clustering threshold (lower = stricter)
            summarization_model: LLM model type ('fast', 'balanced', 'quality', 'premium')
        """
        print("🚀 Initializing News Digest Pipeline")
        
        # Components
        self.collector = NewsCollector()
        self.classifier = NewsClassifier(method=classification_method)
        self.clusterer = NewsClusterer()
        self.summarizer = NewsSummarizer(openrouter_api_key)
        
        # Parameters
        self.classification_method = classification_method
        self.clustering_eps = clustering_eps
        self.summarization_model = summarization_model
        
        # State
        self.raw_news = []
        self.classified_news = []
        self.clusters = {}
        self.digest = {}
        
        print(f"✅ Pipeline ready")
        print(f"   📊 Classification: {classification_method}")
        print(f"   🔍 Clustering: eps={clustering_eps}")
        print(f"   📝 Summarization: {summarization_model}")
    
    def run_complete_pipeline(self, 
                             news_count: int = 50,
                             max_events: int = 7,
                             language: str = 'english',
                             save_results: bool = True) -> Dict[str, Any]:
        """
        Run complete pipeline.
        
        Args:
            news_count: Number of news articles to collect
            max_events: Maximum events in digest
            language: Digest language ('english', 'russian')
            save_results: Whether to save intermediate results
            
        Returns:
            Final digest
        """
        pipeline_start = time.time()
        
        print("🚀 Running Complete Pipeline")
        print("=" * 40)
        print(f"Parameters:")
        print(f"   📰 News count: {news_count}")
        print(f"   📊 Max events: {max_events}")
        print(f"   🌍 Language: {language}")
        
        try:
            # Step 1: Collect news
            print(f"\n📰 STEP 1: Collection")
            print("-" * 25)
            
            start_time = time.time()
            self.raw_news = self.collector.collect_news(news_count)
            collection_time = time.time() - start_time
            
            if not self.raw_news:
                print("❌ No news collected!")
                return {}
            
            print(f"✅ Collected {len(self.raw_news)} articles in {collection_time:.1f}s")
            
            # Step 2: Classification
            print(f"\n🗂️ STEP 2: Classification")
            print("-" * 25)
            
            start_time = time.time()
            self.classified_news = self.classifier.classify_batch(self.raw_news)
            classification_time = time.time() - start_time
            
            from collections import Counter
            category_stats = Counter([news.get('category', 'unknown') for news in self.classified_news])
            
            print(f"✅ Classified in {classification_time:.1f}s")
            print(f"📊 Categories: {dict(category_stats)}")
            
            # Step 3: Clustering
            print(f"\n🔍 STEP 3: Clustering")
            print("-" * 25)
            
            start_time = time.time()
            self.clusters = self.clusterer.cluster_by_category(
                self.classified_news, 
                eps=self.clustering_eps, 
                min_samples=2
            )
            clustering_time = time.time() - start_time
            
            # Get top clusters for summarization
            top_clusters = self.clusterer.get_top_clusters(
                min_size=2, 
                max_clusters=max_events
            )
            
            print(f"✅ Clustered in {clustering_time:.1f}s")
            print(f"🎯 Found {len(top_clusters)} significant events")
            
            if not top_clusters:
                print("❌ No significant events found for digest")
                return {}
            
            # Step 4: Summarization
            print(f"\n📝 STEP 4: Summarization")
            print("-" * 25)
            
            if not self.summarizer.api_key:
                print("⚠️ No API key - using fallback summaries")
            
            start_time = time.time()
            self.digest = self.summarizer.create_daily_digest(
                top_clusters,
                max_items=max_events,
                model_type=self.summarization_model,
                language=language
            )
            summarization_time = time.time() - start_time
            
            total_time = time.time() - pipeline_start
            
            # Final stats
            print(f"\n🎉 PIPELINE COMPLETE")
            print("=" * 40)
            print(f"⏱️ Total time: {total_time:.1f}s")
            print(f"📰 Processed: {len(self.raw_news)} articles")
            print(f"📊 Found: {len(self.digest.get('items', []))} events")
            print(f"🎯 Digest: {self.digest.get('title', 'Ready')}")
            
            # Save results
            if save_results:
                self.save_results()
            
            return self.digest
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def save_results(self) -> Dict[str, str]:
        """Save all pipeline results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        print(f"\n💾 Saving results...")
        
        try:
            os.makedirs('results', exist_ok=True)
            
            # Save classification results
            if self.classified_news:
                import pandas as pd
                df = pd.DataFrame(self.classified_news)
                classification_file = f"results/classification_{timestamp}.csv"
                df.to_csv(classification_file, index=False)
                saved_files['classification'] = classification_file
                print(f"   📊 Classification: {classification_file}")
            
            # Save clusters
            if self.clusters:
                clusters_file = f"results/clusters_{timestamp}.json"
                clean_clusters = {}
                
                for category, clusters in self.clusters.items():
                    clean_clusters[category] = []
                    for cluster in clusters:
                        clean_cluster = {
                            'cluster_id': cluster['cluster_id'],
                            'size': cluster['size'],
                            'main_title': cluster['main_title'],
                            'main_source': cluster.get('main_source', ''),
                            'is_noise': cluster.get('is_noise', False),
                            'keywords': cluster.get('keywords', []),
                            'articles': [
                                {
                                    'title': art['title'],
                                    'source': art.get('source', ''),
                                    'category': art.get('category', '')
                                } for art in cluster['articles']
                            ]
                        }
                        clean_clusters[category].append(clean_cluster)
                
                with open(clusters_file, 'w', encoding='utf-8') as f:
                    json.dump(clean_clusters, f, indent=2, ensure_ascii=False)
                saved_files['clusters'] = clusters_file
                print(f"   🔍 Clusters: {clusters_file}")
            
            # Save digest
            if self.digest:
                digest_file = f"results/digest_{timestamp}.json"
                with open(digest_file, 'w', encoding='utf-8') as f:
                    json.dump(self.digest, f, indent=2, ensure_ascii=False)
                saved_files['digest'] = digest_file
                print(f"   📝 Digest: {digest_file}")
                
                # Save Telegram format
                telegram_file = f"results/digest_telegram_{timestamp}.txt"
                telegram_text = self.summarizer.format_for_telegram(self.digest)
                with open(telegram_file, 'w', encoding='utf-8') as f:
                    f.write(telegram_text)
                saved_files['telegram'] = telegram_file
                print(f"   📱 Telegram: {telegram_file}")
            
            print(f"✅ All results saved")
            return saved_files
            
        except Exception as e:
            print(f"❌ Save failed: {e}")
            return {}
    
    def show_digest_preview(self):
        """Show digest preview."""
        if not self.digest:
            print("❌ No digest available")
            return
        
        print(f"\n📰 DIGEST PREVIEW")
        print("=" * 40)
        print(f"{self.digest['title']}")
        print("-" * 25)
        
        for i, item in enumerate(self.digest.get('items', []), 1):
            category_emoji = {
                'politics': '🏛️', 'business': '💼', 'technology': '💻',
                'sports': '⚽', 'entertainment': '🎬', 'health': '🏥',
                'science': '🔬', 'world': '🌍', 'crime': '🚨'
            }
            emoji = category_emoji.get(item['category'], '📰')
            
            title = item['title'][:60] + ('...' if len(item['title']) > 60 else '')
            print(f"\n{emoji} {i}. {title}")
            print(f"   {item['summary']}")
            print(f"   📊 {item['source_count']} articles | 📡 {', '.join(item['sources'][:2])}")
        
        print(f"\n{self.digest['footer']}")
    
    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """Get pipeline component statistics."""
        return {
            'available_classification_methods': NewsClassifier.get_available_methods(),
            'default_parameters': {
                'classification_method': 'bge_large',
                'clustering_eps': 0.35,
                'summarization_model': 'balanced'
            },
            'supported_languages': ['english', 'russian'],
            'output_formats': ['json', 'telegram', 'csv']
        } 