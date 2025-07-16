"""News clustering module."""

import numpy as np
from typing import List, Dict, Any
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')


class NewsClusterer:
    """Groups similar news articles into event clusters."""
    
    def __init__(self, model_name: str = 'BAAI/bge-large-en-v1.5'):
        """
        Initialize clusterer.
        
        Args:
            model_name: Model for embeddings (using best performer from tests)
        """
        print(f"🔄 Loading clustering model {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        self.clusters = {}
        print(f"✅ Model ready")
    
    def get_text_embedding(self, news_item: Dict[str, Any]) -> np.ndarray:
        """Get embedding for a news article."""
        text = f"{news_item.get('title', '')}. {news_item.get('description', '')}"
        text = text.strip()
        
        if not text:
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension())
        
        return self.embedding_model.encode(text)
    
    def cluster_by_category(self, 
                           classified_news: List[Dict[str, Any]], 
                           eps: float = 0.35, 
                           min_samples: int = 2) -> Dict[str, List[Dict]]:
        """
        Cluster news within each category.
        
        Args:
            classified_news: List of classified news articles
            eps: DBSCAN distance threshold (lower = stricter clustering)
            min_samples: Minimum articles per cluster
            
        Returns:
            Dictionary {category: [cluster1, cluster2, ...]}
        """
        print("🔍 Clustering news by category...")
        
        # Group by category
        by_category = defaultdict(list)
        for news in classified_news:
            category = news.get('category', 'unknown')
            by_category[category].append(news)
        
        all_clusters = {}
        
        for category, news_list in by_category.items():
            print(f"  📊 Processing {len(news_list)} articles in '{category}'")
            
            if len(news_list) < min_samples:
                # Too few articles for clustering
                all_clusters[category] = [{
                    'cluster_id': 0,
                    'articles': news_list,
                    'size': len(news_list),
                    'main_title': news_list[0]['title'] if news_list else 'No title',
                    'keywords': self._extract_keywords(news_list),
                    'is_noise': True
                }]
                continue
            
            # Get embeddings
            embeddings = np.array([self.get_text_embedding(news) for news in news_list])
            
            # DBSCAN clustering with cosine distance
            distance_matrix = cosine_distances(embeddings)
            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distance_matrix)
            
            # Group results
            category_clusters = []
            labels = clustering.labels_
            
            for cluster_id in set(labels):
                mask = (labels == cluster_id)
                cluster_articles = [news_list[i] for i in range(len(news_list)) if mask[i]]
                
                if cluster_articles:
                    # Choose main article (longest text)
                    main_article = max(cluster_articles, 
                                     key=lambda x: len(x.get('description', '') + x.get('title', '')))
                    
                    cluster_info = {
                        'cluster_id': int(cluster_id),
                        'articles': cluster_articles,
                        'size': len(cluster_articles),
                        'main_title': main_article['title'],
                        'main_source': main_article.get('source', 'unknown'),
                        'is_noise': cluster_id == -1,  # DBSCAN noise
                        'keywords': self._extract_keywords(cluster_articles)
                    }
                    category_clusters.append(cluster_info)
            
            # Sort by size (bigger events first)
            category_clusters.sort(key=lambda x: x['size'], reverse=True)
            all_clusters[category] = category_clusters
            
            large_clusters = len([c for c in category_clusters if c['size'] >= 2 and not c['is_noise']])
            print(f"    ✅ Found {large_clusters} significant clusters")
        
        self.clusters = all_clusters
        return all_clusters
    
    def _extract_keywords(self, articles: List[Dict[str, Any]], top_k: int = 5) -> List[str]:
        """Extract keywords from article cluster."""
        all_text = " ".join([f"{art.get('title', '')} {art.get('description', '')}" for art in articles])
        
        words = all_text.lower().split()
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 
            'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'
        }
        
        filtered_words = [w for w in words if len(w) > 3 and w not in stop_words and w.isalpha()]
        
        if not filtered_words:
            return ['news']
        
        word_counts = Counter(filtered_words)
        return [word for word, count in word_counts.most_common(top_k)]
    
    def get_top_clusters(self, 
                        category: str = None, 
                        min_size: int = 2, 
                        max_clusters: int = 10) -> List[Dict[str, Any]]:
        """
        Get top clusters for digest.
        
        Args:
            category: Specific category (None = all categories)
            min_size: Minimum cluster size
            max_clusters: Maximum clusters to return
            
        Returns:
            List of top clusters
        """
        all_clusters = []
        
        categories = [category] if category else self.clusters.keys()
        
        for cat in categories:
            if cat in self.clusters:
                for cluster in self.clusters[cat]:
                    if cluster['size'] >= min_size and not cluster.get('is_noise', False):
                        cluster_with_category = cluster.copy()
                        cluster_with_category['category'] = cat
                        all_clusters.append(cluster_with_category)
        
        # Sort by size and return top
        all_clusters.sort(key=lambda x: x['size'], reverse=True)
        return all_clusters[:max_clusters] 