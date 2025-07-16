"""News summarization module."""

import requests
import json
import time
import os
from typing import List, Dict, Any, Optional
from datetime import datetime


class NewsSummarizer:
    """Creates concise summaries from news clusters using LLM models."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize summarizer.
        
        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Models by quality/price ratio (актуальные модели OpenRouter)
        self.models = {
            'fast': "openai/gpt-4o-mini",             # Fast & cheap
            'balanced': "anthropic/claude-3-haiku",   # Good quality/price
            'quality': "openai/gpt-4o",               # High quality
            'premium': "anthropic/claude-3-5-sonnet", # Premium quality
        }
        
        self.default_model = 'balanced'
    
    def _make_api_request(self, messages: List[Dict], model: str, max_tokens: int = 150) -> Optional[str]:
        """Make request to OpenRouter API."""
        if not self.api_key:
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo/SmartDigest",
            "X-Title": "SmartDigest News Summarizer"
        }
        
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.3,
            "top_p": 0.9,
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=data, timeout=30)
            
            if response.status_code != 200:
                print(f"❌ API Error {response.status_code}: {response.text[:200]}...")
                return None
            
            result = response.json()
            
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"❌ Unexpected API response format: {result}")
                return None
                
        except Exception as e:
            print(f"❌ API request failed: {e}")
            return None
    
    def summarize_cluster(self, 
                         cluster: Dict[str, Any], 
                         model_type: str = None,
                         language: str = 'english') -> Dict[str, Any]:
        """
        Summarize a news cluster.
        
        Args:
            cluster: News cluster with articles, main_title, keywords
            model_type: Model type ('fast', 'balanced', 'quality', 'premium')
            language: Summary language ('english', 'russian')
            
        Returns:
            Summarization result
        """
        model_type = model_type or self.default_model
        model = self.models.get(model_type, self.models[self.default_model])
        
        articles = cluster.get('articles', [])
        if not articles:
            return {
                'summary': 'No articles to summarize',
                'model': model,
                'success': False,
                'error': 'empty_cluster'
            }
        
        # Prepare cluster text
        articles_text = self._prepare_cluster_text(articles)
        
        # Create prompt based on language
        if language == 'russian':
            system_prompt = """You are a professional news editor. Create a brief digest from the provided news articles.

Requirements:
- Write a brief summary of the main event in 2-3 sentences in Russian
- Use only facts from the articles, don't add external information
- Avoid repetitions
- Make it engaging and clear"""

            user_prompt = f"""Create a brief digest from these news articles:

{articles_text}

Brief digest (2-3 sentences in Russian):"""
        else:
            system_prompt = """You are a professional news editor. Create a brief digest from the provided news articles.

Requirements:
- Write a brief summary of the main event in 2-3 sentences
- Use only facts from the articles, don't add external information
- Avoid repetitions
- Make it engaging and clear"""

            user_prompt = f"""Create a brief digest from these news articles:

{articles_text}

Brief digest (2-3 sentences):"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        start_time = time.time()
        summary = self._make_api_request(messages, model, max_tokens=200)
        api_time = time.time() - start_time
        
        if summary:
            return {
                'summary': summary,
                'model': model,
                'model_type': model_type,
                'language': language,
                'success': True,
                'api_time': api_time,
                'source_articles': len(articles),
                'keywords': cluster.get('keywords', []),
                'category': cluster.get('category', 'unknown')
            }
        else:
            # Fallback - simple summarization without API
            fallback_summary = self._create_fallback_summary(cluster, language)
            return {
                'summary': fallback_summary,
                'model': 'fallback',
                'success': True,  # Fallback суммаризация тоже успешна
                'error': 'api_not_available',
                'source_articles': len(articles),
                'keywords': cluster.get('keywords', []),
                'category': cluster.get('category', 'unknown')
            }
    
    def _prepare_cluster_text(self, articles: List[Dict[str, Any]], max_length: int = 2000) -> str:
        """Prepare cluster text for summarization."""
        combined_text = ""
        
        for i, article in enumerate(articles, 1):
            title = article.get('title', '')
            description = article.get('description', '')
            source = article.get('source', 'unknown')
            
            article_text = f"Article {i} ({source}):\nTitle: {title}\nDescription: {description}\n\n"
            
            # Check length limit
            if len(combined_text + article_text) > max_length:
                remaining = max_length - len(combined_text)
                if remaining > 100:
                    combined_text += article_text[:remaining] + "...\n"
                break
            
            combined_text += article_text
        
        return combined_text.strip()
    
    def _create_fallback_summary(self, cluster: Dict[str, Any], language: str = 'english') -> str:
        """Create simple summary without API (fallback)."""
        articles = cluster.get('articles', [])
        main_title = cluster.get('main_title', 'News')
        keywords = cluster.get('keywords', [])
        
        if not articles:
            return "No news available for summarization." if language == 'english' else "Нет новостей для суммаризации."
        
        sources = list(set([art.get('source', '') for art in articles if art.get('source')]))
        source_count = len(sources)
        
        if language == 'russian':
            if source_count <= 1:
                return f"Новость: {main_title}. Ключевые темы: {', '.join(keywords[:3]) if keywords else 'общие новости'}."
            else:
                return f"Событие освещают {source_count} источников: {main_title}. Основные темы: {', '.join(keywords[:3]) if keywords else 'разное'}."
        else:
            if source_count <= 1:
                return f"News: {main_title}. Key topics: {', '.join(keywords[:3]) if keywords else 'general news'}."
            else:
                return f"Event covered by {source_count} sources: {main_title}. Main topics: {', '.join(keywords[:3]) if keywords else 'various'}."
    
    def create_daily_digest(self, 
                           clusters: List[Dict[str, Any]], 
                           max_items: int = 5,
                           model_type: str = None,
                           language: str = 'english') -> Dict[str, Any]:
        """
        Create daily digest from top clusters.
        
        Args:
            clusters: List of clusters (sorted by importance)
            max_items: Maximum items in digest
            model_type: Model type
            language: Digest language
            
        Returns:
            Complete digest
        """
        top_clusters = clusters[:max_items]
        
        # Summarize each cluster
        print(f"📝 Creating summaries for {len(top_clusters)} clusters...")
        summaries = []
        
        for i, cluster in enumerate(top_clusters, 1):
            print(f"  {i}/{len(top_clusters)}: {cluster.get('main_title', 'No title')[:50]}...")
            result = self.summarize_cluster(cluster, model_type, language)
            summaries.append(result)
            time.sleep(0.5)  # Rate limiting
        
        # Build digest
        digest_items = []
        total_articles = 0
        
        for cluster, summary in zip(top_clusters, summaries):
            if summary['success'] and summary['summary']:
                item = {
                    'title': cluster.get('main_title', 'No title'),
                    'summary': summary['summary'],
                    'category': cluster.get('category', 'unknown'),
                    'source_count': cluster.get('size', 1),
                    'keywords': summary.get('keywords', []),
                    'sources': list(set([art.get('source', '') for art in cluster.get('articles', [])]))[:3]
                }
                digest_items.append(item)
                total_articles += cluster.get('size', 1)
        
        # Create digest header
        timestamp = datetime.now().strftime("%Y-%m-%d")
        title = f"📰 News Digest for {timestamp}" if language == 'english' else f"📰 Новостной дайджест за {timestamp}"
        footer = f"Total: {total_articles} articles from {len(digest_items)} events" if language == 'english' else f"Всего: {total_articles} статей из {len(digest_items)} событий"
        
        successful = len([r for r in summaries if r['success']])
        print(f"✅ Complete: {successful}/{len(summaries)} successful summaries")
        
        return {
            'title': title,
            'date': timestamp,
            'items': digest_items,
            'footer': footer,
            'language': language,
            'total_articles': total_articles,
            'total_events': len(digest_items)
        }
    
    def format_for_telegram(self, digest: Dict[str, Any]) -> str:
        """Format digest for Telegram."""
        lines = [f"*{digest['title']}*", ""]
        
        category_emoji = {
            'politics': '🏛️', 'business': '💼', 'technology': '💻',
            'sports': '⚽', 'entertainment': '🎬', 'health': '🏥',
            'science': '🔬', 'world': '🌍', 'crime': '🚨'
        }
        
        for i, item in enumerate(digest['items'], 1):
            emoji = category_emoji.get(item['category'], '📰')
            title = item['title'][:60] + ('...' if len(item['title']) > 60 else '')
            
            lines.append(f"{emoji} *{i}. {title}*")
            lines.append(f"{item['summary']}")
            
            if item['sources']:
                sources_text = ', '.join(item['sources'])
                lines.append(f"📡 _{sources_text}_")
            
            lines.append("")  # Empty line between news
        
        lines.append(f"_{digest['footer']}_")
        return "\n".join(lines) 