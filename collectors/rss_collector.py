"""RSS news collector module."""

import feedparser
from typing import List, Dict, Any
from langdetect import detect


class NewsCollector:
    """Collects news from RSS feeds."""
    
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
        """
        Collect news from RSS sources.
        
        Args:
            max_total: Maximum number of articles to collect
            
        Returns:
            List of news articles
        """
        print(f"📰 Collecting up to {max_total} news articles...")
        
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
        
        print(f"✅ Collected {len(all_news)} articles")
        return all_news[:max_total] 