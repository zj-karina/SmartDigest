"""RSS news collector module."""

import feedparser
import requests
from typing import List, Dict, Any
from langdetect import detect
import time


class NewsCollector:
    """Collects news from RSS feeds."""
    
    def __init__(self):
        self.sources = {
            'cnn': 'http://rss.cnn.com/rss/edition.rss',
            'techcrunch': 'https://techcrunch.com/feed/',
            'guardian': 'https://www.theguardian.com/world/rss',
            'npr': 'https://feeds.npr.org/1001/rss.xml',
            'bbc_world': 'https://feeds.bbci.co.uk/news/world/rss.xml',  # Более стабильный BBC фид
            'ap_news': 'https://feeds.aponline.com/rss/news'  # AP News
        }
        
        # Настройки таймаута для feedparser
        feedparser.USER_AGENT = "SmartDigest/1.0"
    
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
            print(f"  🔍 Checking {source}...")
            try:
                start_time = time.time()
                
                # Используем requests с таймаутом, затем feedparser
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                
                feed = feedparser.parse(response.content)
                elapsed = time.time() - start_time
                
                print(f"    ✅ {source}: {len(feed.entries)} articles ({elapsed:.1f}s)")
                
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
                elapsed = time.time() - start_time if 'start_time' in locals() else 0
                print(f"    ❌ {source}: {e} ({elapsed:.1f}s)")
                continue
        
        print(f"✅ Collected {len(all_news)} articles from {len([s for s in self.sources if any(n['source'] == s for n in all_news)])} sources")
        return all_news[:max_total] 