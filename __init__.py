"""
SmartDigest - Intelligent News Processing Pipeline

A modern news classification, clustering, and summarization system.
"""

__version__ = "1.0.0"
__author__ = "SmartDigest Team"

from .collectors.rss_collector import NewsCollector
from .classifiers.news_classifier import NewsClassifier
from .clustering.news_clusterer import NewsClusterer  
from .summarization.news_summarizer import NewsSummarizer
from .pipeline.digest_pipeline import DigestPipeline

__all__ = [
    'NewsCollector',
    'NewsClassifier', 
    'NewsClusterer',
    'NewsSummarizer',
    'DigestPipeline'
] 