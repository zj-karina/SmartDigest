#!/usr/bin/env python3
"""
Quick demo - classify one news article with different methods
"""

from main import FreeClassifier, LLMClassifier
import os

def demo():
    # Test news
    news = "Apple announces new AI-powered iPhone with revolutionary neural processing capabilities and enhanced machine learning features for photography"
    
    print("🗞️  Test news:")
    print(f"   {news}")
    print()
    
    # Test Free methods
    print("🆓 FREE Methods:")
    free_classifier = FreeClassifier()
    
    # Quick methods (individual)
    keyword_result = free_classifier.classify_keyword_based(news)
    print(f"  Keyword:   {keyword_result['category']} (conf: {keyword_result['confidence']:.3f}, time: {keyword_result['time']:.3f}s)")
    
    semantic_result = free_classifier.classify_semantic(news)
    print(f"  Semantic:  {semantic_result['category']} (conf: {semantic_result['confidence']:.3f}, time: {semantic_result['time']:.3f}s)")
    
    zero_shot_result = free_classifier.classify_zero_shot(news)
    print(f"  Zero-shot: {zero_shot_result['category']} (conf: {zero_shot_result['confidence']:.3f}, time: {zero_shot_result['time']:.3f}s)")
    
    # Batch method example (TF-IDF)
    print("\n  🔹 TF-IDF Methods (batch processing):")
    tfidf_nb_results = free_classifier.classify_tfidf_nb([news])
    print(f"  TF-IDF+NB: {tfidf_nb_results[0]['category']} (conf: {tfidf_nb_results[0]['confidence']:.3f})")
    
    # Test LLM (paid, if key available)
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key:
        print("\n💸 LLM Methods (PAID):")
        llm_classifier = LLMClassifier(api_key)
        
        # Test just one model for demo
        llm_result = llm_classifier.classify_single(news, llm_classifier.models['qwen_72b'])
        print(f"  Qwen 2.5:  {llm_result['category']} (conf: {llm_result['confidence']:.3f}, time: {llm_result['time']:.3f}s)")
        
        print(f"\n  Available LLM models: {list(llm_classifier.models.keys())}")
        print("  Use --all-llms flag to test all models (expensive!)")
    else:
        print("\n⚠️  LLM test skipped (no API key)")
        print("   Set OPENROUTER_API_KEY to test paid methods")
    
    print(f"\n🚀 For full comparison run:")
    print(f"   python main.py --free-only --count 30")
    print(f"   python main.py --count 30  # with LLMs")
    print(f"   python main.py --all-llms --count 20  # test all LLMs ($$)")

if __name__ == "__main__":
    demo() 