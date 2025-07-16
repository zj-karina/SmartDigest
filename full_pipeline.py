#!/usr/bin/env python3
"""
Full pipeline example with production-ready settings.
Includes API usage, proper error handling, and result saving.
"""

import sys
import os
import argparse

from pipeline.digest_pipeline import DigestPipeline


def main():
    parser = argparse.ArgumentParser(description='SmartDigest - Full Pipeline')
    parser.add_argument('--news-count', type=int, default=50, 
                       help='Number of news articles to process (default: 50)')
    parser.add_argument('--max-events', type=int, default=7,
                       help='Maximum events in digest (default: 7)')
    parser.add_argument('--language', choices=['english', 'russian'], default='english',
                       help='Digest language (default: english)')
    parser.add_argument('--classification', default='bge_large',
                       help='Classification method (default: bge_large)')
    parser.add_argument('--clustering-eps', type=float, default=0.35,
                       help='Clustering threshold (default: 0.35)')
    parser.add_argument('--model', choices=['fast', 'balanced', 'quality', 'premium'], 
                       default='balanced', help='Summarization model (default: balanced)')
    parser.add_argument('--api-key', help='OpenRouter API key (or set OPENROUTER_API_KEY)')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save results')
    parser.add_argument('--preview-only', action='store_true', help='Show preview only')
    
    args = parser.parse_args()
    
    print("🚀 SmartDigest - Full Pipeline")
    print("=" * 50)
    print(f"Settings:")
    print(f"  📰 News count: {args.news_count}")
    print(f"  📊 Max events: {args.max_events}")
    print(f"  🌍 Language: {args.language}")
    print(f"  🗂️ Classification: {args.classification}")
    print(f"  🔍 Clustering eps: {args.clustering_eps}")
    print(f"  📝 Model: {args.model}")
    
    # Check API key
    api_key = args.api_key or os.getenv('OPENROUTER_API_KEY')
    if api_key:
        print(f"  🔑 API: OpenRouter (enabled)")
    else:
        print(f"  ⚠️ API: None (fallback summaries)")
    
    print("=" * 50)
    
    try:
        # Create pipeline with specified settings
        pipeline = DigestPipeline(
            openrouter_api_key=api_key,
            classification_method=args.classification,
            clustering_eps=args.clustering_eps,
            summarization_model=args.model
        )
        
        # Run pipeline
        digest = pipeline.run_complete_pipeline(
            news_count=args.news_count,
            max_events=args.max_events,
            language=args.language,
            save_results=not args.no_save and not args.preview_only
        )
        
        if digest:
            # Show results
            pipeline.show_digest_preview()
            
            if args.preview_only:
                print(f"\n📱 Telegram format:")
                print("-" * 30)
                telegram_text = pipeline.summarizer.format_for_telegram(digest)
                print(telegram_text)
            
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"📊 Processed {len(pipeline.raw_news)} articles")
            print(f"🎯 Created digest with {len(digest.get('items', []))} events")
            
            if not args.no_save and not args.preview_only:
                print(f"💾 Results saved to results/ directory")
        else:
            print("❌ Pipeline failed")
            return 1
        
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Pipeline error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 