# SmartDigest

A news processing pipeline that collects RSS feeds and turns them into clean, organized digests.

## What it does

Takes messy news feeds and creates structured summaries:

1. **Grabs** news from BBC, CNN, Reuters, etc.
2. **Sorts** articles by topic (politics, tech, business...)  
3. **Groups** similar stories into events
4. **Writes** 2-3 sentence summaries of each event
5. **Outputs** clean digests ready for bots or websites

## Quick start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic demo (works without API keys)
python examples/quick_demo.py

# Full pipeline with 30 articles
python examples/full_pipeline.py --news-count 30
```

## Project structure

```
src/
├── collectors/      # RSS feed collection
├── classifiers/     # Article categorization  
├── clustering/      # Event grouping
├── summarization/   # LLM summaries
└── pipeline/        # Main workflow
```

## Performance

We tested 20+ classification methods on real news. Here's what works:

- **BGE-Large**: Best for production (80% accuracy, 0.03s per article)
- **Zero-shot BART**: Highest accuracy (90%) but slow (1.1s per article)  
- **E5-Large**: Good but overconfident
- **MiniLM**: Fast fallback option

Full pipeline processes 30 articles into 5 events in about 20 seconds.

## Configuration options

```python
from src.pipeline.digest_pipeline import DigestPipeline

# Fast setup for development
pipeline = DigestPipeline(
    classification_method='minilm',
    clustering_eps=0.4,
    summarization_model='fast'
)

# Production setup
pipeline = DigestPipeline(
    classification_method='bge_large',  # Best balance of speed/accuracy
    clustering_eps=0.35,               # Balanced clustering
    summarization_model='balanced'     # Good quality summaries
)

# Run it
digest = pipeline.run_complete_pipeline(
    news_count=50,
    max_events=7,
    language='english'
)
```

## API costs

For summarization, we use OpenRouter:

- **Qwen 7B**: $0.06 per 1M tokens (fast)
- **Qwen 14B**: $0.12 per 1M tokens (recommended)
- **Qwen 32B**: $0.24 per 1M tokens (high quality)
- **Claude Haiku**: $0.50 per 1M tokens (premium)

Typical cost: **$0.01-0.05 per digest**

## Examples

```bash
# Basic demo (no API needed)
python examples/quick_demo.py

# Custom pipeline
python examples/full_pipeline.py \
  --news-count 50 \
  --max-events 7 \
  --classification bge_large \
  --language english

# Preview without saving
python examples/full_pipeline.py --preview-only

# With OpenRouter API for better summaries
export OPENROUTER_API_KEY='your-key'
python examples/full_pipeline.py --model quality
```

## Output formats

- **JSON**: Full structured data
- **Telegram**: Markdown ready for bots
- **CSV**: Raw classification results

## What we learned

After testing extensively:

- **BGE models** beat everything for production use
- **Zero-shot BART** still wins on pure accuracy but too slow
- **LLM APIs** aren't worth the cost vs free models
- **TF-IDF methods** are broken (classify everything into one category)

The `results/` folder has detailed analysis if you're curious.

## Extending it

Easy to add:
- New news sources (edit `src/collectors/rss_collector.py`)
- Custom categories (edit `src/classifiers/news_classifier.py`)
- Different clustering algorithms 
- Telegram bot integration
- Scheduled digest generation

## Development notes

This started as one massive 900-line `main.py` file with tons of experimental classification methods. We've since:

- Split everything into logical modules
- Kept only the methods that actually work
- Made it production-ready
- Translated all Russian comments to English

The goal was building something useful, not an academic experiment.

## Need help?

Check `docs/pipeline.md` for technical details or look at the examples. 