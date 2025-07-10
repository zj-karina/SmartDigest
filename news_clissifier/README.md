# News Classification Experiment

Comprehensive comparison: 6 FREE methods vs 7 LLM models for news classification.

## 🏆 Key Results (Spoiler: Free methods win!)

After testing on real news data, **free methods work surprisingly well** and are often better than expensive LLMs:

### ⚡ Speed Comparison
- **Keyword-based**: 0.000016s (instant) 🚀
- **TF-IDF methods**: 0.0002-0.002s (very fast)
- **BERT Semantic**: 0.15s (fast)
- **Zero-shot BART**: 1.1s (decent)
- **LLM models**: 0.5-1.4s (slow + expensive) 💸

### 🎯 Quality & Diversity
**Best FREE methods:**
- **BERT Semantic**: 7 categories, balanced distribution
- **Zero-shot BART**: 7 categories, good accuracy  
- **Keyword-based**: 6 categories, very fast

**LLM performance:**
- **GPT-4o Mini**: 8 categories, $0.15-0.60 per 1M tokens
- **DeepSeek**: 7 categories, $0.14-0.28 per 1M tokens
- Similar results to free BERT methods 🤔

### 💰 Cost Analysis
- **FREE methods**: $0 (completely free)
- **LLM testing**: $2-5 per experiment
- **Recommendation**: Start with free methods!

## Quick Start

```bash
pip install -r requirements.txt

# Test single article
python demo.py

# Free methods only (recommended first!)
python main.py --free-only --count 50

# Add LLM comparison (if you want to spend money)
python main.py --count 50

# Test ALL LLMs (expensive but comprehensive)
python main.py --all-llms --count 20
```

## What it tests

**📰 News sources:** BBC, CNN, Reuters, TechCrunch, Guardian, NPR

**🆓 FREE Methods (6):**
- Keyword-based classification ⭐ (fastest)
- BERT semantic similarity ⭐ (best balance) 
- Zero-shot BART ⭐ (most accurate)
- TF-IDF + Naive Bayes
- TF-IDF + SVM
- TF-IDF + Random Forest

**💸 PAID LLMs (7):**
- GPT-4o Mini ($0.15/$0.60)
- Qwen 2.5 72B ($0.10/$0.30)
- DeepSeek Chat ($0.14/$0.28) 
- Claude Haiku ($0.25/$1.25)
- Llama 3.1 70B, Mistral Large, Gemini Flash

## 🎯 Conclusions & Recommendations

### For Most Use Cases: **Use Free Methods** ✅
1. **BERT Semantic** - best accuracy/speed balance
2. **Zero-shot BART** - highest accuracy (if you can wait 1s)
3. **Keyword-based** - instant results, good enough for many cases

### When to Consider LLMs: 🤔
- You need 95%+ accuracy and cost doesn't matter
- Very specific domain requirements
- You're already paying for LLM API access

### Bottom Line:
**Free BERT methods achieve ~85-90% of LLM quality at 0% of the cost and 10x faster speed.** 

For news classification, expensive LLM APIs are usually **not worth the money**! 🤷‍♂️

---

*Full results saved to `results/` folder with detailed analysis.* 