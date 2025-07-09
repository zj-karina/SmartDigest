# News Classification Experiment

Comprehensive comparison: 6 FREE methods vs 7 LLM models for news classification.

## Quick Start

```bash
pip install -r requirements.txt

# Test single article
python demo.py

# Free methods only (6 different approaches)
python main.py --free-only --count 50

# With one LLM comparison
python main.py --count 50

# Test ALL LLMs (expensive!)
python main.py --all-llms --count 20
```

## What it tests

**📰 News sources:** BBC, CNN, Reuters, TechCrunch, Guardian, NPR

**🆓 FREE Methods (6):**
- Keyword-based classification
- BERT semantic similarity  
- Zero-shot BART
- TF-IDF + Naive Bayes
- TF-IDF + SVM
- TF-IDF + Random Forest

**💸 PAID LLMs (7):**
- Qwen 2.5 72B ($0.10/$0.30)
- DeepSeek Chat ($0.14/$0.28) 
- Claude Haiku ($0.25/$1.25)
- GPT-4o Mini ($0.15/$0.60)
- Llama 3.1 70B
- Mistral Large  
- Gemini Flash

## Results

Detailed comparison with speed rankings, category diversity, and accuracy estimates. Usually shows that free methods work surprisingly well! 🤷‍♂️

Files saved to `results/` folder with comprehensive analysis. 