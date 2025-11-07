# Data Collection Guide

This guide shows you how to get training data from the internet or create your own data.txt file.

## Option 1: Use the Download Script

### Quick Start

```bash
# Download Shakespeare text (recommended for testing)
python download_data.py --type shakespeare

# Create a sample data file
python download_data.py --type sample --output data/my_data.txt --samples 200

# Download Wikipedia article (requires: pip install wikipedia)
python download_data.py --type wikipedia --title "Artificial Intelligence" --output data/ai_article.txt
```

### Available Options

**Shakespeare Dataset:**
```bash
python download_data.py --type shakespeare
```
Downloads classic Shakespeare text - great for testing!

**Create Sample Data:**
```bash
python download_data.py --type sample --output data/my_data.txt --samples 100
```
Creates a file with sample sentences about ML/AI.

**Wikipedia Article:**
```bash
python download_data.py --type wikipedia --title "Machine Learning" --output data/ml_article.txt
```
Downloads a Wikipedia article (requires `pip install wikipedia`).

## Option 2: Manual Data Collection

### Method A: Create Your Own data.txt

1. **Create a text file:**
```bash
nano data/my_data.txt
# or
vim data/my_data.txt
```

2. **Add your text** (one sentence per line):
```
This is my first training sample.
This is my second training sample.
Add as many lines as you want.
```

3. **Save and use:**
```bash
python train.py --data data/my_data.txt
```

### Method B: Download from Public Datasets

**Shakespeare Text:**
```bash
curl -o data/shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

**Book Corpus Sample:**
```bash
# Download Project Gutenberg books
curl -o data/book.txt https://www.gutenberg.org/files/1342/1342-0.txt  # Pride and Prejudice
```

**News Articles:**
```bash
# Download news text
curl -o data/news.txt https://raw.githubusercontent.com/sunnysai12345/News_Summary/master/news_summary_more.csv
```

### Method C: Scrape Your Own Data

**From Wikipedia (Python):**
```python
import wikipedia

page = wikipedia.page("Machine Learning")
with open("data/ml_article.txt", "w") as f:
    f.write(page.content)
```

**From a Website:**
```python
import requests
from bs4 import BeautifulSoup

url = "https://example.com/article"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
text = soup.get_text()

with open("data/scraped.txt", "w") as f:
    f.write(text)
```

## Option 3: Use Existing Datasets

### Popular NLP Datasets

**WikiText-2:**
```bash
# Download WikiText-2
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip
unzip wikitext-2-v1.zip
# Use: wikitext-2/wiki.train.tokens
```

**OpenWebText Sample:**
```bash
# Download sample
curl -o data/openwebtext_sample.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

**BookCorpus:**
```bash
# Various book sources available
# Check: https://github.com/soskek/bookcorpus
```

## Data Format Requirements

Your `data.txt` file should:
- Have **one text sample per line**
- Use **UTF-8 encoding**
- Be **plain text** (no special formatting)

**Example format:**
```
This is the first training example.
This is the second training example.
Each line becomes one training sample.
```

**Good:**
```
Hello world!
This is a sentence.
Machine learning is cool.
```

**Bad:**
```
This is paragraph 1 with multiple sentences. This is sentence 2.
This is paragraph 2.
```

## Preprocessing Tips

1. **Clean your data:**
```python
import re

with open("raw_data.txt", "r") as f:
    text = f.read()

# Remove extra whitespace
text = re.sub(r'\s+', ' ', text)

# Split into sentences
sentences = text.split('.')

# Write one per line
with open("data/cleaned_data.txt", "w") as f:
    for sentence in sentences:
        if sentence.strip():
            f.write(sentence.strip() + '\n')
```

2. **Split long texts:**
```python
# If you have long texts, split them into sentences
text = "Long paragraph here. Another sentence. More text."
sentences = text.split('.')
for sentence in sentences:
    if sentence.strip():
        print(sentence.strip())
```

## Quick Test

1. **Create a small test file:**
```bash
cat > data/test.txt << EOF
Hello world!
This is a test.
Language models are cool.
EOF
```

2. **Train with it:**
```bash
python train.py --data data/test.txt --output ./checkpoints
```

## Recommended Data Sources

- **Small (for testing):** Shakespeare text, sample_data.txt
- **Medium (for training):** Wikipedia articles, news articles
- **Large (for serious training):** WikiText-2, BookCorpus, OpenWebText

## Next Steps

Once you have your data.txt file:

```bash
# Train your model
python train.py --data data/your_data.txt --output ./checkpoints

# Or use the sample data
python train.py --data data/sample_data.txt --output ./checkpoints
```

Happy training! 🚀

