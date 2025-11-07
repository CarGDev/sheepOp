# Tokenizer Improvements

## Overview

Based on the tokenization challenges discussed in the transcript, we've implemented an improved BPE (Byte Pair Encoding) tokenizer that addresses common issues with tokenization in large language models.

## Key Improvements

### 1. UTF-8 Byte-Level Encoding
- **What**: Tokenizer works at the byte level (0-255) instead of character level
- **Why**: Handles all Unicode characters consistently, regardless of language
- **Benefit**: Better support for non-English languages and special characters

### 2. GPT-4 Style Regex Pattern
- **What**: Improved regex pattern for splitting text into chunks
- **Improvements**:
  - Case-insensitive matching for contractions (`'s`, `'t`, `'ll`, etc.)
  - Better whitespace handling (groups multiple spaces for Python code)
  - Limits number merging to 1-3 digits (prevents overly long number tokens)
- **Benefit**: More consistent tokenization, especially for code and numbers

### 3. BPE Training Algorithm
- **What**: Full Byte Pair Encoding implementation
- **Features**:
  - `get_stats()`: Counts consecutive token pairs
  - `merge()`: Replaces frequent pairs with single tokens
  - Iterative training process
- **Benefit**: Creates efficient vocabulary that compresses common sequences

### 4. Special Token Handling
- **What**: Proper handling of special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`)
- **Features**:
  - Special tokens are excluded from BPE merging
  - EOS token stops decoding
  - Configurable special token set
- **Benefit**: Clean separation between data tokens and control tokens

### 5. Trailing Whitespace Detection
- **What**: Warns when text ends with trailing whitespace
- **Why**: Trailing spaces can cause poor tokenization (as seen in GPT-3.5 playground warnings)
- **Benefit**: Helps developers avoid tokenization issues

### 6. Better Python Code Handling
- **What**: Improved whitespace merging for indentation
- **Why**: GPT-2 had issues with Python code because each space was a separate token
- **Benefit**: More efficient tokenization of Python code (fewer tokens per file)

### 7. Number Tokenization Limits
- **What**: Limits number merging to 1-3 digits
- **Why**: Prevents creating tokens for very long number sequences
- **Benefit**: Better arithmetic performance (numbers are more consistently tokenized)

## Usage

### Basic Usage

```python
from data.example import SimpleTokenizer

# Create tokenizer (uses BPE by default)
tokenizer = SimpleTokenizer(use_bpe=True, vocab_size=50257)

# Encode text
tokens = tokenizer.encode("Hello world!")
print(tokens)  # [15496, 1917, 0]

# Decode tokens
text = tokenizer.decode(tokens)
print(text)  # "Hello world!"
```

### Training a Custom Tokenizer

```python
from data.example import BPETokenizer

# Create tokenizer
tokenizer = BPETokenizer(vocab_size=50257)

# Train on your corpus
texts = [
    "Your training text here...",
    "More training text...",
]

tokenizer.train(texts, num_merges=50000, verbose=True)

# Save trained tokenizer
tokenizer.save("merges.json", "vocab.json")
```

### Loading a Pre-trained Tokenizer

```python
from data.example import BPETokenizer

# Load saved tokenizer
tokenizer = BPETokenizer()
tokenizer.load("merges.json", "vocab.json")

# Use it
tokens = tokenizer.encode("Hello world!")
```

## Addressing Common Issues

### Issue: "Can't spell words well"
- **Cause**: Long tokens (like "defaultstyle" as single token)
- **Fix**: Better regex splitting prevents over-merging

### Issue: "Bad at arithmetic"
- **Cause**: Arbitrary number tokenization (sometimes 1 token, sometimes 2-3)
- **Fix**: Limits number merging to 1-3 digits for consistency

### Issue: "Python code inefficient"
- **Cause**: Each space is separate token (GPT-2 issue)
- **Fix**: Multiple spaces merge into single tokens

### Issue: "Non-English languages worse"
- **Cause**: Tokenizer trained primarily on English
- **Fix**: UTF-8 byte-level encoding handles all languages consistently

### Issue: "Trailing whitespace warning"
- **Cause**: Models see very few examples of trailing spaces
- **Fix**: Warning helps developers detect and fix the issue

### Issue: "Solid gold Magikarp" (untrained tokens)
- **Cause**: Tokenizer creates tokens for strings not in training data
- **Fix**: Proper validation and fallback handling for unknown tokens

## Backward Compatibility

The `SimpleTokenizer` class maintains backward compatibility:
- If `use_bpe=False`, uses character-level tokenization (old behavior)
- If `use_bpe=True` (default), uses new BPE tokenizer
- All existing code continues to work without changes

## Technical Details

### BPE Algorithm
1. Start with byte-level vocabulary (256 tokens)
2. Count consecutive token pairs in training data
3. Find most frequent pair
4. Merge pair into new token
5. Repeat until target vocabulary size reached

### Encoding Process
1. Split text using regex pattern
2. Convert each chunk to UTF-8 bytes
3. Apply BPE merges (greedy, left-to-right)
4. Return token IDs

### Decoding Process
1. Look up token IDs in vocabulary
2. Convert bytes back to UTF-8
3. Handle special tokens (EOS stops decoding)
4. Return decoded text

## References

Based on improvements discussed in:
- GPT-2 paper tokenization section
- GPT-4 tokenizer improvements
- Common tokenization challenges and solutions

