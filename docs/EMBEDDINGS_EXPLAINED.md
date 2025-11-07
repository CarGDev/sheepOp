# What are Embeddings? Step-by-Step Explanation

Complete step-by-step explanation of embeddings in transformer models: how words become numbers that capture meaning.

## Table of Contents

1. [The Problem Embeddings Solve](#11-the-problem-embeddings-solve)
2. [What is an Embedding?](#12-what-is-an-embedding)
3. [How Embeddings Work](#13-how-embeddings-work)
4. [Step-by-Step Example: Embedding "Hello"](#14-step-by-step-example-embedding-hello)
5. [Why Embeddings Matter](#15-why-embeddings-matter)
6. [Complete Example: Embedding Multiple Words](#16-complete-example-embedding-multiple-words)
7. [Visual Representation](#17-visual-representation)
8. [Key Takeaways](#18-key-takeaways)

---

## 1.1 The Problem Embeddings Solve

### The Challenge

**Computers understand numbers, not words.**

Your model receives:
- Input: `"Hello"` (a word, not a number)

But neural networks need:
- Input: Numbers (like `[0.1, -0.2, 0.3, ...]`)

### The Solution: Embeddings

**Embeddings convert words (or tokens) into numbers (vectors) that capture meaning.**

---

## 1.2 What is an Embedding?

### Simple Definition

An **embedding** is a numerical representation of a word or token that captures its semantic meaning.

**Think of it like this:**
- Each word gets a unique "address" in a high-dimensional space
- Similar words end up close together
- Different words are far apart

### Visual Analogy

Imagine a map where:
- Words are cities
- Similar words are nearby cities
- Different words are distant cities

```
          Semantic Space (2D visualization)

    "cat"    "dog"
      ●        ●
      
                "car"    "vehicle"
                  ●         ●
      
      "king"                  "queen"
        ●                         ●
```

In reality, embeddings use **512 dimensions** (not 2D), but the concept is the same.

---

## 1.3 How Embeddings Work

### Step 1: Vocabulary Mapping

**Create a mapping from words to numbers:**

```
Vocabulary:
"Hello" → Token ID: 72
"World" → Token ID: 87
"the"   → Token ID: 32
...
```

**Result:** Each word has a unique ID number

### Step 2: Embedding Matrix

**Create a matrix where each row represents a word:**

```
Embedding Matrix E:

        Dimension 0  Dimension 1  Dimension 2  ...  Dimension 511
Token 0  [  0.05   ,   -0.10   ,    0.20   , ...,     0.15   ]
Token 1  [ -0.08   ,    0.12   ,   -0.05   , ...,     0.08   ]
Token 2  [  0.10   ,   -0.15   ,    0.25   , ...,     0.12   ]
...
Token 72 [  0.10   ,   -0.20   ,    0.30   , ...,     0.05   ]  ← "Hello"
...
Token 87 [  0.15   ,   -0.18   ,    0.28   , ...,     0.10   ]  ← "World"
```

**Key Points:**
- Each row is a 512-dimensional vector
- Each row represents one token/word
- The values are learned during training

### Step 3: Lookup Operation

**When you need an embedding, look it up:**

```
Input: Token ID = 72 ("Hello")
    ↓
Lookup: E[72]
    ↓
Output: [0.10, -0.20, 0.30, ..., 0.05]  (512 numbers)
```

---

## 1.4 Step-by-Step Example: Embedding "Hello"

### Input

```
Word: "Hello"
Token ID: 72
```

### Process

**Step 1: Get Token ID**
```
"Hello" → Lookup in vocabulary → 72
```

**Step 2: Lookup Embedding**
```
E[72] = [0.10, -0.20, 0.30, 0.15, -0.05, ..., 0.05]
```

**Step 3: Result**
```
Embedding vector: [0.10, -0.20, 0.30, ..., 0.05]
Dimension: 512 numbers
Meaning: Numerical representation of "Hello"
```

### What These Numbers Mean

**Individual numbers don't mean much by themselves**, but **together** they represent:
- Semantic meaning (what the word means)
- Contextual relationships (how it relates to other words)
- Syntactic information (grammatical role)

**Key Insight:** The model learns these values during training to capture meaning.

---

## 1.5 Why Embeddings Matter

### Benefit 1: Continuous Space

**Before Embeddings:**
```
"Hello" = 72
"World" = 87
Distance: |72 - 87| = 15 (meaningless!)
```

**After Embeddings:**
```
"Hello" = [0.10, -0.20, 0.30, ...]
"World" = [0.15, -0.18, 0.28, ...]
Distance: Can measure similarity mathematically!
```

### Benefit 2: Semantic Relationships

**Similar words have similar embeddings:**

```
"cat"    ≈ [0.8, 0.2, 0.1, ...]
"dog"    ≈ [0.7, 0.3, 0.1, ...]  ← Similar to "cat"
"car"    ≈ [0.1, 0.9, 0.8, ...]  ← Different from "cat"
```

**Distance in embedding space = semantic similarity**

### Benefit 3: Mathematical Operations

**You can do math with embeddings:**

```
"king" - "man" + "woman" ≈ "queen"
```

This works because embeddings capture semantic relationships!

---

## 1.6 Complete Example: Embedding Multiple Words

### Input Sentence

```
"Hello World"
```

### Step-by-Step Processing

**Step 1: Tokenize**
```
"Hello" → Token ID: 72
"World" → Token ID: 87
```

**Step 2: Lookup Embeddings**
```
E[72] = [0.10, -0.20, 0.30, ..., 0.05]  (512 numbers)
E[87] = [0.15, -0.18, 0.28, ..., 0.10]  (512 numbers)
```

**Step 3: Stack Together**
```
Embedding Matrix:
[
  [0.10, -0.20, 0.30, ..., 0.05],  ← "Hello"
  [0.15, -0.18, 0.28, ..., 0.10]   ← "World"
]
Shape: [2, 512]
```

**Result:** Each word becomes a 512-dimensional vector

---

## 1.7 Visual Representation

### Embedding Space Visualization

```
2D Projection of 512-Dimensional Embedding Space:

     0.3 │                    "World"
         │                      ●
     0.2 │        "Hello"
         │           ●
     0.1 │
         │
     0.0 ├───────────────────────────
         │
    -0.1 │
         │
    -0.2 │
         │
    -0.3 │
```

**Reality:** Embeddings exist in 512-dimensional space, but we can visualize them in 2D or 3D projections.

### Similarity Visualization

```
Word Similarities (distance in embedding space):

"cat" ──── 0.15 distance ──── "dog"   (similar)
 "cat" ──── 2.5 distance ──── "car"   (different)
"king" ──── 0.8 distance ──── "queen" (related)
```

---

## 1.8 Key Takeaways: Embeddings

✅ **Embeddings convert words to numbers**  
✅ **Each word becomes a vector (list of numbers)**  
✅ **Similar words have similar vectors**  
✅ **Enables mathematical operations on words**  
✅ **Learned during training to capture meaning**

---

*This document provides a step-by-step explanation of embeddings, the fundamental component that converts words into numerical representations in transformer models.*

